"""
负载均衡器 - 实现智能负载分发和动态调整
支持多种负载均衡策略和自适应调整
"""

import time
import threading
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
from collections import deque, defaultdict
import random

from .gpu_manager import GPUManager, GPUMetrics, GPUStatus

logger = logging.getLogger(__name__)


class LoadBalancingStrategy(Enum):
    """负载均衡策略"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_RESPONSE_TIME = "least_response_time"
    RESOURCE_BASED = "resource_based"
    ADAPTIVE = "adaptive"


@dataclass
class LoadBalancerConfig:
    """负载均衡器配置"""
    strategy: LoadBalancingStrategy = LoadBalancingStrategy.ADAPTIVE
    health_check_interval: float = 5.0
    rebalance_interval: float = 30.0
    enable_auto_scaling: bool = True
    max_queue_size: int = 1000
    request_timeout: float = 30.0
    enable_circuit_breaker: bool = True
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0


@dataclass
class RequestMetrics:
    """请求指标"""
    device_id: int
    start_time: float
    end_time: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    
    @property
    def response_time(self) -> float:
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time


class CircuitBreaker:
    """熔断器"""
    
    def __init__(self, device_id: int, threshold: int = 5, timeout: float = 60.0):
        self.device_id = device_id
        self.threshold = threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
    def call(self, func: Callable, *args, **kwargs):
        """执行调用并处理熔断"""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
                logger.info(f"Circuit breaker for GPU {self.device_id} changed to HALF_OPEN")
            else:
                raise Exception(f"Circuit breaker OPEN for GPU {self.device_id}")
        
        try:
            result = func(*args, **kwargs)
            
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
                logger.info(f"Circuit breaker for GPU {self.device_id} changed to CLOSED")
            
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.threshold:
                self.state = "OPEN"
                logger.warning(f"Circuit breaker for GPU {self.device_id} changed to OPEN")
            
            raise e
    
    def is_available(self) -> bool:
        """检查是否可用"""
        return self.state != "OPEN"


class LoadBalancer:
    """
    负载均衡器
    实现智能的请求分发和负载管理
    """
    
    def __init__(self, gpu_manager: GPUManager, config: LoadBalancerConfig):
        self.gpu_manager = gpu_manager
        self.config = config
        
        # 策略相关状态
        self.current_device_index = 0  # 轮询索引
        self.device_weights = {}  # 设备权重
        self.device_connections = defaultdict(int)  # 设备连接数
        self.device_response_times = defaultdict(lambda: deque(maxlen=100))  # 响应时间历史
        
        # 请求队列和指标
        self.request_queue = deque(maxlen=config.max_queue_size)
        self.active_requests = {}  # request_id -> RequestMetrics
        self.completed_requests = deque(maxlen=1000)
        
        # 熔断器
        self.circuit_breakers = {}
        if config.enable_circuit_breaker:
            for device_id in gpu_manager.gpu_configs.keys():
                self.circuit_breakers[device_id] = CircuitBreaker(
                    device_id, config.circuit_breaker_threshold, config.circuit_breaker_timeout
                )
        
        # 后台任务
        self.is_running = False
        self.health_check_thread = None
        self.rebalance_thread = None
        
        # 统计信息
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "strategy_switches": 0,
            "rebalance_events": 0,
            "circuit_breaker_trips": 0
        }
        
        # 初始化权重
        self._initialize_weights()
        
        logger.info(f"LoadBalancer initialized with strategy: {config.strategy.value}")
    
    def start(self):
        """启动负载均衡器"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # 启动健康检查
        self.health_check_thread = threading.Thread(
            target=self._health_check_loop, daemon=True
        )
        self.health_check_thread.start()
        
        # 启动重平衡
        self.rebalance_thread = threading.Thread(
            target=self._rebalance_loop, daemon=True
        )
        self.rebalance_thread.start()
        
        logger.info("LoadBalancer started")
    
    def stop(self):
        """停止负载均衡器"""
        self.is_running = False
        
        if self.health_check_thread:
            self.health_check_thread.join(timeout=5.0)
        
        if self.rebalance_thread:
            self.rebalance_thread.join(timeout=5.0)
        
        logger.info("LoadBalancer stopped")
    
    def select_device(self, request_id: str, expert_id: Optional[int] = None) -> Optional[int]:
        """
        选择设备
        
        Args:
            request_id: 请求ID
            expert_id: 专家ID（如果指定）
            
        Returns:
            Optional[int]: 选择的设备ID
        """
        try:
            self.stats["total_requests"] += 1
            
            # 如果指定了专家，检查其分配
            if expert_id is not None:
                allocated_device = self.gpu_manager.expert_allocations.get(expert_id)
                if allocated_device is not None:
                    if self._is_device_available(allocated_device):
                        return self._select_device_with_circuit_breaker(allocated_device, request_id)
            
            # 根据策略选择设备
            if self.config.strategy == LoadBalancingStrategy.ROUND_ROBIN:
                device_id = self._round_robin_select()
            elif self.config.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
                device_id = self._least_connections_select()
            elif self.config.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
                device_id = self._weighted_round_robin_select()
            elif self.config.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
                device_id = self._least_response_time_select()
            elif self.config.strategy == LoadBalancingStrategy.RESOURCE_BASED:
                device_id = self._resource_based_select()
            elif self.config.strategy == LoadBalancingStrategy.ADAPTIVE:
                device_id = self._adaptive_select()
            else:
                device_id = self._round_robin_select()
            
            if device_id is not None:
                return self._select_device_with_circuit_breaker(device_id, request_id)
            
            return None
            
        except Exception as e:
            logger.error(f"Device selection failed: {e}")
            self.stats["failed_requests"] += 1
            return None
    
    def _select_device_with_circuit_breaker(self, device_id: int, request_id: str) -> Optional[int]:
        """使用熔断器选择设备"""
        if self.config.enable_circuit_breaker:
            circuit_breaker = self.circuit_breakers.get(device_id)
            if circuit_breaker and not circuit_breaker.is_available():
                logger.warning(f"Device {device_id} unavailable due to circuit breaker")
                return self._fallback_device_selection(request_id)
        
        # 记录请求开始
        self._start_request(request_id, device_id)
        return device_id
    
    def _fallback_device_selection(self, request_id: str) -> Optional[int]:
        """备用设备选择"""
        available_devices = [
            device_id for device_id in self.gpu_manager.gpu_configs.keys()
            if self._is_device_available(device_id)
        ]
        
        if available_devices:
            device_id = random.choice(available_devices)
            self._start_request(request_id, device_id)
            return device_id
        
        return None
    
    def _round_robin_select(self) -> Optional[int]:
        """轮询选择"""
        available_devices = [
            device_id for device_id in self.gpu_manager.gpu_configs.keys()
            if self._is_device_available(device_id)
        ]
        
        if not available_devices:
            return None
        
        device_id = available_devices[self.current_device_index % len(available_devices)]
        self.current_device_index += 1
        
        return device_id
    
    def _least_connections_select(self) -> Optional[int]:
        """最少连接选择"""
        available_devices = [
            device_id for device_id in self.gpu_manager.gpu_configs.keys()
            if self._is_device_available(device_id)
        ]
        
        if not available_devices:
            return None
        
        return min(available_devices, key=lambda d: self.device_connections[d])
    
    def _weighted_round_robin_select(self) -> Optional[int]:
        """加权轮询选择"""
        available_devices = [
            device_id for device_id in self.gpu_manager.gpu_configs.keys()
            if self._is_device_available(device_id)
        ]
        
        if not available_devices:
            return None
        
        # 根据权重选择
        total_weight = sum(self.device_weights.get(d, 1.0) for d in available_devices)
        if total_weight <= 0:
            return available_devices[0]
        
        rand_val = random.uniform(0, total_weight)
        current_weight = 0
        
        for device_id in available_devices:
            current_weight += self.device_weights.get(device_id, 1.0)
            if rand_val <= current_weight:
                return device_id
        
        return available_devices[-1]
    
    def _least_response_time_select(self) -> Optional[int]:
        """最短响应时间选择"""
        available_devices = [
            device_id for device_id in self.gpu_manager.gpu_configs.keys()
            if self._is_device_available(device_id)
        ]
        
        if not available_devices:
            return None
        
        def get_avg_response_time(device_id):
            times = self.device_response_times[device_id]
            return sum(times) / len(times) if times else 0.0
        
        return min(available_devices, key=get_avg_response_time)
    
    def _resource_based_select(self) -> Optional[int]:
        """基于资源的选择"""
        best_device = None
        best_score = float('-inf')
        
        for device_id in self.gpu_manager.gpu_configs.keys():
            if not self._is_device_available(device_id):
                continue
            
            metrics = self.gpu_manager.get_device_metrics(device_id)
            if metrics is None:
                continue
            
            # 计算资源分数
            memory_score = (1.0 - metrics.memory_utilization) * 100
            gpu_score = (1.0 - metrics.gpu_utilization) * 100
            connection_score = max(0, (10 - self.device_connections[device_id]) / 10) * 50
            
            total_score = memory_score + gpu_score + connection_score
            
            if total_score > best_score:
                best_score = total_score
                best_device = device_id
        
        return best_device
    
    def _adaptive_select(self) -> Optional[int]:
        """自适应选择"""
        # 根据当前系统状态动态选择策略
        total_load = sum(self.device_connections.values())
        
        if total_load < 10:
            # 低负载时使用轮询
            return self._round_robin_select()
        elif total_load < 50:
            # 中等负载时使用资源基础选择
            return self._resource_based_select()
        else:
            # 高负载时使用最少连接
            return self._least_connections_select()
    
    def _is_device_available(self, device_id: int) -> bool:
        """检查设备是否可用"""
        metrics = self.gpu_manager.get_device_metrics(device_id)
        if metrics is None:
            return False
        
        if metrics.status in [GPUStatus.ERROR, GPUStatus.MAINTENANCE]:
            return False
        
        if self.config.enable_circuit_breaker:
            circuit_breaker = self.circuit_breakers.get(device_id)
            if circuit_breaker and not circuit_breaker.is_available():
                return False
        
        return True
    
    def _start_request(self, request_id: str, device_id: int):
        """开始请求"""
        self.device_connections[device_id] += 1
        self.active_requests[request_id] = RequestMetrics(
            device_id=device_id,
            start_time=time.time()
        )
    
    def complete_request(self, request_id: str, success: bool = True, error_message: str = None):
        """完成请求"""
        if request_id not in self.active_requests:
            return
        
        request_metrics = self.active_requests.pop(request_id)
        request_metrics.end_time = time.time()
        request_metrics.success = success
        request_metrics.error_message = error_message
        
        # 更新连接数
        self.device_connections[request_metrics.device_id] -= 1
        
        # 记录响应时间
        response_time = request_metrics.response_time
        self.device_response_times[request_metrics.device_id].append(response_time)
        
        # 更新统计
        if success:
            self.stats["successful_requests"] += 1
        else:
            self.stats["failed_requests"] += 1
            
            # 触发熔断器
            if self.config.enable_circuit_breaker:
                circuit_breaker = self.circuit_breakers.get(request_metrics.device_id)
                if circuit_breaker:
                    try:
                        circuit_breaker.call(lambda: None)
                    except:
                        self.stats["circuit_breaker_trips"] += 1
        
        # 保存完成的请求
        self.completed_requests.append(request_metrics)
    
    def _initialize_weights(self):
        """初始化设备权重"""
        for device_id in self.gpu_manager.gpu_configs.keys():
            self.device_weights[device_id] = 1.0
    
    def _update_weights(self):
        """更新设备权重"""
        for device_id in self.gpu_manager.gpu_configs.keys():
            metrics = self.gpu_manager.get_device_metrics(device_id)
            if metrics is None:
                self.device_weights[device_id] = 0.0
                continue
            
            # 基于性能指标计算权重
            memory_factor = 1.0 - metrics.memory_utilization
            gpu_factor = 1.0 - metrics.gpu_utilization
            
            # 响应时间因子
            response_times = self.device_response_times[device_id]
            if response_times:
                avg_response_time = sum(response_times) / len(response_times)
                response_factor = max(0.1, 1.0 / (1.0 + avg_response_time))
            else:
                response_factor = 1.0
            
            # 计算综合权重
            weight = memory_factor * gpu_factor * response_factor
            self.device_weights[device_id] = max(0.1, weight)
    
    def _health_check_loop(self):
        """健康检查循环"""
        while self.is_running:
            try:
                self._update_weights()
                time.sleep(self.config.health_check_interval)
            except Exception as e:
                logger.error(f"Health check error: {e}")
                time.sleep(5.0)
    
    def _rebalance_loop(self):
        """重平衡循环"""
        while self.is_running:
            try:
                if self._should_rebalance():
                    self._perform_rebalance()
                    self.stats["rebalance_events"] += 1
                
                time.sleep(self.config.rebalance_interval)
            except Exception as e:
                logger.error(f"Rebalance error: {e}")
                time.sleep(10.0)
    
    def _should_rebalance(self) -> bool:
        """判断是否需要重平衡"""
        # 检查负载分布
        load_distribution = self.gpu_manager.get_load_distribution()
        load_balance_score = load_distribution.get("load_balance_score", 1.0)
        
        return load_balance_score < 0.7  # 负载不均衡阈值
    
    def _perform_rebalance(self):
        """执行重平衡"""
        logger.info("Performing load rebalance")
        result = self.gpu_manager.rebalance_load()
        
        if result.get("success"):
            migrations = result.get("migrations", [])
            logger.info(f"Rebalanced {len(migrations)} experts")
        else:
            logger.warning(f"Rebalance failed: {result.get('error')}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            "active_requests": len(self.active_requests),
            "device_connections": dict(self.device_connections),
            "device_weights": dict(self.device_weights),
            "avg_response_times": {
                device_id: sum(times) / len(times) if times else 0.0
                for device_id, times in self.device_response_times.items()
            },
            "circuit_breaker_states": {
                device_id: cb.state
                for device_id, cb in self.circuit_breakers.items()
            } if self.config.enable_circuit_breaker else {},
            "current_strategy": self.config.strategy.value
        }
