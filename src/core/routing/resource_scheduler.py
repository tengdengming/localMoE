"""
资源调度器 - 统一的GPU资源调度和管理
整合GPU管理器和负载均衡器，提供高级调度功能
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor

from .gpu_manager import GPUManager, GPUConfig
from .load_balancer import LoadBalancer, LoadBalancerConfig, LoadBalancingStrategy

logger = logging.getLogger(__name__)


class SchedulingPolicy(Enum):
    """调度策略"""
    FIFO = "fifo"  # 先进先出
    PRIORITY = "priority"  # 优先级调度
    FAIR_SHARE = "fair_share"  # 公平共享
    DEADLINE = "deadline"  # 截止时间调度
    ADAPTIVE = "adaptive"  # 自适应调度


@dataclass
class SchedulingRequest:
    """调度请求"""
    request_id: str
    expert_ids: List[int]
    priority: int = 5  # 1-10，数字越小优先级越高
    deadline: Optional[float] = None
    memory_requirement: float = 4.0
    estimated_duration: float = 1.0
    user_id: Optional[str] = None
    submitted_time: float = None
    
    def __post_init__(self):
        if self.submitted_time is None:
            self.submitted_time = time.time()


@dataclass
class SchedulingResult:
    """调度结果"""
    request_id: str
    allocated_devices: Dict[int, int]  # expert_id -> device_id
    estimated_start_time: float
    estimated_completion_time: float
    queue_position: Optional[int] = None
    success: bool = True
    error_message: Optional[str] = None


class ResourceScheduler:
    """
    资源调度器
    提供高级的GPU资源调度和管理功能
    """
    
    def __init__(
        self,
        gpu_configs: Optional[List[GPUConfig]] = None,
        load_balancer_config: Optional[LoadBalancerConfig] = None,
        scheduling_policy: SchedulingPolicy = SchedulingPolicy.ADAPTIVE,
        max_queue_size: int = 1000,
        enable_preemption: bool = False
    ):
        # 初始化GPU管理器
        self.gpu_manager = GPUManager(gpu_configs)
        
        # 初始化负载均衡器
        if load_balancer_config is None:
            load_balancer_config = LoadBalancerConfig()
        self.load_balancer = LoadBalancer(self.gpu_manager, load_balancer_config)
        
        # 调度配置
        self.scheduling_policy = scheduling_policy
        self.max_queue_size = max_queue_size
        self.enable_preemption = enable_preemption
        
        # 调度队列
        self.pending_queue = []  # 待调度请求
        self.running_requests = {}  # 正在运行的请求
        self.completed_requests = []  # 已完成的请求
        
        # 用户配额管理
        self.user_quotas = {}  # user_id -> quota_info
        self.user_usage = {}   # user_id -> current_usage
        
        # 线程池用于异步调度
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # 调度统计
        self.stats = {
            "total_requests": 0,
            "scheduled_requests": 0,
            "failed_requests": 0,
            "preempted_requests": 0,
            "avg_queue_time": 0.0,
            "avg_execution_time": 0.0
        }
        
        # 启动组件
        self.load_balancer.start()
        
        logger.info(f"ResourceScheduler initialized with policy: {scheduling_policy.value}")
    
    async def submit_request(self, request: SchedulingRequest) -> SchedulingResult:
        """
        提交调度请求
        
        Args:
            request: 调度请求
            
        Returns:
            SchedulingResult: 调度结果
        """
        try:
            self.stats["total_requests"] += 1
            
            # 验证请求
            if not self._validate_request(request):
                return SchedulingResult(
                    request_id=request.request_id,
                    allocated_devices={},
                    estimated_start_time=0,
                    estimated_completion_time=0,
                    success=False,
                    error_message="Invalid request"
                )
            
            # 检查用户配额
            if not self._check_user_quota(request):
                return SchedulingResult(
                    request_id=request.request_id,
                    allocated_devices={},
                    estimated_start_time=0,
                    estimated_completion_time=0,
                    success=False,
                    error_message="User quota exceeded"
                )
            
            # 尝试立即调度
            immediate_result = await self._try_immediate_scheduling(request)
            if immediate_result.success:
                return immediate_result
            
            # 加入队列
            if len(self.pending_queue) >= self.max_queue_size:
                return SchedulingResult(
                    request_id=request.request_id,
                    allocated_devices={},
                    estimated_start_time=0,
                    estimated_completion_time=0,
                    success=False,
                    error_message="Queue is full"
                )
            
            # 插入队列（根据调度策略排序）
            self._insert_into_queue(request)
            
            # 估算等待时间
            queue_position = self._get_queue_position(request.request_id)
            estimated_start_time = self._estimate_start_time(request, queue_position)
            estimated_completion_time = estimated_start_time + request.estimated_duration
            
            return SchedulingResult(
                request_id=request.request_id,
                allocated_devices={},
                estimated_start_time=estimated_start_time,
                estimated_completion_time=estimated_completion_time,
                queue_position=queue_position,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Failed to submit request {request.request_id}: {e}")
            self.stats["failed_requests"] += 1
            return SchedulingResult(
                request_id=request.request_id,
                allocated_devices={},
                estimated_start_time=0,
                estimated_completion_time=0,
                success=False,
                error_message=str(e)
            )
    
    async def _try_immediate_scheduling(self, request: SchedulingRequest) -> SchedulingResult:
        """尝试立即调度"""
        allocated_devices = {}
        
        # 为每个专家分配设备
        for expert_id in request.expert_ids:
            device_id = self.load_balancer.select_device(request.request_id, expert_id)
            if device_id is None:
                # 回滚已分配的设备
                for allocated_expert, allocated_device in allocated_devices.items():
                    self.gpu_manager.deallocate_expert(allocated_expert)
                
                return SchedulingResult(
                    request_id=request.request_id,
                    allocated_devices={},
                    estimated_start_time=0,
                    estimated_completion_time=0,
                    success=False,
                    error_message="No available devices"
                )
            
            # 分配专家到设备
            success = self.gpu_manager.allocate_expert(expert_id, request.memory_requirement)
            if not success:
                # 回滚
                for allocated_expert, allocated_device in allocated_devices.items():
                    self.gpu_manager.deallocate_expert(allocated_expert)
                
                return SchedulingResult(
                    request_id=request.request_id,
                    allocated_devices={},
                    estimated_start_time=0,
                    estimated_completion_time=0,
                    success=False,
                    error_message=f"Failed to allocate expert {expert_id}"
                )
            
            allocated_devices[expert_id] = device_id
        
        # 记录运行中的请求
        self.running_requests[request.request_id] = {
            "request": request,
            "allocated_devices": allocated_devices,
            "start_time": time.time()
        }
        
        # 更新用户使用量
        self._update_user_usage(request, allocated_devices, increment=True)
        
        self.stats["scheduled_requests"] += 1
        
        return SchedulingResult(
            request_id=request.request_id,
            allocated_devices=allocated_devices,
            estimated_start_time=time.time(),
            estimated_completion_time=time.time() + request.estimated_duration,
            success=True
        )
    
    def _validate_request(self, request: SchedulingRequest) -> bool:
        """验证请求"""
        if not request.expert_ids:
            return False
        
        if request.memory_requirement <= 0:
            return False
        
        if request.estimated_duration <= 0:
            return False
        
        # 检查专家ID是否有效
        for expert_id in request.expert_ids:
            if expert_id < 0 or expert_id >= 8:  # 假设最多8个专家
                return False
        
        return True
    
    def _check_user_quota(self, request: SchedulingRequest) -> bool:
        """检查用户配额"""
        if request.user_id is None:
            return True  # 匿名用户不限制
        
        user_quota = self.user_quotas.get(request.user_id, {
            "max_concurrent_requests": 10,
            "max_gpu_hours_per_day": 24.0,
            "max_memory_gb": 32.0
        })
        
        current_usage = self.user_usage.get(request.user_id, {
            "concurrent_requests": 0,
            "gpu_hours_today": 0.0,
            "memory_usage_gb": 0.0
        })
        
        # 检查并发请求数
        if current_usage["concurrent_requests"] >= user_quota["max_concurrent_requests"]:
            return False
        
        # 检查内存使用
        required_memory = len(request.expert_ids) * request.memory_requirement
        if current_usage["memory_usage_gb"] + required_memory > user_quota["max_memory_gb"]:
            return False
        
        return True
    
    def _insert_into_queue(self, request: SchedulingRequest):
        """插入队列"""
        if self.scheduling_policy == SchedulingPolicy.FIFO:
            self.pending_queue.append(request)
        elif self.scheduling_policy == SchedulingPolicy.PRIORITY:
            # 按优先级插入
            inserted = False
            for i, existing_request in enumerate(self.pending_queue):
                if request.priority < existing_request.priority:
                    self.pending_queue.insert(i, request)
                    inserted = True
                    break
            if not inserted:
                self.pending_queue.append(request)
        elif self.scheduling_policy == SchedulingPolicy.DEADLINE:
            # 按截止时间排序
            if request.deadline is not None:
                inserted = False
                for i, existing_request in enumerate(self.pending_queue):
                    if (existing_request.deadline is None or 
                        request.deadline < existing_request.deadline):
                        self.pending_queue.insert(i, request)
                        inserted = True
                        break
                if not inserted:
                    self.pending_queue.append(request)
            else:
                self.pending_queue.append(request)
        else:
            # 默认FIFO
            self.pending_queue.append(request)
    
    def _get_queue_position(self, request_id: str) -> Optional[int]:
        """获取队列位置"""
        for i, request in enumerate(self.pending_queue):
            if request.request_id == request_id:
                return i + 1
        return None
    
    def _estimate_start_time(self, request: SchedulingRequest, queue_position: int) -> float:
        """估算开始时间"""
        current_time = time.time()
        
        if queue_position <= 1:
            return current_time
        
        # 简化估算：假设前面的请求平均执行时间
        avg_execution_time = self.stats.get("avg_execution_time", 1.0)
        estimated_wait_time = (queue_position - 1) * avg_execution_time
        
        return current_time + estimated_wait_time
    
    def _update_user_usage(
        self, 
        request: SchedulingRequest, 
        allocated_devices: Dict[int, int], 
        increment: bool = True
    ):
        """更新用户使用量"""
        if request.user_id is None:
            return
        
        if request.user_id not in self.user_usage:
            self.user_usage[request.user_id] = {
                "concurrent_requests": 0,
                "gpu_hours_today": 0.0,
                "memory_usage_gb": 0.0
            }
        
        usage = self.user_usage[request.user_id]
        memory_delta = len(allocated_devices) * request.memory_requirement
        
        if increment:
            usage["concurrent_requests"] += 1
            usage["memory_usage_gb"] += memory_delta
        else:
            usage["concurrent_requests"] = max(0, usage["concurrent_requests"] - 1)
            usage["memory_usage_gb"] = max(0, usage["memory_usage_gb"] - memory_delta)
    
    async def complete_request(self, request_id: str, success: bool = True) -> bool:
        """
        完成请求
        
        Args:
            request_id: 请求ID
            success: 是否成功完成
            
        Returns:
            bool: 是否成功处理
        """
        try:
            if request_id not in self.running_requests:
                logger.warning(f"Request {request_id} not found in running requests")
                return False
            
            request_info = self.running_requests.pop(request_id)
            request = request_info["request"]
            allocated_devices = request_info["allocated_devices"]
            start_time = request_info["start_time"]
            
            # 释放资源
            for expert_id in allocated_devices.keys():
                self.gpu_manager.deallocate_expert(expert_id)
            
            # 完成负载均衡器中的请求
            self.load_balancer.complete_request(request_id, success)
            
            # 更新用户使用量
            self._update_user_usage(request, allocated_devices, increment=False)
            
            # 记录完成的请求
            execution_time = time.time() - start_time
            self.completed_requests.append({
                "request": request,
                "execution_time": execution_time,
                "success": success,
                "completion_time": time.time()
            })
            
            # 更新统计
            self._update_execution_stats(execution_time)
            
            # 尝试调度队列中的下一个请求
            await self._schedule_next_request()
            
            logger.info(f"Completed request {request_id} (success: {success})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to complete request {request_id}: {e}")
            return False
    
    async def _schedule_next_request(self):
        """调度队列中的下一个请求"""
        if not self.pending_queue:
            return
        
        # 尝试调度队列中的请求
        scheduled_requests = []
        
        for request in self.pending_queue[:]:
            result = await self._try_immediate_scheduling(request)
            if result.success:
                self.pending_queue.remove(request)
                scheduled_requests.append(request.request_id)
        
        if scheduled_requests:
            logger.info(f"Scheduled {len(scheduled_requests)} requests from queue")
    
    def _update_execution_stats(self, execution_time: float):
        """更新执行统计"""
        # 更新平均执行时间
        if self.stats["avg_execution_time"] == 0:
            self.stats["avg_execution_time"] = execution_time
        else:
            # 指数移动平均
            alpha = 0.1
            self.stats["avg_execution_time"] = (
                alpha * execution_time + 
                (1 - alpha) * self.stats["avg_execution_time"]
            )
    
    def get_queue_status(self) -> Dict[str, Any]:
        """获取队列状态"""
        return {
            "pending_requests": len(self.pending_queue),
            "running_requests": len(self.running_requests),
            "completed_requests": len(self.completed_requests),
            "queue_details": [
                {
                    "request_id": req.request_id,
                    "priority": req.priority,
                    "expert_count": len(req.expert_ids),
                    "wait_time": time.time() - req.submitted_time,
                    "estimated_start_time": self._estimate_start_time(req, i + 1)
                }
                for i, req in enumerate(self.pending_queue[:10])  # 只显示前10个
            ]
        }
    
    def get_resource_utilization(self) -> Dict[str, Any]:
        """获取资源利用率"""
        gpu_metrics = self.gpu_manager.get_all_metrics()
        load_stats = self.load_balancer.get_statistics()
        
        return {
            "gpu_utilization": {
                f"gpu_{device_id}": {
                    "memory_utilization": metrics.memory_utilization,
                    "gpu_utilization": metrics.gpu_utilization,
                    "allocated_experts": len(metrics.allocated_experts),
                    "status": metrics.status.value
                }
                for device_id, metrics in gpu_metrics.items()
            },
            "load_balancer_stats": load_stats,
            "scheduler_stats": self.stats
        }
    
    def cleanup(self):
        """清理资源"""
        self.load_balancer.stop()
        self.gpu_manager.cleanup()
        self.executor.shutdown(wait=True)
        
        logger.info("ResourceScheduler cleaned up")
