"""
GPU管理器 - 基于DesignPlan.md的四张L40S显卡资源调度
实现GPU资源监控、分配和优化
"""

import torch
import threading
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import psutil

logger = logging.getLogger(__name__)


class GPUStatus(Enum):
    """GPU状态枚举"""
    AVAILABLE = "available"
    BUSY = "busy"
    OVERLOADED = "overloaded"
    ERROR = "error"
    MAINTENANCE = "maintenance"


@dataclass
class GPUConfig:
    """GPU配置"""
    device_id: int
    memory_limit_gb: float = 40.0  # L40S 48GB，预留8GB
    utilization_threshold: float = 0.85
    temperature_threshold: float = 85.0
    enable_monitoring: bool = True
    monitoring_interval: float = 1.0


@dataclass
class GPUMetrics:
    """GPU指标"""
    device_id: int
    memory_used_gb: float
    memory_total_gb: float
    memory_utilization: float
    gpu_utilization: float
    temperature: float
    power_usage: float
    status: GPUStatus
    last_updated: float
    active_processes: int
    allocated_experts: List[int]


class GPUMonitor:
    """GPU监控器"""
    
    def __init__(self, device_id: int, config: GPUConfig):
        self.device_id = device_id
        self.config = config
        self.metrics_history = []
        self.is_monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """开始监控"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info(f"Started monitoring GPU {self.device_id}")
    
    def stop_monitoring(self):
        """停止监控"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        
        logger.info(f"Stopped monitoring GPU {self.device_id}")
    
    def _monitor_loop(self):
        """监控循环"""
        while self.is_monitoring:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # 保持最近1000次记录
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-1000:]
                
                time.sleep(self.config.monitoring_interval)
                
            except Exception as e:
                logger.error(f"GPU {self.device_id} monitoring error: {e}")
                time.sleep(5.0)  # 错误时延长间隔
    
    def _collect_metrics(self) -> GPUMetrics:
        """收集GPU指标"""
        try:
            if not torch.cuda.is_available() or self.device_id >= torch.cuda.device_count():
                return self._create_error_metrics("CUDA not available or invalid device")
            
            torch.cuda.set_device(self.device_id)
            
            # 内存信息
            memory_used = torch.cuda.memory_allocated(self.device_id) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(self.device_id) / 1024**3
            memory_total = torch.cuda.get_device_properties(self.device_id).total_memory / 1024**3
            memory_utilization = memory_used / memory_total
            
            # GPU利用率（简化实现）
            gpu_utilization = min(memory_utilization * 1.2, 1.0)  # 估算
            
            # 温度和功耗（需要nvidia-ml-py）
            temperature, power_usage = self._get_thermal_metrics()
            
            # 活跃进程数
            active_processes = self._count_active_processes()
            
            # 确定状态
            status = self._determine_status(memory_utilization, gpu_utilization, temperature)
            
            return GPUMetrics(
                device_id=self.device_id,
                memory_used_gb=memory_used,
                memory_total_gb=memory_total,
                memory_utilization=memory_utilization,
                gpu_utilization=gpu_utilization,
                temperature=temperature,
                power_usage=power_usage,
                status=status,
                last_updated=time.time(),
                active_processes=active_processes,
                allocated_experts=[]  # 将由GPUManager填充
            )
            
        except Exception as e:
            logger.error(f"Failed to collect metrics for GPU {self.device_id}: {e}")
            return self._create_error_metrics(str(e))
    
    def _get_thermal_metrics(self) -> Tuple[float, float]:
        """获取温度和功耗指标"""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)
            
            # 温度
            temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            
            # 功耗
            power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # 转换为瓦特
            
            return float(temperature), float(power_usage)
            
        except Exception:
            # Fallback值
            return 65.0, 200.0
    
    def _count_active_processes(self) -> int:
        """统计活跃进程数"""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)
            
            processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            return len(processes)
            
        except Exception:
            return 1  # 假设至少有一个进程
    
    def _determine_status(
        self, 
        memory_util: float, 
        gpu_util: float, 
        temperature: float
    ) -> GPUStatus:
        """确定GPU状态"""
        if temperature > self.config.temperature_threshold:
            return GPUStatus.ERROR
        
        if memory_util > self.config.utilization_threshold or gpu_util > self.config.utilization_threshold:
            return GPUStatus.OVERLOADED
        
        if memory_util > 0.5 or gpu_util > 0.5:
            return GPUStatus.BUSY
        
        return GPUStatus.AVAILABLE
    
    def _create_error_metrics(self, error_msg: str) -> GPUMetrics:
        """创建错误指标"""
        return GPUMetrics(
            device_id=self.device_id,
            memory_used_gb=0.0,
            memory_total_gb=48.0,  # L40S默认显存
            memory_utilization=0.0,
            gpu_utilization=0.0,
            temperature=0.0,
            power_usage=0.0,
            status=GPUStatus.ERROR,
            last_updated=time.time(),
            active_processes=0,
            allocated_experts=[]
        )
    
    def get_latest_metrics(self) -> Optional[GPUMetrics]:
        """获取最新指标"""
        if self.metrics_history:
            return self.metrics_history[-1]
        return None
    
    def get_average_metrics(self, window_size: int = 60) -> Optional[GPUMetrics]:
        """获取平均指标"""
        if not self.metrics_history:
            return None
        
        recent_metrics = self.metrics_history[-window_size:]
        if not recent_metrics:
            return None
        
        # 计算平均值
        avg_memory_util = sum(m.memory_utilization for m in recent_metrics) / len(recent_metrics)
        avg_gpu_util = sum(m.gpu_utilization for m in recent_metrics) / len(recent_metrics)
        avg_temperature = sum(m.temperature for m in recent_metrics) / len(recent_metrics)
        avg_power = sum(m.power_usage for m in recent_metrics) / len(recent_metrics)
        
        latest = recent_metrics[-1]
        
        return GPUMetrics(
            device_id=self.device_id,
            memory_used_gb=latest.memory_used_gb,
            memory_total_gb=latest.memory_total_gb,
            memory_utilization=avg_memory_util,
            gpu_utilization=avg_gpu_util,
            temperature=avg_temperature,
            power_usage=avg_power,
            status=latest.status,
            last_updated=latest.last_updated,
            active_processes=latest.active_processes,
            allocated_experts=latest.allocated_experts
        )


class GPUManager:
    """
    GPU管理器
    管理四张L40S显卡的资源分配和监控
    """
    
    def __init__(self, gpu_configs: Optional[List[GPUConfig]] = None):
        self.device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        
        # 默认配置四张L40S
        if gpu_configs is None:
            gpu_configs = [
                GPUConfig(device_id=i, memory_limit_gb=40.0)
                for i in range(min(4, self.device_count))
            ]
        
        self.gpu_configs = {config.device_id: config for config in gpu_configs}
        self.monitors = {}
        self.expert_allocations = {}  # expert_id -> device_id
        self.device_experts = {i: set() for i in range(self.device_count)}
        
        # 初始化监控器
        for config in gpu_configs:
            if config.enable_monitoring:
                monitor = GPUMonitor(config.device_id, config)
                self.monitors[config.device_id] = monitor
                monitor.start_monitoring()
        
        # 统计信息
        self.stats = {
            "allocations": 0,
            "deallocations": 0,
            "allocation_failures": 0,
            "load_balancing_events": 0
        }
        
        logger.info(f"GPUManager initialized with {len(gpu_configs)} GPUs")
    
    def allocate_expert(self, expert_id: int, memory_requirement: float = 4.0) -> Optional[int]:
        """
        分配专家到GPU
        
        Args:
            expert_id: 专家ID
            memory_requirement: 内存需求(GB)
            
        Returns:
            Optional[int]: 分配的设备ID，失败返回None
        """
        try:
            # 检查是否已分配
            if expert_id in self.expert_allocations:
                device_id = self.expert_allocations[expert_id]
                logger.debug(f"Expert {expert_id} already allocated to GPU {device_id}")
                return device_id
            
            # 选择最佳GPU
            best_device = self._select_best_device(memory_requirement)
            if best_device is None:
                self.stats["allocation_failures"] += 1
                logger.warning(f"Failed to allocate expert {expert_id}: no suitable GPU")
                return None
            
            # 执行分配
            self.expert_allocations[expert_id] = best_device
            self.device_experts[best_device].add(expert_id)
            self.stats["allocations"] += 1
            
            logger.info(f"Allocated expert {expert_id} to GPU {best_device}")
            return best_device
            
        except Exception as e:
            logger.error(f"Failed to allocate expert {expert_id}: {e}")
            self.stats["allocation_failures"] += 1
            return None
    
    def deallocate_expert(self, expert_id: int) -> bool:
        """
        释放专家分配
        
        Args:
            expert_id: 专家ID
            
        Returns:
            bool: 是否成功释放
        """
        try:
            if expert_id not in self.expert_allocations:
                logger.warning(f"Expert {expert_id} not allocated")
                return False
            
            device_id = self.expert_allocations[expert_id]
            
            # 移除分配
            del self.expert_allocations[expert_id]
            self.device_experts[device_id].discard(expert_id)
            self.stats["deallocations"] += 1
            
            logger.info(f"Deallocated expert {expert_id} from GPU {device_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to deallocate expert {expert_id}: {e}")
            return False
    
    def _select_best_device(self, memory_requirement: float) -> Optional[int]:
        """选择最佳设备"""
        best_device = None
        best_score = float('-inf')
        
        for device_id in self.gpu_configs.keys():
            score = self._calculate_device_score(device_id, memory_requirement)
            if score > best_score:
                best_score = score
                best_device = device_id
        
        return best_device if best_score > 0 else None
    
    def _calculate_device_score(self, device_id: int, memory_requirement: float) -> float:
        """计算设备分数"""
        try:
            # 获取设备指标
            monitor = self.monitors.get(device_id)
            if monitor is None:
                return 0.0
            
            metrics = monitor.get_latest_metrics()
            if metrics is None or metrics.status == GPUStatus.ERROR:
                return 0.0
            
            # 检查内存是否足够
            available_memory = metrics.memory_total_gb - metrics.memory_used_gb
            if available_memory < memory_requirement:
                return 0.0
            
            # 检查状态
            if metrics.status == GPUStatus.OVERLOADED:
                return 0.0
            
            # 计算分数（越低越好）
            memory_score = (1.0 - metrics.memory_utilization) * 100
            gpu_score = (1.0 - metrics.gpu_utilization) * 100
            temperature_score = max(0, (85 - metrics.temperature) / 85) * 50
            load_score = max(0, (10 - len(self.device_experts[device_id])) / 10) * 50
            
            total_score = memory_score + gpu_score + temperature_score + load_score
            
            return total_score
            
        except Exception as e:
            logger.error(f"Failed to calculate score for device {device_id}: {e}")
            return 0.0
    
    def get_device_metrics(self, device_id: int) -> Optional[GPUMetrics]:
        """获取设备指标"""
        monitor = self.monitors.get(device_id)
        if monitor:
            metrics = monitor.get_latest_metrics()
            if metrics:
                # 添加专家分配信息
                metrics.allocated_experts = list(self.device_experts[device_id])
            return metrics
        return None
    
    def get_all_metrics(self) -> Dict[int, GPUMetrics]:
        """获取所有设备指标"""
        all_metrics = {}
        for device_id in self.gpu_configs.keys():
            metrics = self.get_device_metrics(device_id)
            if metrics:
                all_metrics[device_id] = metrics
        return all_metrics
    
    def get_load_distribution(self) -> Dict[str, Any]:
        """获取负载分布"""
        distribution = {}
        total_experts = len(self.expert_allocations)
        
        for device_id in self.gpu_configs.keys():
            expert_count = len(self.device_experts[device_id])
            distribution[f"gpu_{device_id}"] = {
                "expert_count": expert_count,
                "load_percentage": (expert_count / max(1, total_experts)) * 100,
                "experts": list(self.device_experts[device_id])
            }
        
        return {
            "distribution": distribution,
            "total_experts": total_experts,
            "load_balance_score": self._calculate_load_balance_score()
        }
    
    def _calculate_load_balance_score(self) -> float:
        """计算负载均衡分数"""
        if not self.expert_allocations:
            return 1.0
        
        expert_counts = [len(experts) for experts in self.device_experts.values()]
        if not expert_counts:
            return 1.0
        
        min_count = min(expert_counts)
        max_count = max(expert_counts)
        
        if max_count == 0:
            return 1.0
        
        return min_count / max_count
    
    def rebalance_load(self) -> Dict[str, Any]:
        """重新平衡负载"""
        try:
            self.stats["load_balancing_events"] += 1
            
            # 获取当前分布
            current_distribution = self.get_load_distribution()
            
            # 简化的重平衡策略：将过载设备的专家迁移到空闲设备
            migrations = []
            
            # 找到过载和空闲的设备
            overloaded_devices = []
            underloaded_devices = []
            
            for device_id, info in current_distribution["distribution"].items():
                device_num = int(device_id.split('_')[1])
                expert_count = info["expert_count"]
                
                if expert_count > 2:  # 假设每个GPU最多2个专家
                    overloaded_devices.append((device_num, expert_count, info["experts"]))
                elif expert_count < 2:
                    underloaded_devices.append((device_num, expert_count))
            
            # 执行迁移
            for overloaded_device, count, experts in overloaded_devices:
                if not underloaded_devices:
                    break
                
                # 迁移多余的专家
                excess_experts = experts[2:]  # 保留前2个
                
                for expert_id in excess_experts:
                    if underloaded_devices:
                        target_device, target_count = underloaded_devices.pop(0)
                        
                        # 执行迁移
                        self.deallocate_expert(expert_id)
                        new_device = self.allocate_expert(expert_id)
                        
                        if new_device == target_device:
                            migrations.append({
                                "expert_id": expert_id,
                                "from_device": overloaded_device,
                                "to_device": target_device
                            })
                        
                        # 更新目标设备计数
                        if target_count + 1 < 2:
                            underloaded_devices.append((target_device, target_count + 1))
            
            return {
                "success": True,
                "migrations": migrations,
                "new_distribution": self.get_load_distribution()
            }
            
        except Exception as e:
            logger.error(f"Load rebalancing failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            "device_count": len(self.gpu_configs),
            "total_experts": len(self.expert_allocations),
            "load_distribution": self.get_load_distribution(),
            "device_status": {
                device_id: metrics.status.value if metrics else "unknown"
                for device_id, metrics in self.get_all_metrics().items()
            }
        }
    
    def cleanup(self):
        """清理资源"""
        # 停止所有监控器
        for monitor in self.monitors.values():
            monitor.stop_monitoring()
        
        # 清理分配
        self.expert_allocations.clear()
        for experts in self.device_experts.values():
            experts.clear()
        
        logger.info("GPUManager cleaned up")
