"""
指标收集器 - 收集系统性能和业务指标
支持多种指标类型和自定义指标
"""

import time
import threading
import logging
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import psutil
import torch

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """指标类型"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class Metric:
    """指标定义"""
    name: str
    metric_type: MetricType
    description: str
    labels: Dict[str, str] = field(default_factory=dict)
    value: Union[int, float] = 0
    timestamp: float = field(default_factory=time.time)
    
    def update(self, value: Union[int, float], labels: Optional[Dict[str, str]] = None):
        """更新指标值"""
        if self.metric_type == MetricType.COUNTER:
            self.value += value
        else:
            self.value = value
        
        if labels:
            self.labels.update(labels)
        
        self.timestamp = time.time()


@dataclass
class HistogramBucket:
    """直方图桶"""
    upper_bound: float
    count: int = 0


class MetricsCollector:
    """
    指标收集器
    收集和管理各种系统和业务指标
    """
    
    def __init__(self, collection_interval: float = 10.0):
        self.collection_interval = collection_interval
        self.metrics: Dict[str, Metric] = {}
        self.histograms: Dict[str, List[HistogramBucket]] = {}
        self.time_series: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # 收集线程
        self.collection_thread = None
        self.is_collecting = False
        
        # 自定义收集器
        self.custom_collectors: List[Callable[[], Dict[str, Any]]] = []
        
        # 锁
        self.metrics_lock = threading.RLock()
        
        # 初始化基础指标
        self._initialize_base_metrics()
        
        logger.info("MetricsCollector initialized")
    
    def _initialize_base_metrics(self):
        """初始化基础指标"""
        base_metrics = [
            # 系统指标
            Metric("system_cpu_usage", MetricType.GAUGE, "System CPU usage percentage"),
            Metric("system_memory_usage", MetricType.GAUGE, "System memory usage percentage"),
            Metric("system_disk_usage", MetricType.GAUGE, "System disk usage percentage"),
            
            # GPU指标
            Metric("gpu_memory_usage", MetricType.GAUGE, "GPU memory usage", {"device": ""}),
            Metric("gpu_utilization", MetricType.GAUGE, "GPU utilization", {"device": ""}),
            Metric("gpu_temperature", MetricType.GAUGE, "GPU temperature", {"device": ""}),
            
            # 推理指标
            Metric("inference_requests_total", MetricType.COUNTER, "Total inference requests"),
            Metric("inference_requests_success", MetricType.COUNTER, "Successful inference requests"),
            Metric("inference_requests_failed", MetricType.COUNTER, "Failed inference requests"),
            Metric("inference_latency_seconds", MetricType.HISTOGRAM, "Inference latency in seconds"),
            Metric("inference_queue_size", MetricType.GAUGE, "Current inference queue size"),
            
            # 专家指标
            Metric("expert_activations_total", MetricType.COUNTER, "Total expert activations", {"expert_id": ""}),
            Metric("expert_memory_usage", MetricType.GAUGE, "Expert memory usage", {"expert_id": ""}),
            Metric("expert_load_balance_score", MetricType.GAUGE, "Expert load balance score"),
            
            # API指标
            Metric("api_requests_total", MetricType.COUNTER, "Total API requests", {"method": "", "endpoint": ""}),
            Metric("api_request_duration_seconds", MetricType.HISTOGRAM, "API request duration"),
            Metric("api_active_connections", MetricType.GAUGE, "Active API connections"),
            
            # 错误指标
            Metric("errors_total", MetricType.COUNTER, "Total errors", {"type": "", "component": ""}),
        ]
        
        with self.metrics_lock:
            for metric in base_metrics:
                self.metrics[metric.name] = metric
        
        # 初始化直方图桶
        self._initialize_histograms()
    
    def _initialize_histograms(self):
        """初始化直方图桶"""
        histogram_configs = {
            "inference_latency_seconds": [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            "api_request_duration_seconds": [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
        }
        
        with self.metrics_lock:
            for metric_name, bounds in histogram_configs.items():
                buckets = [HistogramBucket(bound) for bound in bounds]
                buckets.append(HistogramBucket(float('inf')))  # +Inf bucket
                self.histograms[metric_name] = buckets
    
    def start_collection(self):
        """开始指标收集"""
        if self.is_collecting:
            return
        
        self.is_collecting = True
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        
        logger.info("Started metrics collection")
    
    def stop_collection(self):
        """停止指标收集"""
        self.is_collecting = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5.0)
        
        logger.info("Stopped metrics collection")
    
    def _collection_loop(self):
        """指标收集循环"""
        while self.is_collecting:
            try:
                self._collect_system_metrics()
                self._collect_gpu_metrics()
                self._collect_custom_metrics()
                
                time.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                time.sleep(5.0)
    
    def _collect_system_metrics(self):
        """收集系统指标"""
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=None)
            self.update_metric("system_cpu_usage", cpu_percent)
            
            # 内存使用率
            memory = psutil.virtual_memory()
            self.update_metric("system_memory_usage", memory.percent)
            
            # 磁盘使用率
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.update_metric("system_disk_usage", disk_percent)
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
    
    def _collect_gpu_metrics(self):
        """收集GPU指标"""
        try:
            if not torch.cuda.is_available():
                return
            
            for i in range(torch.cuda.device_count()):
                torch.cuda.set_device(i)
                
                # 内存使用率
                allocated = torch.cuda.memory_allocated(i)
                total = torch.cuda.get_device_properties(i).total_memory
                memory_percent = (allocated / total) * 100
                
                self.update_metric(
                    "gpu_memory_usage", 
                    memory_percent, 
                    {"device": str(i)}
                )
                
                # GPU利用率（简化估算）
                gpu_util = min(memory_percent * 1.2, 100.0)
                self.update_metric(
                    "gpu_utilization", 
                    gpu_util, 
                    {"device": str(i)}
                )
                
                # 温度（需要nvidia-ml-py）
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    
                    self.update_metric(
                        "gpu_temperature", 
                        temp, 
                        {"device": str(i)}
                    )
                except:
                    # Fallback温度
                    self.update_metric(
                        "gpu_temperature", 
                        65.0, 
                        {"device": str(i)}
                    )
                    
        except Exception as e:
            logger.error(f"Failed to collect GPU metrics: {e}")
    
    def _collect_custom_metrics(self):
        """收集自定义指标"""
        for collector in self.custom_collectors:
            try:
                custom_metrics = collector()
                for name, value in custom_metrics.items():
                    if isinstance(value, dict):
                        # 带标签的指标
                        metric_value = value.get("value", 0)
                        labels = value.get("labels", {})
                        self.update_metric(name, metric_value, labels)
                    else:
                        # 简单指标
                        self.update_metric(name, value)
                        
            except Exception as e:
                logger.error(f"Custom metrics collection error: {e}")
    
    def register_metric(self, metric: Metric):
        """注册新指标"""
        with self.metrics_lock:
            self.metrics[metric.name] = metric
        
        logger.debug(f"Registered metric: {metric.name}")
    
    def update_metric(
        self, 
        name: str, 
        value: Union[int, float], 
        labels: Optional[Dict[str, str]] = None
    ):
        """更新指标值"""
        with self.metrics_lock:
            if name in self.metrics:
                self.metrics[name].update(value, labels)
                
                # 记录时间序列
                self.time_series[name].append({
                    "value": value,
                    "timestamp": time.time(),
                    "labels": labels or {}
                })
            else:
                logger.warning(f"Unknown metric: {name}")
    
    def increment_counter(
        self, 
        name: str, 
        value: Union[int, float] = 1, 
        labels: Optional[Dict[str, str]] = None
    ):
        """增加计数器"""
        with self.metrics_lock:
            if name in self.metrics and self.metrics[name].metric_type == MetricType.COUNTER:
                self.metrics[name].update(value, labels)
            else:
                logger.warning(f"Counter metric not found or wrong type: {name}")
    
    def observe_histogram(
        self, 
        name: str, 
        value: float, 
        labels: Optional[Dict[str, str]] = None
    ):
        """观察直方图值"""
        with self.metrics_lock:
            if name in self.histograms:
                buckets = self.histograms[name]
                for bucket in buckets:
                    if value <= bucket.upper_bound:
                        bucket.count += 1
                
                # 更新指标
                if name in self.metrics:
                    self.metrics[name].update(value, labels)
            else:
                logger.warning(f"Histogram metric not found: {name}")
    
    def get_metric(self, name: str) -> Optional[Metric]:
        """获取指标"""
        with self.metrics_lock:
            return self.metrics.get(name)
    
    def get_all_metrics(self) -> Dict[str, Metric]:
        """获取所有指标"""
        with self.metrics_lock:
            return self.metrics.copy()
    
    def get_time_series(self, name: str, limit: int = 100) -> List[Dict[str, Any]]:
        """获取时间序列数据"""
        with self.metrics_lock:
            if name in self.time_series:
                series = list(self.time_series[name])
                return series[-limit:] if limit > 0 else series
            return []
    
    def get_histogram_buckets(self, name: str) -> Optional[List[HistogramBucket]]:
        """获取直方图桶"""
        with self.metrics_lock:
            return self.histograms.get(name)
    
    def add_custom_collector(self, collector: Callable[[], Dict[str, Any]]):
        """添加自定义收集器"""
        self.custom_collectors.append(collector)
        logger.info("Added custom metrics collector")
    
    def remove_custom_collector(self, collector: Callable[[], Dict[str, Any]]):
        """移除自定义收集器"""
        if collector in self.custom_collectors:
            self.custom_collectors.remove(collector)
            logger.info("Removed custom metrics collector")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """获取指标摘要"""
        with self.metrics_lock:
            summary = {
                "total_metrics": len(self.metrics),
                "metric_types": {},
                "last_collection": time.time(),
                "collection_interval": self.collection_interval,
                "is_collecting": self.is_collecting
            }
            
            # 按类型统计
            for metric in self.metrics.values():
                metric_type = metric.metric_type.value
                summary["metric_types"][metric_type] = summary["metric_types"].get(metric_type, 0) + 1
            
            return summary
    
    def export_metrics(self, format: str = "prometheus") -> str:
        """导出指标"""
        if format == "prometheus":
            return self._export_prometheus_format()
        elif format == "json":
            return self._export_json_format()
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_prometheus_format(self) -> str:
        """导出Prometheus格式"""
        lines = []
        
        with self.metrics_lock:
            for metric in self.metrics.values():
                # 指标帮助信息
                lines.append(f"# HELP {metric.name} {metric.description}")
                lines.append(f"# TYPE {metric.name} {metric.metric_type.value}")
                
                # 指标值
                labels_str = ""
                if metric.labels:
                    label_pairs = [f'{k}="{v}"' for k, v in metric.labels.items()]
                    labels_str = "{" + ",".join(label_pairs) + "}"
                
                lines.append(f"{metric.name}{labels_str} {metric.value} {int(metric.timestamp * 1000)}")
                
                # 直方图桶
                if metric.name in self.histograms:
                    buckets = self.histograms[metric.name]
                    cumulative_count = 0
                    
                    for bucket in buckets:
                        cumulative_count += bucket.count
                        bound_str = "+Inf" if bucket.upper_bound == float('inf') else str(bucket.upper_bound)
                        lines.append(f"{metric.name}_bucket{{le=\"{bound_str}\"}} {cumulative_count}")
                    
                    lines.append(f"{metric.name}_count {cumulative_count}")
                    lines.append(f"{metric.name}_sum {metric.value * cumulative_count}")
        
        return "\n".join(lines)
    
    def _export_json_format(self) -> str:
        """导出JSON格式"""
        import json
        
        with self.metrics_lock:
            data = {
                "timestamp": time.time(),
                "metrics": {}
            }
            
            for name, metric in self.metrics.items():
                data["metrics"][name] = {
                    "type": metric.metric_type.value,
                    "description": metric.description,
                    "value": metric.value,
                    "labels": metric.labels,
                    "timestamp": metric.timestamp
                }
                
                # 添加直方图数据
                if name in self.histograms:
                    buckets = self.histograms[name]
                    data["metrics"][name]["buckets"] = [
                        {
                            "upper_bound": bucket.upper_bound,
                            "count": bucket.count
                        }
                        for bucket in buckets
                    ]
            
            return json.dumps(data, indent=2)
    
    def reset_metrics(self):
        """重置所有指标"""
        with self.metrics_lock:
            for metric in self.metrics.values():
                if metric.metric_type == MetricType.COUNTER:
                    metric.value = 0
                metric.timestamp = time.time()
            
            # 重置直方图
            for buckets in self.histograms.values():
                for bucket in buckets:
                    bucket.count = 0
            
            # 清空时间序列
            self.time_series.clear()
        
        logger.info("All metrics reset")
    
    def cleanup(self):
        """清理资源"""
        self.stop_collection()
        self.custom_collectors.clear()
        
        with self.metrics_lock:
            self.metrics.clear()
            self.histograms.clear()
            self.time_series.clear()
        
        logger.info("MetricsCollector cleaned up")
