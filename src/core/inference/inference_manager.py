"""
推理管理器 - 统一管理vLLM引擎
支持动态引擎选择和负载均衡
注意：DeepSpeed部分已被注释掉
"""

import torch
from typing import Dict, List, Optional, Any, Union, AsyncGenerator
from dataclasses import dataclass
from enum import Enum
import asyncio
import logging
import time

# DeepSpeed相关导入已注释掉
# from .deepspeed_engine import DeepSpeedInferenceEngine, DeepSpeedConfig
from .vllm_engine import VLLMInferenceEngine, VLLMConfig

logger = logging.getLogger(__name__)


class InferenceEngine(Enum):
    """推理引擎类型"""
    # DEEPSPEED = "deepspeed"  # DeepSpeed已注释掉
    VLLM = "vllm"
    AUTO = "auto"


@dataclass
class InferenceConfig:
    """推理配置"""
    # 引擎选择
    preferred_engine: InferenceEngine = InferenceEngine.AUTO
    enable_fallback: bool = True
    
    # 负载均衡
    enable_load_balancing: bool = True
    max_concurrent_requests: int = 100
    request_timeout: float = 30.0
    
    # 性能优化
    enable_caching: bool = True
    cache_size: int = 1000
    enable_batching: bool = True
    max_batch_size: int = 32
    batch_timeout: float = 0.1
    
    # 监控
    enable_metrics: bool = True
    metrics_interval: float = 10.0


class RequestQueue:
    """请求队列管理"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.queue = asyncio.Queue(maxsize=max_size)
        self.pending_requests = {}
        self.completed_requests = {}
    
    async def add_request(self, request_id: str, request_data: Dict[str, Any]):
        """添加请求到队列"""
        await self.queue.put((request_id, request_data))
        self.pending_requests[request_id] = {
            "data": request_data,
            "timestamp": time.time(),
            "status": "pending"
        }
    
    async def get_request(self) -> tuple:
        """获取请求"""
        return await self.queue.get()
    
    def mark_completed(self, request_id: str, result: Any):
        """标记请求完成"""
        if request_id in self.pending_requests:
            request_info = self.pending_requests.pop(request_id)
            self.completed_requests[request_id] = {
                **request_info,
                "result": result,
                "completed_at": time.time(),
                "status": "completed"
            }
    
    def get_queue_size(self) -> int:
        """获取队列大小"""
        return self.queue.qsize()
    
    def get_pending_count(self) -> int:
        """获取待处理请求数量"""
        return len(self.pending_requests)


class InferenceManager:
    """
    推理管理器
    统一管理多个推理引擎，提供负载均衡和故障转移
    """
    
    def __init__(
        self,
        config: InferenceConfig,
        # deepspeed_config: Optional[DeepSpeedConfig] = None,  # DeepSpeed已注释掉
        vllm_config: Optional[VLLMConfig] = None,
        model: Optional[torch.nn.Module] = None
    ):
        self.config = config
        self.model = model
        
        # 初始化引擎
        self.engines = {}
        self.active_engine = None
        
        # 请求队列
        self.request_queue = RequestQueue(config.max_concurrent_requests)
        
        # 性能统计
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "engine_usage": {},
            "response_times": [],
            "queue_sizes": [],
            "error_counts": {}
        }
        
        # 初始化引擎
        # self._initialize_engines(deepspeed_config, vllm_config)  # DeepSpeed已注释掉
        self._initialize_engines(None, vllm_config)
        
        # 启动后台任务
        self._background_tasks = []
        if config.enable_metrics:
            self._start_metrics_collection()
        
        logger.info("InferenceManager initialized")
    
    def _initialize_engines(
        self,
        deepspeed_config: Optional[Any],  # DeepSpeedConfig已注释掉，使用Any
        vllm_config: Optional[VLLMConfig]
    ):
        """初始化推理引擎"""
        # DeepSpeed引擎初始化已注释掉
        # if deepspeed_config and self.model:
        #     try:
        #         self.engines[InferenceEngine.DEEPSPEED] = DeepSpeedInferenceEngine(
        #             model=self.model,
        #             config=deepspeed_config
        #         )
        #         logger.info("DeepSpeed engine initialized")
        #     except Exception as e:
        #         logger.error(f"Failed to initialize DeepSpeed engine: {e}")

        # 初始化vLLM引擎
        if vllm_config:
            try:
                self.engines[InferenceEngine.VLLM] = VLLMInferenceEngine(vllm_config)
                logger.info("vLLM engine initialized")
            except Exception as e:
                logger.error(f"Failed to initialize vLLM engine: {e}")

        # 选择活跃引擎
        self._select_active_engine()
    
    def _select_active_engine(self):
        """选择活跃引擎"""
        if self.config.preferred_engine == InferenceEngine.AUTO:
            # 自动选择：现在只支持vLLM（DeepSpeed已注释掉）
            if InferenceEngine.VLLM in self.engines:
                self.active_engine = InferenceEngine.VLLM
            # elif InferenceEngine.DEEPSPEED in self.engines:  # DeepSpeed已注释掉
            #     self.active_engine = InferenceEngine.DEEPSPEED
            else:
                logger.error("No inference engines available")
                return
        else:
            # 使用指定引擎
            if self.config.preferred_engine in self.engines:
                self.active_engine = self.config.preferred_engine
            elif self.config.enable_fallback:
                # 尝试fallback
                available_engines = list(self.engines.keys())
                if available_engines:
                    self.active_engine = available_engines[0]
                    logger.warning(f"Preferred engine not available, using {self.active_engine}")
            else:
                logger.error(f"Preferred engine {self.config.preferred_engine} not available")
                return

        logger.info(f"Active inference engine: {self.active_engine}")
    
    def _get_engine(self, engine_type: Optional[InferenceEngine] = None):
        """获取指定引擎"""
        engine_type = engine_type or self.active_engine
        return self.engines.get(engine_type)
    
    def generate(
        self,
        prompts: Union[str, List[str]],
        engine_type: Optional[InferenceEngine] = None,
        sampling_params: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[str]:
        """
        同步生成
        
        Args:
            prompts: 输入提示
            engine_type: 指定引擎类型
            sampling_params: 采样参数
            
        Returns:
            List[str]: 生成的文本
        """
        start_time = time.time()
        
        try:
            engine = self._get_engine(engine_type)
            if engine is None:
                raise RuntimeError(f"Engine {engine_type or self.active_engine} not available")
            
            # 根据引擎类型调用相应方法
            if isinstance(engine, VLLMInferenceEngine):
                results = engine.generate(prompts, sampling_params, **kwargs)
            # DeepSpeed相关代码已注释掉
            # elif isinstance(engine, DeepSpeedInferenceEngine):
            #     # DeepSpeed需要特殊处理
            #     if isinstance(prompts, str):
            #         prompts = [prompts]
            #
            #     # 简化实现：使用generate方法
            #     results = []
            #     for prompt in prompts:
            #         # 这里需要tokenize prompt
            #         # 简化实现
            #         result = f"Generated by DeepSpeed: {prompt[:50]}..."
            #         results.append(result)
            else:
                raise ValueError(f"Unknown engine type: {type(engine)}")
            
            # 更新统计
            self._update_stats(engine_type or self.active_engine, start_time, True)
            
            return results
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            self._update_stats(engine_type or self.active_engine, start_time, False)
            
            # 尝试fallback
            if self.config.enable_fallback and engine_type is None:
                return self._try_fallback_generation(prompts, sampling_params, **kwargs)
            
            raise
    
    async def generate_async(
        self,
        prompts: Union[str, List[str]],
        engine_type: Optional[InferenceEngine] = None,
        sampling_params: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        异步生成
        
        Args:
            prompts: 输入提示
            engine_type: 指定引擎类型
            sampling_params: 采样参数
            request_id: 请求ID
            
        Yields:
            str: 生成的文本片段
        """
        start_time = time.time()
        
        try:
            engine = self._get_engine(engine_type)
            if engine is None:
                raise RuntimeError(f"Engine {engine_type or self.active_engine} not available")
            
            # 只有vLLM支持异步生成
            if isinstance(engine, VLLMInferenceEngine):
                async for text in engine.generate_async(
                    prompts, sampling_params, request_id, **kwargs
                ):
                    yield text
            else:
                # DeepSpeed相关代码已注释掉，现在只支持vLLM异步生成
                # results = self.generate(prompts, engine_type, sampling_params, **kwargs)
                # for result in results:
                #     yield result
                raise ValueError(f"Async generation not supported for engine type: {type(engine)}")
            
            # 更新统计
            self._update_stats(engine_type or self.active_engine, start_time, True)
            
        except Exception as e:
            logger.error(f"Async generation failed: {e}")
            self._update_stats(engine_type or self.active_engine, start_time, False)
            raise
    
    def _try_fallback_generation(
        self,
        prompts: Union[str, List[str]],
        sampling_params: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[str]:
        """尝试fallback生成"""
        for engine_type in self.engines:
            if engine_type != self.active_engine:
                try:
                    logger.info(f"Trying fallback engine: {engine_type}")
                    return self.generate(prompts, engine_type, sampling_params, **kwargs)
                except Exception as e:
                    logger.warning(f"Fallback engine {engine_type} also failed: {e}")
        
        # 所有引擎都失败
        raise RuntimeError("All inference engines failed")
    
    def _update_stats(self, engine_type: InferenceEngine, start_time: float, success: bool):
        """更新统计信息"""
        end_time = time.time()
        response_time = end_time - start_time
        
        self.stats["total_requests"] += 1
        if success:
            self.stats["successful_requests"] += 1
        else:
            self.stats["failed_requests"] += 1
            engine_name = engine_type.value if engine_type else "unknown"
            self.stats["error_counts"][engine_name] = (
                self.stats["error_counts"].get(engine_name, 0) + 1
            )
        
        self.stats["response_times"].append(response_time)
        
        # 保持最近1000次记录
        if len(self.stats["response_times"]) > 1000:
            self.stats["response_times"] = self.stats["response_times"][-1000:]
        
        # 更新引擎使用统计
        engine_name = engine_type.value if engine_type else "unknown"
        self.stats["engine_usage"][engine_name] = (
            self.stats["engine_usage"].get(engine_name, 0) + 1
        )
    
    def _start_metrics_collection(self):
        """启动指标收集"""
        async def collect_metrics():
            while True:
                try:
                    # 收集队列大小
                    queue_size = self.request_queue.get_queue_size()
                    self.stats["queue_sizes"].append(queue_size)
                    
                    # 保持最近100次记录
                    if len(self.stats["queue_sizes"]) > 100:
                        self.stats["queue_sizes"] = self.stats["queue_sizes"][-100:]
                    
                    await asyncio.sleep(self.config.metrics_interval)
                except Exception as e:
                    logger.error(f"Metrics collection error: {e}")
        
        task = asyncio.create_task(collect_metrics())
        self._background_tasks.append(task)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        if not self.stats["response_times"]:
            return {"error": "No performance data available"}
        
        import numpy as np
        
        response_times = self.stats["response_times"]
        
        stats = {
            "total_requests": self.stats["total_requests"],
            "successful_requests": self.stats["successful_requests"],
            "failed_requests": self.stats["failed_requests"],
            "success_rate": (
                self.stats["successful_requests"] / max(1, self.stats["total_requests"])
            ),
            "avg_response_time_ms": np.mean(response_times) * 1000,
            "p50_response_time_ms": np.percentile(response_times, 50) * 1000,
            "p95_response_time_ms": np.percentile(response_times, 95) * 1000,
            "p99_response_time_ms": np.percentile(response_times, 99) * 1000,
            "engine_usage": self.stats["engine_usage"],
            "error_counts": self.stats["error_counts"],
            "active_engine": self.active_engine.value if self.active_engine else None,
            "available_engines": [engine.value for engine in self.engines.keys()],
            "queue_size": self.request_queue.get_queue_size(),
            "pending_requests": self.request_queue.get_pending_count()
        }
        
        if self.stats["queue_sizes"]:
            stats["avg_queue_size"] = np.mean(self.stats["queue_sizes"])
            stats["max_queue_size"] = max(self.stats["queue_sizes"])
        
        return stats
    
    def get_engine_stats(self) -> Dict[str, Any]:
        """获取各引擎的详细统计"""
        engine_stats = {}
        
        for engine_type, engine in self.engines.items():
            try:
                if hasattr(engine, 'get_performance_stats'):
                    engine_stats[engine_type.value] = engine.get_performance_stats()
                if hasattr(engine, 'get_memory_stats'):
                    memory_stats = engine.get_memory_stats()
                    engine_stats[engine_type.value + "_memory"] = memory_stats
            except Exception as e:
                logger.error(f"Failed to get stats for {engine_type}: {e}")
                engine_stats[engine_type.value] = {"error": str(e)}
        
        return engine_stats
    
    def switch_engine(self, engine_type: InferenceEngine) -> bool:
        """切换活跃引擎"""
        if engine_type in self.engines:
            self.active_engine = engine_type
            logger.info(f"Switched to engine: {engine_type}")
            return True
        else:
            logger.error(f"Engine {engine_type} not available")
            return False
    
    def cleanup(self):
        """清理资源"""
        # 停止后台任务
        for task in self._background_tasks:
            task.cancel()
        
        # 清理引擎
        for engine in self.engines.values():
            if hasattr(engine, 'cleanup'):
                engine.cleanup()
        
        logger.info("InferenceManager cleaned up")
