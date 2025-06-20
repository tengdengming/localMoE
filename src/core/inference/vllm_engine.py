"""
vLLM推理引擎 - 高性能推理优化
支持PagedAttention、连续批处理和动态批处理
"""

import torch
from typing import Dict, List, Optional, Any, Union, AsyncGenerator
from dataclasses import dataclass
import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

try:
    from vllm import LLM, SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    logger.warning("vLLM not available, using fallback implementation")


@dataclass
class VLLMConfig:
    """vLLM配置"""
    # 模型配置
    model_name: str = "microsoft/DialoGPT-medium"
    tensor_parallel_size: int = 4
    pipeline_parallel_size: int = 1
    
    # 内存配置
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 2048
    block_size: int = 16
    swap_space: int = 4  # GB
    
    # 批处理配置
    max_num_batched_tokens: int = 8192
    max_num_seqs: int = 256
    max_paddings: int = 256
    
    # 性能优化
    enable_prefix_caching: bool = True
    use_v2_block_manager: bool = True
    enable_chunked_prefill: bool = True
    max_chunked_prefill_tokens: int = 512
    
    # 量化配置 - 针对L40S优化
    quantization: Optional[str] = None  # "awq", "gptq", "fp8", "squeezellm"
    kv_cache_dtype: Optional[str] = None  # "fp8_e5m2" for L40S FP8 support
    quantization_param_path: Optional[str] = None  # AWQ/GPTQ参数路径
    load_format: str = "auto"
    
    # 推理配置
    seed: int = 42
    trust_remote_code: bool = False
    revision: Optional[str] = None
    
    # 调度配置
    scheduler_delay_factor: float = 0.0
    enable_lora: bool = False
    max_lora_rank: int = 16


class VLLMInferenceEngine:
    """
    vLLM推理引擎
    提供高性能的批处理推理能力
    """
    
    def __init__(self, config: VLLMConfig):
        self.config = config
        self.engine = None
        self.async_engine = None
        self.is_initialized = False
        
        # 性能统计
        self.stats = {
            "requests_processed": 0,
            "total_tokens_generated": 0,
            "total_input_tokens": 0,
            "batch_sizes": [],
            "latencies": [],
            "throughput_history": []
        }
        
        # 线程池用于异步处理
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # 初始化引擎
        self._initialize_engine()
        
        logger.info("VLLMInferenceEngine initialized")
    
    def _initialize_engine(self):
        """初始化vLLM引擎"""
        if not VLLM_AVAILABLE:
            logger.error("vLLM is not available")
            return
        
        try:
            # 构建引擎参数 - 针对L40S优化
            engine_kwargs = {
                "model": self.config.model_name,
                "tensor_parallel_size": self.config.tensor_parallel_size,
                "pipeline_parallel_size": self.config.pipeline_parallel_size,
                "gpu_memory_utilization": self.config.gpu_memory_utilization,
                "max_model_len": self.config.max_model_len,
                "block_size": self.config.block_size,
                "swap_space": self.config.swap_space,
                "max_num_batched_tokens": self.config.max_num_batched_tokens,
                "max_num_seqs": self.config.max_num_seqs,
                "max_paddings": self.config.max_paddings,
                "enable_prefix_caching": self.config.enable_prefix_caching,
                "use_v2_block_manager": self.config.use_v2_block_manager,
                "enable_chunked_prefill": self.config.enable_chunked_prefill,
                "max_chunked_prefill_tokens": self.config.max_chunked_prefill_tokens,
                "load_format": self.config.load_format,
                "seed": self.config.seed,
                "trust_remote_code": self.config.trust_remote_code,
                "revision": self.config.revision
            }

            # 添加量化配置
            if self.config.quantization:
                engine_kwargs["quantization"] = self.config.quantization

                # L40S FP8支持
                if self.config.quantization == "fp8" and self.config.kv_cache_dtype:
                    engine_kwargs["kv_cache_dtype"] = self.config.kv_cache_dtype

                # AWQ/GPTQ参数路径
                if self.config.quantization_param_path:
                    engine_kwargs["quantization_param_path"] = self.config.quantization_param_path

            # 同步引擎
            self.engine = LLM(**engine_kwargs)
            
            # 异步引擎配置
            engine_args = AsyncEngineArgs(
                model=self.config.model_name,
                tensor_parallel_size=self.config.tensor_parallel_size,
                pipeline_parallel_size=self.config.pipeline_parallel_size,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                max_model_len=self.config.max_model_len,
                block_size=self.config.block_size,
                swap_space=self.config.swap_space,
                max_num_batched_tokens=self.config.max_num_batched_tokens,
                max_num_seqs=self.config.max_num_seqs,
                scheduler_delay_factor=self.config.scheduler_delay_factor,
                enable_lora=self.config.enable_lora,
                max_lora_rank=self.config.max_lora_rank
            )
            
            # 异步引擎
            self.async_engine = AsyncLLMEngine.from_engine_args(engine_args)
            
            self.is_initialized = True
            logger.info("vLLM engines initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize vLLM engine: {e}")
            # 使用fallback实现
            self._initialize_fallback_engine()
    
    def _initialize_fallback_engine(self):
        """初始化fallback引擎"""
        logger.info("Using fallback inference engine")
        self.engine = None
        self.async_engine = None
        self.is_initialized = True
    
    def generate(
        self,
        prompts: Union[str, List[str]],
        sampling_params: Optional[Dict[str, Any]] = None,
        use_tqdm: bool = False
    ) -> List[str]:
        """
        同步生成
        
        Args:
            prompts: 输入提示
            sampling_params: 采样参数
            use_tqdm: 是否显示进度条
            
        Returns:
            List[str]: 生成的文本
        """
        if not self.is_initialized:
            raise RuntimeError("Engine not initialized")
        
        if isinstance(prompts, str):
            prompts = [prompts]
        
        start_time = time.time()
        
        # 默认采样参数
        default_params = {
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 50,
            "max_tokens": 512,
            "repetition_penalty": 1.1
        }
        
        if sampling_params:
            default_params.update(sampling_params)
        
        try:
            if self.engine is not None:
                # 使用vLLM引擎
                vllm_sampling_params = SamplingParams(**default_params)
                outputs = self.engine.generate(
                    prompts, 
                    vllm_sampling_params,
                    use_tqdm=use_tqdm
                )
                
                # 提取生成的文本
                generated_texts = [output.outputs[0].text for output in outputs]
            else:
                # Fallback实现
                generated_texts = self._fallback_generate(prompts, default_params)
            
            # 更新统计
            end_time = time.time()
            latency = end_time - start_time
            
            self.stats["requests_processed"] += len(prompts)
            self.stats["batch_sizes"].append(len(prompts))
            self.stats["latencies"].append(latency)
            
            # 计算token数量（简化估算）
            input_tokens = sum(len(prompt.split()) for prompt in prompts)
            output_tokens = sum(len(text.split()) for text in generated_texts)
            
            self.stats["total_input_tokens"] += input_tokens
            self.stats["total_tokens_generated"] += output_tokens
            
            # 计算吞吐量
            throughput = output_tokens / latency if latency > 0 else 0
            self.stats["throughput_history"].append(throughput)
            
            return generated_texts
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    async def generate_async(
        self,
        prompts: Union[str, List[str]],
        sampling_params: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        异步生成
        
        Args:
            prompts: 输入提示
            sampling_params: 采样参数
            request_id: 请求ID
            
        Yields:
            str: 生成的文本片段
        """
        if not self.is_initialized:
            raise RuntimeError("Engine not initialized")
        
        if isinstance(prompts, str):
            prompts = [prompts]
        
        # 默认采样参数
        default_params = {
            "temperature": 0.8,
            "top_p": 0.95,
            "max_tokens": 512,
            "stream": True
        }
        
        if sampling_params:
            default_params.update(sampling_params)
        
        try:
            if self.async_engine is not None:
                # 使用vLLM异步引擎
                for prompt in prompts:
                    vllm_sampling_params = SamplingParams(**default_params)
                    
                    async for request_output in self.async_engine.generate(
                        prompt, 
                        vllm_sampling_params, 
                        request_id or f"req_{time.time()}"
                    ):
                        if request_output.outputs:
                            yield request_output.outputs[0].text
            else:
                # Fallback异步实现
                async for text in self._fallback_generate_async(prompts, default_params):
                    yield text
                    
        except Exception as e:
            logger.error(f"Async generation failed: {e}")
            raise
    
    def _fallback_generate(
        self, 
        prompts: List[str], 
        params: Dict[str, Any]
    ) -> List[str]:
        """Fallback生成实现"""
        # 简单的fallback实现
        generated_texts = []
        for prompt in prompts:
            # 模拟生成过程
            generated_text = f"Generated response for: {prompt[:50]}..."
            generated_texts.append(generated_text)
        
        return generated_texts
    
    async def _fallback_generate_async(
        self, 
        prompts: List[str], 
        params: Dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        """Fallback异步生成实现"""
        for prompt in prompts:
            # 模拟流式生成
            words = f"Generated response for: {prompt[:50]}...".split()
            for word in words:
                yield word + " "
                await asyncio.sleep(0.01)  # 模拟生成延迟
    
    def batch_generate(
        self,
        prompts: List[str],
        batch_size: int = 32,
        sampling_params: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        批量生成
        
        Args:
            prompts: 输入提示列表
            batch_size: 批次大小
            sampling_params: 采样参数
            
        Returns:
            List[str]: 生成的文本列表
        """
        all_results = []
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_results = self.generate(
                batch_prompts, 
                sampling_params=sampling_params
            )
            all_results.extend(batch_results)
        
        return all_results
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        if self.engine is not None:
            try:
                # 从vLLM引擎获取信息
                model_config = self.engine.llm_engine.model_config
                return {
                    "model_name": self.config.model_name,
                    "max_model_len": model_config.max_model_len,
                    "vocab_size": model_config.get_vocab_size(),
                    "hidden_size": model_config.get_hidden_size(),
                    "num_layers": model_config.get_num_layers(),
                    "num_attention_heads": model_config.get_num_attention_heads(),
                    "tensor_parallel_size": self.config.tensor_parallel_size,
                    "pipeline_parallel_size": self.config.pipeline_parallel_size
                }
            except Exception as e:
                logger.warning(f"Failed to get model info: {e}")
        
        return {
            "model_name": self.config.model_name,
            "tensor_parallel_size": self.config.tensor_parallel_size,
            "pipeline_parallel_size": self.config.pipeline_parallel_size,
            "max_model_len": self.config.max_model_len
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        if not self.stats["latencies"]:
            return {"error": "No performance data available"}
        
        import numpy as np
        
        latencies = self.stats["latencies"]
        throughputs = self.stats["throughput_history"]
        batch_sizes = self.stats["batch_sizes"]
        
        return {
            "requests_processed": self.stats["requests_processed"],
            "total_input_tokens": self.stats["total_input_tokens"],
            "total_tokens_generated": self.stats["total_tokens_generated"],
            "avg_latency_ms": np.mean(latencies) * 1000,
            "p50_latency_ms": np.percentile(latencies, 50) * 1000,
            "p95_latency_ms": np.percentile(latencies, 95) * 1000,
            "p99_latency_ms": np.percentile(latencies, 99) * 1000,
            "avg_throughput_tokens_per_sec": np.mean(throughputs) if throughputs else 0,
            "avg_batch_size": np.mean(batch_sizes) if batch_sizes else 0,
            "max_batch_size": max(batch_sizes) if batch_sizes else 0,
            "total_throughput": (
                self.stats["total_tokens_generated"] / sum(latencies) 
                if latencies and sum(latencies) > 0 else 0
            )
        }
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """获取内存统计"""
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
        
        stats = {}
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            
            stats[f"gpu_{i}"] = {
                "allocated_gb": allocated,
                "reserved_gb": reserved,
                "total_gb": total,
                "utilization": allocated / total,
                "free_gb": total - allocated
            }
        
        return stats
    
    def cleanup(self):
        """清理资源"""
        if self.executor:
            self.executor.shutdown(wait=True)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("vLLM engine cleaned up")
