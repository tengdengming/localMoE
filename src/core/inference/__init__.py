"""
推理引擎模块
集成vLLM，支持分布式推理和模型并行
注意：DeepSpeed部分已被注释掉
"""

# DeepSpeed相关导入已注释掉
# from .deepspeed_engine import DeepSpeedInferenceEngine, DeepSpeedConfig
from .vllm_engine import VLLMInferenceEngine, VLLMConfig
from .inference_manager import InferenceManager, InferenceConfig
from .model_parallel import ModelParallelManager

__all__ = [
    # DeepSpeed相关已注释掉
    # "DeepSpeedInferenceEngine",
    # "DeepSpeedConfig",
    "VLLMInferenceEngine",
    "VLLMConfig",
    "InferenceManager",
    "InferenceConfig",
    "ModelParallelManager"
]
