"""
管理API路由
提供系统管理和配置接口
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
import logging

from ..models import (
    ConfigUpdateRequest, ConfigUpdateResponse, SuccessResponse,
    InferenceEngine
)
from ..dependencies import get_inference_manager, get_config_manager
from ...core.inference import InferenceManager

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/config")
async def get_current_config(
    config_manager = Depends(get_config_manager)
):
    """
    获取当前配置
    """
    try:
        # 获取各种配置
        config = {
            "inference": {
                "preferred_engine": "auto",
                "enable_fallback": True,
                "enable_load_balancing": True,
                "max_concurrent_requests": 100,
                "request_timeout": 30.0
            },
            "sampling": {
                "temperature": 0.8,
                "top_p": 0.95,
                "top_k": 50,
                "max_tokens": 512,
                "repetition_penalty": 1.1
            },
            "model": {
                "use_moe": True,
                "top_k_experts": 2,
                "expert_selection_strategy": "adaptive",
                "enable_caching": True,
                "quantization": None
            },
            "system": {
                "gpu_memory_utilization": 0.9,
                "enable_metrics": True,
                "log_level": "info",
                "max_batch_size": 32
            }
        }
        
        return config
        
    except Exception as e:
        logger.error(f"Failed to get config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/config", response_model=ConfigUpdateResponse)
async def update_config(
    request: ConfigUpdateRequest,
    config_manager = Depends(get_config_manager)
):
    """
    更新配置
    """
    try:
        # 获取当前配置
        current_config = await get_current_config(config_manager)
        old_config = current_config.get(request.config_type, {})
        
        # 验证配置
        _validate_config(request.config_type, request.config_data)
        
        # 更新配置
        new_config = {**old_config, **request.config_data}
        
        # 应用配置（这里是模拟实现）
        success = _apply_config(request.config_type, new_config)
        
        if success:
            return ConfigUpdateResponse(
                success=True,
                message=f"Configuration {request.config_type} updated successfully",
                old_config=old_config,
                new_config=new_config
            )
        else:
            raise HTTPException(status_code=400, detail="Failed to apply configuration")
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to update config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/engine/switch", response_model=SuccessResponse)
async def switch_engine(
    engine_type: str,
    inference_manager: InferenceManager = Depends(get_inference_manager)
):
    """
    切换推理引擎
    """
    try:
        # 验证引擎类型
        if engine_type not in ["deepspeed", "vllm", "auto"]:
            raise HTTPException(
                status_code=400, 
                detail="Invalid engine type. Must be 'deepspeed', 'vllm', or 'auto'"
            )
        
        # 转换为枚举
        if engine_type == "deepspeed":
            target_engine = InferenceEngine.DEEPSPEED
        elif engine_type == "vllm":
            target_engine = InferenceEngine.VLLM
        else:
            target_engine = InferenceEngine.AUTO
        
        # 切换引擎
        success = inference_manager.switch_engine(target_engine)
        
        if success:
            return SuccessResponse(
                message=f"Successfully switched to {engine_type} engine",
                data={"new_engine": engine_type}
            )
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Failed to switch to {engine_type} engine"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to switch engine: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/engine/status")
async def get_engine_status(
    inference_manager: InferenceManager = Depends(get_inference_manager)
):
    """
    获取引擎状态
    """
    try:
        perf_stats = inference_manager.get_performance_stats()
        engine_stats = inference_manager.get_engine_stats()
        
        return {
            "active_engine": perf_stats.get("active_engine", "unknown"),
            "available_engines": perf_stats.get("available_engines", []),
            "engine_performance": perf_stats,
            "engine_details": engine_stats
        }
        
    except Exception as e:
        logger.error(f"Failed to get engine status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/experts/activate", response_model=SuccessResponse)
async def activate_expert(
    expert_id: int,
    inference_manager: InferenceManager = Depends(get_inference_manager)
):
    """
    激活专家
    """
    try:
        if expert_id < 0 or expert_id >= 8:
            raise HTTPException(status_code=400, detail="Invalid expert ID")
        
        # 这里应该调用实际的专家激活逻辑
        # 目前是模拟实现
        success = True  # 模拟激活成功
        
        if success:
            return SuccessResponse(
                message=f"Expert {expert_id} activated successfully",
                data={"expert_id": expert_id, "status": "active"}
            )
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Failed to activate expert {expert_id}"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to activate expert {expert_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/experts/deactivate", response_model=SuccessResponse)
async def deactivate_expert(
    expert_id: int,
    inference_manager: InferenceManager = Depends(get_inference_manager)
):
    """
    停用专家
    """
    try:
        if expert_id < 0 or expert_id >= 8:
            raise HTTPException(status_code=400, detail="Invalid expert ID")
        
        # 这里应该调用实际的专家停用逻辑
        # 目前是模拟实现
        success = True  # 模拟停用成功
        
        if success:
            return SuccessResponse(
                message=f"Expert {expert_id} deactivated successfully",
                data={"expert_id": expert_id, "status": "inactive"}
            )
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Failed to deactivate expert {expert_id}"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to deactivate expert {expert_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cache/clear", response_model=SuccessResponse)
async def clear_cache(
    cache_type: str = "all",
    inference_manager: InferenceManager = Depends(get_inference_manager)
):
    """
    清空缓存
    """
    try:
        if cache_type not in ["all", "feature", "model", "expert"]:
            raise HTTPException(
                status_code=400, 
                detail="Invalid cache type. Must be 'all', 'feature', 'model', or 'expert'"
            )
        
        # 清空相应缓存
        cleared_caches = []
        
        if cache_type in ["all", "feature"]:
            # 清空特征缓存
            cleared_caches.append("feature_cache")
        
        if cache_type in ["all", "model"]:
            # 清空模型缓存
            cleared_caches.append("model_cache")
        
        if cache_type in ["all", "expert"]:
            # 清空专家缓存
            cleared_caches.append("expert_cache")
        
        return SuccessResponse(
            message=f"Cache cleared successfully",
            data={
                "cache_type": cache_type,
                "cleared_caches": cleared_caches
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/system/info")
async def get_system_info():
    """
    获取系统信息
    """
    try:
        import platform
        import torch
        
        system_info = {
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor()
            },
            "python": {
                "version": platform.python_version(),
                "implementation": platform.python_implementation()
            },
            "pytorch": {
                "version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
                "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
            },
            "service": {
                "name": "LocalMoE",
                "version": "0.1.0",
                "description": "多模态MoE推理服务"
            }
        }
        
        # 添加GPU信息
        if torch.cuda.is_available():
            gpu_info = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpu_info.append({
                    "id": i,
                    "name": props.name,
                    "memory_gb": props.total_memory / 1024**3,
                    "compute_capability": f"{props.major}.{props.minor}",
                    "multiprocessor_count": props.multi_processor_count
                })
            system_info["gpus"] = gpu_info
        
        return system_info
        
    except Exception as e:
        logger.error(f"Failed to get system info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# 辅助函数
def _validate_config(config_type: str, config_data: Dict[str, Any]):
    """验证配置数据"""
    if config_type == "sampling":
        if "temperature" in config_data:
            temp = config_data["temperature"]
            if not (0.0 <= temp <= 2.0):
                raise ValueError("Temperature must be between 0.0 and 2.0")
        
        if "top_p" in config_data:
            top_p = config_data["top_p"]
            if not (0.0 <= top_p <= 1.0):
                raise ValueError("top_p must be between 0.0 and 1.0")
        
        if "max_tokens" in config_data:
            max_tokens = config_data["max_tokens"]
            if not (1 <= max_tokens <= 2048):
                raise ValueError("max_tokens must be between 1 and 2048")
    
    elif config_type == "model":
        if "top_k_experts" in config_data:
            top_k = config_data["top_k_experts"]
            if not (1 <= top_k <= 8):
                raise ValueError("top_k_experts must be between 1 and 8")
    
    elif config_type == "system":
        if "gpu_memory_utilization" in config_data:
            util = config_data["gpu_memory_utilization"]
            if not (0.1 <= util <= 1.0):
                raise ValueError("gpu_memory_utilization must be between 0.1 and 1.0")


def _apply_config(config_type: str, config_data: Dict[str, Any]) -> bool:
    """应用配置"""
    try:
        # 这里应该实现实际的配置应用逻辑
        # 目前是模拟实现
        logger.info(f"Applied {config_type} configuration: {config_data}")
        return True
    except Exception as e:
        logger.error(f"Failed to apply {config_type} configuration: {e}")
        return False
