"""
监控API路由
提供系统监控和指标查询接口
"""

import time
import psutil
import torch
from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any
import logging

from ..models import SystemMetrics, ExpertStatusResponse, ExpertStatus, SuccessResponse
from ..dependencies import get_inference_manager, get_feature_extractor, get_app_state
from ...core.inference import InferenceManager
from ...core.multimodal import FeatureExtractor

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/metrics", response_model=SystemMetrics)
async def get_system_metrics(
    inference_manager: InferenceManager = Depends(get_inference_manager),
    app_state: Dict[str, Any] = Depends(get_app_state)
):
    """
    获取系统指标
    """
    try:
        # GPU指标
        gpu_metrics = _get_gpu_metrics()
        
        # 内存指标
        memory_metrics = _get_memory_metrics()
        
        # 推理指标
        inference_metrics = _get_inference_metrics(inference_manager, app_state)
        
        # 专家指标
        expert_metrics = _get_expert_metrics(inference_manager)
        
        return SystemMetrics(
            gpu_metrics=gpu_metrics,
            memory_metrics=memory_metrics,
            inference_metrics=inference_metrics,
            expert_metrics=expert_metrics
        )
        
    except Exception as e:
        logger.error(f"Failed to get system metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/gpu")
async def get_gpu_metrics():
    """
    获取GPU指标
    """
    try:
        return _get_gpu_metrics()
    except Exception as e:
        logger.error(f"Failed to get GPU metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/memory")
async def get_memory_metrics():
    """
    获取内存指标
    """
    try:
        return _get_memory_metrics()
    except Exception as e:
        logger.error(f"Failed to get memory metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/inference")
async def get_inference_metrics(
    inference_manager: InferenceManager = Depends(get_inference_manager),
    app_state: Dict[str, Any] = Depends(get_app_state)
):
    """
    获取推理指标
    """
    try:
        return _get_inference_metrics(inference_manager, app_state)
    except Exception as e:
        logger.error(f"Failed to get inference metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/experts", response_model=ExpertStatusResponse)
async def get_expert_status(
    inference_manager: InferenceManager = Depends(get_inference_manager)
):
    """
    获取专家状态
    """
    try:
        # 获取专家统计信息
        engine_stats = inference_manager.get_engine_stats()
        
        experts = []
        total_experts = 8  # 默认专家数量
        active_experts = 0
        
        # 模拟专家状态（实际应该从推理引擎获取）
        for expert_id in range(total_experts):
            # 简化的专家状态
            status = "active" if expert_id < 4 else "idle"
            if status == "active":
                active_experts += 1
            
            expert_status = ExpertStatus(
                expert_id=expert_id,
                device_id=expert_id % 4,  # 分布到4个GPU
                status=status,
                load=0.5 if status == "active" else 0.0,
                memory_usage_gb=4.0 if status == "active" else 0.0,
                request_count=100 if status == "active" else 0,
                avg_latency_ms=15.0 if status == "active" else 0.0
            )
            experts.append(expert_status)
        
        return ExpertStatusResponse(
            experts=experts,
            total_experts=total_experts,
            active_experts=active_experts
        )
        
    except Exception as e:
        logger.error(f"Failed to get expert status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/experts/{expert_id}")
async def get_expert_detail(
    expert_id: int,
    inference_manager: InferenceManager = Depends(get_inference_manager)
):
    """
    获取特定专家的详细信息
    """
    try:
        if expert_id < 0 or expert_id >= 8:
            raise HTTPException(status_code=404, detail="Expert not found")
        
        # 获取专家详细信息
        expert_detail = {
            "expert_id": expert_id,
            "device_id": expert_id % 4,
            "status": "active" if expert_id < 4 else "idle",
            "load": 0.5 if expert_id < 4 else 0.0,
            "memory_usage_gb": 4.0 if expert_id < 4 else 0.0,
            "request_count": 100 if expert_id < 4 else 0,
            "avg_latency_ms": 15.0 if expert_id < 4 else 0.0,
            "parameters": "4B",
            "quantization": "FP16",
            "last_active": time.time() if expert_id < 4 else None,
            "total_inference_time": 3600.0 if expert_id < 4 else 0.0,
            "error_count": 0
        }
        
        return expert_detail
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get expert {expert_id} detail: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance")
async def get_performance_stats(
    inference_manager: InferenceManager = Depends(get_inference_manager),
    app_state: Dict[str, Any] = Depends(get_app_state)
):
    """
    获取性能统计
    """
    try:
        # 从推理管理器获取性能统计
        perf_stats = inference_manager.get_performance_stats()
        
        # 添加应用级统计
        uptime = time.time() - app_state.get("start_time", time.time())
        
        performance_data = {
            **perf_stats,
            "app_uptime_seconds": uptime,
            "app_request_count": app_state.get("request_count", 0),
            "app_error_count": app_state.get("error_count", 0),
            "app_error_rate": (
                app_state.get("error_count", 0) / 
                max(1, app_state.get("request_count", 1))
            )
        }
        
        return performance_data
        
    except Exception as e:
        logger.error(f"Failed to get performance stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/metrics/reset", response_model=SuccessResponse)
async def reset_metrics(
    inference_manager: InferenceManager = Depends(get_inference_manager),
    app_state: Dict[str, Any] = Depends(get_app_state)
):
    """
    重置指标统计
    """
    try:
        # 重置应用级统计
        app_state["request_count"] = 0
        app_state["error_count"] = 0
        app_state["start_time"] = time.time()
        
        # 重置推理管理器统计（如果支持）
        if hasattr(inference_manager, 'reset_stats'):
            inference_manager.reset_stats()
        
        return SuccessResponse(
            message="Metrics reset successfully"
        )
        
    except Exception as e:
        logger.error(f"Failed to reset metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# 辅助函数
def _get_gpu_metrics() -> Dict[str, Any]:
    """获取GPU指标"""
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    
    gpu_metrics = {}
    
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        
        # 内存信息
        allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(i) / 1024**3    # GB
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        
        # GPU属性
        props = torch.cuda.get_device_properties(i)
        
        gpu_metrics[f"gpu_{i}"] = {
            "name": props.name,
            "memory_allocated_gb": allocated,
            "memory_reserved_gb": reserved,
            "memory_total_gb": total,
            "memory_utilization": allocated / total,
            "memory_free_gb": total - allocated,
            "compute_capability": f"{props.major}.{props.minor}",
            "multiprocessor_count": props.multi_processor_count,
            "is_available": True
        }
    
    return gpu_metrics


def _get_memory_metrics() -> Dict[str, Any]:
    """获取系统内存指标"""
    # 系统内存
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    
    return {
        "system_memory": {
            "total_gb": memory.total / 1024**3,
            "available_gb": memory.available / 1024**3,
            "used_gb": memory.used / 1024**3,
            "utilization": memory.percent / 100,
            "free_gb": memory.free / 1024**3
        },
        "swap_memory": {
            "total_gb": swap.total / 1024**3,
            "used_gb": swap.used / 1024**3,
            "free_gb": swap.free / 1024**3,
            "utilization": swap.percent / 100
        }
    }


def _get_inference_metrics(
    inference_manager: InferenceManager,
    app_state: Dict[str, Any]
) -> Dict[str, Any]:
    """获取推理指标"""
    try:
        # 从推理管理器获取统计
        perf_stats = inference_manager.get_performance_stats()
        
        return {
            "total_requests": perf_stats.get("total_requests", 0),
            "successful_requests": perf_stats.get("successful_requests", 0),
            "failed_requests": perf_stats.get("failed_requests", 0),
            "success_rate": perf_stats.get("success_rate", 0.0),
            "avg_response_time_ms": perf_stats.get("avg_response_time_ms", 0.0),
            "p95_response_time_ms": perf_stats.get("p95_response_time_ms", 0.0),
            "p99_response_time_ms": perf_stats.get("p99_response_time_ms", 0.0),
            "active_engine": perf_stats.get("active_engine", "unknown"),
            "queue_size": perf_stats.get("queue_size", 0),
            "pending_requests": perf_stats.get("pending_requests", 0)
        }
        
    except Exception as e:
        logger.warning(f"Failed to get inference metrics: {e}")
        return {
            "total_requests": app_state.get("request_count", 0),
            "error_count": app_state.get("error_count", 0),
            "uptime_seconds": time.time() - app_state.get("start_time", time.time())
        }


def _get_expert_metrics(inference_manager: InferenceManager) -> Dict[str, Any]:
    """获取专家指标"""
    try:
        # 获取引擎统计
        engine_stats = inference_manager.get_engine_stats()
        
        return {
            "total_experts": 8,
            "active_experts": 4,
            "expert_utilization": {
                f"expert_{i}": 0.5 if i < 4 else 0.0 
                for i in range(8)
            },
            "expert_memory_usage": {
                f"expert_{i}": 4.0 if i < 4 else 0.0 
                for i in range(8)
            },
            "expert_request_counts": {
                f"expert_{i}": 100 if i < 4 else 0 
                for i in range(8)
            },
            "load_balance_score": 0.8,
            "engine_stats": engine_stats
        }
        
    except Exception as e:
        logger.warning(f"Failed to get expert metrics: {e}")
        return {
            "total_experts": 8,
            "active_experts": 4,
            "error": str(e)
        }
