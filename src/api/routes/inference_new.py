"""
推理API路由 - 简化版本，使用模拟引擎
"""

import time
import logging
from typing import Dict, List, Any
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse

from ..models import (
    InferenceRequest, InferenceResponse, BatchInferenceRequest, 
    BatchInferenceResponse, ExpertStatusResponse
)
from ...core.inference.manager import inference_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["inference"])


async def get_inference_manager():
    """获取推理管理器依赖"""
    if not inference_manager.is_available:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Inference manager not available",
                "error_code": "HTTP_503",
                "details": None,
                "request_id": None,
                "timestamp": time.time()
            }
        )
    return inference_manager


@router.post("/inference", response_model=InferenceResponse)
async def inference(
    request: InferenceRequest,
    manager = Depends(get_inference_manager)
):
    """执行推理"""
    try:
        logger.info(f"收到推理请求: {request.request_id}")
        response = await manager.inference(request)
        logger.info(f"推理完成: {request.request_id}")
        return response
    except Exception as e:
        logger.error(f"推理失败: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Inference failed",
                "error_code": "HTTP_500",
                "details": {"exception": str(e)},
                "request_id": request.request_id,
                "timestamp": time.time()
            }
        )


@router.post("/batch_inference", response_model=BatchInferenceResponse)
async def batch_inference(
    request: BatchInferenceRequest,
    manager = Depends(get_inference_manager)
):
    """批量推理"""
    try:
        logger.info(f"收到批量推理请求，数量: {len(request.requests)}")
        response = await manager.batch_inference(request)
        logger.info(f"批量推理完成")
        return response
    except Exception as e:
        logger.error(f"批量推理失败: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Batch inference failed",
                "error_code": "HTTP_500",
                "details": {"exception": str(e)},
                "request_id": None,
                "timestamp": time.time()
            }
        )


@router.get("/stats")
async def get_stats(manager = Depends(get_inference_manager)):
    """获取系统统计信息"""
    try:
        stats = manager.get_stats()
        return JSONResponse(content=stats)
    except Exception as e:
        logger.error(f"获取统计信息失败: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to get stats",
                "error_code": "HTTP_500",
                "details": {"exception": str(e)},
                "request_id": None,
                "timestamp": time.time()
            }
        )


@router.get("/models")
async def get_models(manager = Depends(get_inference_manager)):
    """获取模型列表"""
    try:
        models = manager.get_models()
        return JSONResponse(content={"models": models})
    except Exception as e:
        logger.error(f"获取模型列表失败: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to get models",
                "error_code": "HTTP_500",
                "details": {"exception": str(e)},
                "request_id": None,
                "timestamp": time.time()
            }
        )


@router.get("/experts", response_model=ExpertStatusResponse)
async def get_experts(manager = Depends(get_inference_manager)):
    """获取专家状态"""
    try:
        experts = manager.get_expert_status()
        return ExpertStatusResponse(
            experts=experts,
            total_experts=len(experts),
            active_experts=sum(1 for e in experts if e.status == "active")
        )
    except Exception as e:
        logger.error(f"获取专家状态失败: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to get experts",
                "error_code": "HTTP_500",
                "details": {"exception": str(e)},
                "request_id": None,
                "timestamp": time.time()
            }
        )
