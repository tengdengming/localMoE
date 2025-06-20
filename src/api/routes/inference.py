"""
推理API路由
处理推理请求和响应
"""

import asyncio
import time
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
import logging

from ..models import *
from ..dependencies import get_inference_manager, get_feature_extractor
from ...core.inference import InferenceManager
from ...core.multimodal import FeatureExtractor

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/inference", response_model=InferenceResponse)
async def inference(
    request: InferenceRequest,
    background_tasks: BackgroundTasks,
    inference_manager: InferenceManager = Depends(get_inference_manager),
    feature_extractor: FeatureExtractor = Depends(get_feature_extractor)
):
    """
    单次推理接口
    """
    start_time = time.time()
    request_id = request.request_id or str(uuid.uuid4())
    
    try:
        # 设置默认参数
        sampling_params = request.sampling_params or SamplingParams()
        model_config = request.model_config or ModelConfig()
        
        # 根据模式处理输入
        if request.mode == InferenceMode.TEXT_ONLY:
            # 纯文本推理
            result = await _process_text_inference(
                request.text, sampling_params, inference_manager
            )
        elif request.mode == InferenceMode.CODE_ONLY:
            # 纯代码推理
            result = await _process_code_inference(
                request.code, sampling_params, inference_manager
            )
        elif request.mode == InferenceMode.MULTIMODAL:
            # 多模态推理
            result = await _process_multimodal_inference(
                request.text, request.code, sampling_params, 
                inference_manager, feature_extractor, model_config
            )
        else:
            raise HTTPException(status_code=400, detail="不支持的推理模式")
        
        # 计算推理时间
        inference_time = (time.time() - start_time) * 1000
        
        # 构建响应
        inference_result = InferenceResult(
            generated_text=result["generated_text"],
            input_tokens=result.get("input_tokens", 0),
            output_tokens=result.get("output_tokens", 0),
            inference_time_ms=inference_time,
            experts_used=result.get("experts_used", []),
            model_info=result.get("model_info", {})
        )
        
        return InferenceResponse(
            success=True,
            request_id=request_id,
            result=inference_result
        )
        
    except Exception as e:
        logger.error(f"Inference failed for request {request_id}: {e}")
        return InferenceResponse(
            success=False,
            request_id=request_id,
            error=str(e)
        )


@router.post("/inference/batch", response_model=BatchInferenceResponse)
async def batch_inference(
    request: BatchInferenceRequest,
    background_tasks: BackgroundTasks,
    inference_manager: InferenceManager = Depends(get_inference_manager),
    feature_extractor: FeatureExtractor = Depends(get_feature_extractor)
):
    """
    批量推理接口
    """
    start_time = time.time()
    
    try:
        batch_size = request.batch_size or len(request.requests)
        results = []
        
        # 处理批量请求
        for i in range(0, len(request.requests), batch_size):
            batch_requests = request.requests[i:i + batch_size]
            
            # 并发处理批次
            batch_tasks = []
            for req in batch_requests:
                task = _process_single_request(
                    req, inference_manager, feature_extractor
                )
                batch_tasks.append(task)
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # 处理结果
            for result in batch_results:
                if isinstance(result, Exception):
                    results.append(InferenceResponse(
                        success=False,
                        request_id=str(uuid.uuid4()),
                        error=str(result)
                    ))
                else:
                    results.append(result)
        
        total_time = (time.time() - start_time) * 1000
        
        return BatchInferenceResponse(
            success=True,
            results=results,
            batch_info={
                "total_requests": len(request.requests),
                "batch_size": batch_size,
                "successful_requests": sum(1 for r in results if r.success),
                "failed_requests": sum(1 for r in results if not r.success)
            },
            total_time_ms=total_time
        )
        
    except Exception as e:
        logger.error(f"Batch inference failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/inference/stream")
async def stream_inference(
    request: InferenceRequest,
    inference_manager: InferenceManager = Depends(get_inference_manager),
    feature_extractor: FeatureExtractor = Depends(get_feature_extractor)
):
    """
    流式推理接口
    """
    request_id = request.request_id or str(uuid.uuid4())
    
    try:
        # 设置流式参数
        sampling_params = request.sampling_params or SamplingParams()
        sampling_params.stream = True
        
        async def generate_stream():
            try:
                # 根据模式生成流式响应
                if request.mode == InferenceMode.MULTIMODAL:
                    async for token in _stream_multimodal_inference(
                        request.text, request.code, sampling_params,
                        inference_manager, feature_extractor, request_id
                    ):
                        yield f"data: {token}\n\n"
                else:
                    # 简化的流式实现
                    prompt = request.text or request.code
                    async for token in inference_manager.generate_async(
                        prompt, sampling_params=sampling_params.dict(), 
                        request_id=request_id
                    ):
                        response = StreamResponse(
                            token=token,
                            request_id=request_id
                        )
                        yield f"data: {response.json()}\n\n"
                
                # 发送结束标记
                final_response = StreamResponse(
                    token="",
                    is_final=True,
                    request_id=request_id
                )
                yield f"data: {final_response.json()}\n\n"
                
            except Exception as e:
                error_response = ErrorResponse(
                    error=str(e),
                    error_code="STREAM_ERROR",
                    request_id=request_id
                )
                yield f"data: {error_response.json()}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Request-ID": request_id
            }
        )
        
    except Exception as e:
        logger.error(f"Stream inference failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# 辅助函数
async def _process_single_request(
    request: InferenceRequest,
    inference_manager: InferenceManager,
    feature_extractor: FeatureExtractor
) -> InferenceResponse:
    """处理单个推理请求"""
    start_time = time.time()
    request_id = request.request_id or str(uuid.uuid4())
    
    try:
        sampling_params = request.sampling_params or SamplingParams()
        model_config = request.model_config or ModelConfig()
        
        if request.mode == InferenceMode.MULTIMODAL:
            result = await _process_multimodal_inference(
                request.text, request.code, sampling_params,
                inference_manager, feature_extractor, model_config
            )
        else:
            prompt = request.text or request.code
            generated = inference_manager.generate(
                prompt, sampling_params=sampling_params.dict()
            )
            result = {
                "generated_text": generated[0] if generated else "",
                "input_tokens": len(prompt.split()) if prompt else 0,
                "output_tokens": len(generated[0].split()) if generated else 0,
                "experts_used": [],
                "model_info": {}
            }
        
        inference_time = (time.time() - start_time) * 1000
        
        return InferenceResponse(
            success=True,
            request_id=request_id,
            result=InferenceResult(
                generated_text=result["generated_text"],
                input_tokens=result.get("input_tokens", 0),
                output_tokens=result.get("output_tokens", 0),
                inference_time_ms=inference_time,
                experts_used=result.get("experts_used", []),
                model_info=result.get("model_info", {})
            )
        )
        
    except Exception as e:
        return InferenceResponse(
            success=False,
            request_id=request_id,
            error=str(e)
        )


async def _process_text_inference(
    text: str,
    sampling_params: SamplingParams,
    inference_manager: InferenceManager
) -> dict:
    """处理文本推理"""
    generated = inference_manager.generate(
        text, sampling_params=sampling_params.dict()
    )
    
    return {
        "generated_text": generated[0] if generated else "",
        "input_tokens": len(text.split()),
        "output_tokens": len(generated[0].split()) if generated else 0,
        "experts_used": [],
        "model_info": {"mode": "text_only"}
    }


async def _process_code_inference(
    code: str,
    sampling_params: SamplingParams,
    inference_manager: InferenceManager
) -> dict:
    """处理代码推理"""
    generated = inference_manager.generate(
        code, sampling_params=sampling_params.dict()
    )
    
    return {
        "generated_text": generated[0] if generated else "",
        "input_tokens": len(code.split()),
        "output_tokens": len(generated[0].split()) if generated else 0,
        "experts_used": [],
        "model_info": {"mode": "code_only"}
    }


async def _process_multimodal_inference(
    text: str,
    code: str,
    sampling_params: SamplingParams,
    inference_manager: InferenceManager,
    feature_extractor: FeatureExtractor,
    model_config: ModelConfig
) -> dict:
    """处理多模态推理"""
    try:
        # 提取多模态特征
        features = feature_extractor.extract_multimodal_features(text, code)
        
        # 构建推理输入
        combined_prompt = f"Text: {text}\nCode: {code}\nGenerate:"
        
        # 执行推理
        generated = inference_manager.generate(
            combined_prompt, sampling_params=sampling_params.dict()
        )
        
        return {
            "generated_text": generated[0] if generated else "",
            "input_tokens": len(text.split()) + len(code.split()),
            "output_tokens": len(generated[0].split()) if generated else 0,
            "experts_used": list(range(model_config.top_k_experts)),
            "model_info": {
                "mode": "multimodal",
                "fusion_strategy": "cross_attention",
                "top_k_experts": model_config.top_k_experts
            }
        }
        
    except Exception as e:
        logger.error(f"Multimodal inference failed: {e}")
        raise


async def _stream_multimodal_inference(
    text: str,
    code: str,
    sampling_params: SamplingParams,
    inference_manager: InferenceManager,
    feature_extractor: FeatureExtractor,
    request_id: str
) -> AsyncGenerator[str, None]:
    """流式多模态推理"""
    try:
        # 提取特征
        features = feature_extractor.extract_multimodal_features(text, code)
        
        # 构建提示
        combined_prompt = f"Text: {text}\nCode: {code}\nGenerate:"
        
        # 流式生成
        async for token in inference_manager.generate_async(
            combined_prompt,
            sampling_params=sampling_params.dict(),
            request_id=request_id
        ):
            response = StreamResponse(
                token=token,
                request_id=request_id
            )
            yield response.json()
            
    except Exception as e:
        logger.error(f"Stream multimodal inference failed: {e}")
        error_response = ErrorResponse(
            error=str(e),
            error_code="MULTIMODAL_STREAM_ERROR",
            request_id=request_id
        )
        yield error_response.json()
