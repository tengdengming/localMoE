"""
推理管理器 - 统一管理不同的推理引擎
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from enum import Enum

from ...api.models import (
    InferenceRequest, InferenceResponse, BatchInferenceRequest, 
    BatchInferenceResponse, SystemMetrics, ExpertStatus
)
from .mock_engine import MockInferenceEngine

logger = logging.getLogger(__name__)


class EngineType(str, Enum):
    """引擎类型"""
    MOCK = "mock"
    VLLM = "vllm"
    DEEPSPEED = "deepspeed"


class InferenceManager:
    """推理管理器"""
    
    def __init__(self, engine_type: EngineType = EngineType.MOCK):
        self.engine_type = engine_type
        self.engine = None
        self.is_initialized = False
        self.is_available = False
        
    async def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """初始化推理管理器"""
        try:
            logger.info(f"初始化推理管理器，引擎类型: {self.engine_type}")
            
            if self.engine_type == EngineType.MOCK:
                self.engine = MockInferenceEngine()
                success = await self.engine.initialize()
            elif self.engine_type == EngineType.VLLM:
                # TODO: 实现vLLM引擎初始化
                logger.warning("vLLM引擎暂未实现，使用模拟引擎")
                self.engine = MockInferenceEngine()
                success = await self.engine.initialize()
            elif self.engine_type == EngineType.DEEPSPEED:
                # TODO: 实现DeepSpeed引擎初始化
                logger.warning("DeepSpeed引擎暂未实现，使用模拟引擎")
                self.engine = MockInferenceEngine()
                success = await self.engine.initialize()
            else:
                logger.error(f"不支持的引擎类型: {self.engine_type}")
                return False
            
            if success:
                self.is_initialized = True
                self.is_available = True
                logger.info("✅ 推理管理器初始化成功")
            else:
                logger.error("❌ 推理管理器初始化失败")
                
            return success
            
        except Exception as e:
            logger.error(f"推理管理器初始化异常: {e}")
            return False
    
    async def inference(self, request: InferenceRequest) -> InferenceResponse:
        """执行推理"""
        if not self.is_available:
            raise RuntimeError("Inference manager not available")
        
        try:
            return await self.engine.inference(request)
        except Exception as e:
            logger.error(f"推理执行失败: {e}")
            return InferenceResponse(
                success=False,
                request_id=request.request_id or "unknown",
                result=None,
                error=str(e)
            )
    
    async def batch_inference(self, request: BatchInferenceRequest) -> BatchInferenceResponse:
        """批量推理"""
        if not self.is_available:
            raise RuntimeError("Inference manager not available")
        
        try:
            results = []
            successful_count = 0
            failed_count = 0
            
            # 并发执行推理
            tasks = [self.inference(req) for req in request.requests]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            for response in responses:
                if isinstance(response, Exception):
                    failed_count += 1
                    results.append(InferenceResponse(
                        success=False,
                        request_id="unknown",
                        result=None,
                        error=str(response)
                    ))
                else:
                    if response.success:
                        successful_count += 1
                    else:
                        failed_count += 1
                    results.append(response)
            
            return BatchInferenceResponse(
                success=True,
                results=results,
                batch_info={
                    "total_requests": len(request.requests),
                    "successful": successful_count,
                    "failed": failed_count,
                    "batch_size": request.batch_size or len(request.requests)
                },
                total_time_ms=sum(
                    r.result.inference_time_ms if r.result else 0 
                    for r in results if isinstance(r, InferenceResponse)
                )
            )
            
        except Exception as e:
            logger.error(f"批量推理执行失败: {e}")
            return BatchInferenceResponse(
                success=False,
                results=[],
                batch_info={"error": str(e)},
                total_time_ms=0
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self.is_available or not self.engine:
            return {
                "status": "unavailable",
                "engine_type": self.engine_type.value,
                "is_initialized": self.is_initialized
            }
        
        stats = self.engine.get_stats()
        stats.update({
            "manager": {
                "engine_type": self.engine_type.value,
                "is_initialized": self.is_initialized,
                "is_available": self.is_available
            }
        })
        return stats
    
    def get_models(self) -> List[Dict[str, Any]]:
        """获取模型列表"""
        if not self.is_available or not self.engine:
            return []
        
        return self.engine.get_models()
    
    def get_expert_status(self) -> List[ExpertStatus]:
        """获取专家状态"""
        if not self.is_available or not self.engine:
            return []
        
        return self.engine.get_expert_status()
    
    async def shutdown(self):
        """关闭推理管理器"""
        logger.info("关闭推理管理器...")
        
        if self.engine:
            await self.engine.shutdown()
        
        self.is_available = False
        self.is_initialized = False
        logger.info("✅ 推理管理器已关闭")


# 全局推理管理器实例
inference_manager = InferenceManager(EngineType.MOCK)
