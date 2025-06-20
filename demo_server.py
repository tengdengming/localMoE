#!/usr/bin/env python3
"""
LocalMoE 演示服务器 (CPU版本)
适用于Windows环境下的功能演示
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/demo_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 请求和响应模型
class InferenceRequest(BaseModel):
    text: str
    model_config: Optional[Dict[str, Any]] = {}
    max_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.7

class InferenceResponse(BaseModel):
    generated_text: str
    model_used: str
    inference_time: float
    timestamp: str
    metadata: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    version: str
    uptime: float
    system_info: Dict[str, Any]

class ModelInfo(BaseModel):
    name: str
    type: str
    status: str
    parameters: Dict[str, Any]

# 模拟的推理引擎
class MockInferenceEngine:
    """模拟推理引擎，用于演示"""
    
    def __init__(self):
        self.model_name = "LocalMoE-Demo-CPU"
        self.start_time = time.time()
        self.request_count = 0
        
        # 模拟的响应模板
        self.response_templates = [
            "基于您的输入 '{input}', 我理解您想要了解相关信息。这是一个演示响应。",
            "您提到了 '{input}', 这是一个有趣的话题。在LocalMoE系统中，我们可以处理各种类型的查询。",
            "关于 '{input}' 的问题，我可以为您提供以下见解：这是一个CPU版本的演示服务。",
            "感谢您的输入 '{input}'。LocalMoE演示系统正在为您生成响应...",
        ]
    
    async def generate(self, text: str, **kwargs) -> str:
        """模拟文本生成"""
        self.request_count += 1
        
        # 模拟推理延迟
        await asyncio.sleep(0.5)
        
        # 选择响应模板
        template_idx = self.request_count % len(self.response_templates)
        template = self.response_templates[template_idx]
        
        # 生成响应
        response = template.format(input=text[:50])
        
        # 添加一些随机内容
        if kwargs.get('max_tokens', 100) > 50:
            response += f"\n\n这是第 {self.request_count} 次推理请求。"
            response += f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return response
    
    def get_stats(self) -> Dict[str, Any]:
        """获取引擎统计信息"""
        return {
            "model_name": self.model_name,
            "uptime": time.time() - self.start_time,
            "total_requests": self.request_count,
            "status": "running",
            "engine_type": "mock_cpu"
        }

# 创建FastAPI应用
app = FastAPI(
    title="LocalMoE Demo Server",
    description="LocalMoE 演示服务器 (CPU版本)",
    version="1.0.0-demo"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量
inference_engine = MockInferenceEngine()
server_start_time = time.time()

@app.on_event("startup")
async def startup_event():
    """服务启动事件"""
    logger.info("🚀 LocalMoE Demo Server 启动中...")
    logger.info("📍 运行环境: Windows CPU")
    logger.info("⚠️  注意: 这是演示版本，DeepSpeed已被注释掉")
    logger.info("✅ 服务启动完成")

@app.on_event("shutdown")
async def shutdown_event():
    """服务关闭事件"""
    logger.info("🛑 LocalMoE Demo Server 正在关闭...")

@app.get("/", response_model=Dict[str, str])
async def root():
    """根路径"""
    return {
        "message": "LocalMoE Demo Server",
        "status": "running",
        "version": "1.0.0-demo",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    uptime = time.time() - server_start_time
    
    return HealthResponse(
        status="healthy",
        version="1.0.0-demo",
        uptime=uptime,
        system_info={
            "platform": "windows",
            "engine": "mock_cpu",
            "deepspeed_enabled": False,
            "vllm_enabled": False,
            "demo_mode": True
        }
    )

@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """列出可用模型"""
    return [
        ModelInfo(
            name="LocalMoE-Demo-CPU",
            type="mock",
            status="ready",
            parameters={
                "max_tokens": 500,
                "temperature_range": [0.1, 2.0],
                "supports_streaming": False,
                "demo_mode": True
            }
        )
    ]

@app.post("/v1/inference", response_model=InferenceResponse)
async def inference(request: InferenceRequest):
    """推理接口"""
    start_time = time.time()
    
    try:
        logger.info(f"收到推理请求: {request.text[:100]}...")
        
        # 验证输入
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="输入文本不能为空")
        
        if len(request.text) > 1000:
            raise HTTPException(status_code=400, detail="输入文本过长 (最大1000字符)")
        
        # 执行推理
        generated_text = await inference_engine.generate(
            request.text,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        inference_time = time.time() - start_time
        
        logger.info(f"推理完成，耗时: {inference_time:.2f}秒")
        
        return InferenceResponse(
            generated_text=generated_text,
            model_used="LocalMoE-Demo-CPU",
            inference_time=inference_time,
            timestamp=datetime.now().isoformat(),
            metadata={
                "request_length": len(request.text),
                "response_length": len(generated_text),
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "demo_mode": True
            }
        )
        
    except Exception as e:
        logger.error(f"推理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"推理失败: {str(e)}")

@app.get("/v1/stats")
async def get_stats():
    """获取统计信息"""
    engine_stats = inference_engine.get_stats()
    
    return {
        "server": {
            "uptime": time.time() - server_start_time,
            "version": "1.0.0-demo",
            "platform": "windows"
        },
        "engine": engine_stats,
        "system": {
            "demo_mode": True,
            "deepspeed_enabled": False,
            "vllm_enabled": False,
            "gpu_available": False
        }
    }

@app.get("/v1/config")
async def get_config():
    """获取配置信息"""
    return {
        "model": {
            "name": "LocalMoE-Demo-CPU",
            "type": "mock",
            "quantization": "none",
            "max_sequence_length": 1000
        },
        "inference": {
            "engine": "mock",
            "max_concurrent_requests": 10,
            "timeout": 30
        },
        "deployment": {
            "environment": "demo",
            "platform": "windows",
            "gpu_count": 0,
            "deepspeed_commented": True
        }
    }

if __name__ == "__main__":
    print("🚀 启动 LocalMoE Demo Server...")
    print("📍 运行环境: Windows CPU")
    print("⚠️  注意: DeepSpeed已被注释掉，这是演示版本")
    print("🌐 服务地址: http://localhost:8000")
    print("📚 API文档: http://localhost:8000/docs")
    
    # 启动服务
    uvicorn.run(
        "demo_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
