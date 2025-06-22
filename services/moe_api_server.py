#!/usr/bin/env python3
"""
vLLM MoE模型 FastAPI服务包装器
提供统一的REST API接口，支持多种MoE模型
"""

import os
import sys
import time
import asyncio
import logging
from typing import Dict, List, Optional, Any, AsyncGenerator
from contextlib import asynccontextmanager

import yaml
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# vLLM imports
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.utils import random_uuid

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局变量
engine: Optional[AsyncLLMEngine] = None
model_config: Dict[str, Any] = {}

# Pydantic模型
class ChatMessage(BaseModel):
    role: str = Field(..., description="消息角色: system, user, assistant")
    content: str = Field(..., description="消息内容")

class ChatCompletionRequest(BaseModel):
    model: str = Field(default="current", description="模型名称")
    messages: List[ChatMessage] = Field(..., description="对话消息列表")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="采样温度")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="核采样概率")
    top_k: int = Field(default=50, ge=1, description="Top-K采样")
    max_tokens: int = Field(default=1024, ge=1, le=8192, description="最大生成长度")
    stream: bool = Field(default=False, description="是否流式输出")
    stop: Optional[List[str]] = Field(default=None, description="停止词列表")

class CompletionRequest(BaseModel):
    model: str = Field(default="current", description="模型名称")
    prompt: str = Field(..., description="输入提示")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="采样温度")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="核采样概率")
    top_k: int = Field(default=50, ge=1, description="Top-K采样")
    max_tokens: int = Field(default=1024, ge=1, le=8192, description="最大生成长度")
    stream: bool = Field(default=False, description="是否流式输出")
    stop: Optional[List[str]] = Field(default=None, description="停止词列表")
    n: int = Field(default=1, ge=1, le=5, description="生成数量")

class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "vllm-moe"

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Usage

class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Usage

# 应用生命周期管理
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化
    logger.info("🚀 启动vLLM MoE API服务...")
    await initialize_engine()
    yield
    # 关闭时清理
    logger.info("🛑 关闭vLLM MoE API服务...")
    if engine:
        # 这里可以添加清理逻辑
        pass

# 创建FastAPI应用
app = FastAPI(
    title="vLLM MoE API Server",
    description="基于vLLM的MoE模型API服务",
    version="1.0.0",
    lifespan=lifespan
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def initialize_engine():
    """初始化vLLM引擎"""
    global engine, model_config
    
    try:
        # 从环境变量获取配置
        model_name = os.getenv("VLLM_MODEL", "mistralai/Mixtral-8x7B-Instruct-v0.1")
        tensor_parallel_size = int(os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "2"))
        gpu_memory_utilization = float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.85"))
        max_model_len = int(os.getenv("VLLM_MAX_MODEL_LEN", "8192"))
        max_num_seqs = int(os.getenv("VLLM_MAX_NUM_SEQS", "256"))
        dtype = os.getenv("VLLM_DTYPE", "bfloat16")
        
        logger.info(f"初始化模型: {model_name}")
        logger.info(f"张量并行: {tensor_parallel_size}")
        logger.info(f"GPU内存利用率: {gpu_memory_utilization}")
        
        # 创建引擎参数
        engine_args = AsyncEngineArgs(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            dtype=dtype,
            trust_remote_code=True,
            enforce_eager=False,
            worker_use_ray=False,
            engine_use_ray=False,
        )
        
        # 创建异步引擎
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        
        # 保存模型配置
        model_config = {
            "model_name": model_name,
            "tensor_parallel_size": tensor_parallel_size,
            "max_model_len": max_model_len,
            "dtype": dtype
        }
        
        logger.info("✅ vLLM引擎初始化完成")
        
    except Exception as e:
        logger.error(f"❌ 引擎初始化失败: {e}")
        sys.exit(1)

def format_chat_prompt(messages: List[ChatMessage]) -> str:
    """格式化聊天提示"""
    # 简单的聊天格式，可以根据具体模型调整
    formatted_messages = []
    for msg in messages:
        if msg.role == "system":
            formatted_messages.append(f"System: {msg.content}")
        elif msg.role == "user":
            formatted_messages.append(f"User: {msg.content}")
        elif msg.role == "assistant":
            formatted_messages.append(f"Assistant: {msg.content}")
    
    formatted_messages.append("Assistant:")
    return "\n".join(formatted_messages)

async def generate_stream(request_id: str, prompt: str, sampling_params: SamplingParams) -> AsyncGenerator[str, None]:
    """流式生成"""
    async for request_output in engine.generate(prompt, sampling_params, request_id):
        for output in request_output.outputs:
            yield output.text

@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "vLLM MoE API Server",
        "model": model_config.get("model_name", "unknown"),
        "status": "running"
    }

@app.get("/v1/models")
async def list_models():
    """列出可用模型"""
    return {
        "object": "list",
        "data": [
            ModelInfo(
                id=model_config.get("model_name", "current"),
                created=int(time.time()),
                owned_by="vllm-moe"
            )
        ]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """聊天补全接口"""
    if not engine:
        raise HTTPException(status_code=503, detail="引擎未初始化")
    
    # 格式化提示
    prompt = format_chat_prompt(request.messages)
    
    # 创建采样参数
    sampling_params = SamplingParams(
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        max_tokens=request.max_tokens,
        stop=request.stop,
        use_beam_search=False,
    )
    
    request_id = random_uuid()
    
    if request.stream:
        # 流式响应
        async def stream_generator():
            async for text in generate_stream(request_id, prompt, sampling_params):
                chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": text},
                        "finish_reason": None
                    }]
                }
                yield f"data: {chunk}\n\n"
            
            # 结束标记
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(stream_generator(), media_type="text/plain")
    
    else:
        # 非流式响应
        results = []
        async for request_output in engine.generate(prompt, sampling_params, request_id):
            results.append(request_output)
        
        if not results:
            raise HTTPException(status_code=500, detail="生成失败")
        
        final_output = results[-1]
        generated_text = final_output.outputs[0].text
        
        return ChatCompletionResponse(
            id=request_id,
            created=int(time.time()),
            model=request.model,
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": generated_text
                },
                "finish_reason": "stop"
            }],
            usage=Usage(
                prompt_tokens=len(final_output.prompt_token_ids),
                completion_tokens=len(final_output.outputs[0].token_ids),
                total_tokens=len(final_output.prompt_token_ids) + len(final_output.outputs[0].token_ids)
            )
        )

@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    """文本补全接口"""
    if not engine:
        raise HTTPException(status_code=503, detail="引擎未初始化")

    # 创建采样参数
    sampling_params = SamplingParams(
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        max_tokens=request.max_tokens,
        stop=request.stop,
        n=request.n,
        use_beam_search=False,
    )

    request_id = random_uuid()

    if request.stream:
        # 流式响应
        async def stream_generator():
            async for text in generate_stream(request_id, request.prompt, sampling_params):
                chunk = {
                    "id": request_id,
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [{
                        "index": 0,
                        "text": text,
                        "finish_reason": None
                    }]
                }
                yield f"data: {chunk}\n\n"

            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_generator(), media_type="text/plain")

    else:
        # 非流式响应
        results = []
        async for request_output in engine.generate(request.prompt, sampling_params, request_id):
            results.append(request_output)

        if not results:
            raise HTTPException(status_code=500, detail="生成失败")

        final_output = results[-1]
        choices = []

        for i, output in enumerate(final_output.outputs):
            choices.append({
                "index": i,
                "text": output.text,
                "finish_reason": "stop"
            })

        return CompletionResponse(
            id=request_id,
            created=int(time.time()),
            model=request.model,
            choices=choices,
            usage=Usage(
                prompt_tokens=len(final_output.prompt_token_ids),
                completion_tokens=sum(len(output.token_ids) for output in final_output.outputs),
                total_tokens=len(final_output.prompt_token_ids) + sum(len(output.token_ids) for output in final_output.outputs)
            )
        )

@app.get("/health")
async def health_check():
    """健康检查"""
    if not engine:
        raise HTTPException(status_code=503, detail="引擎未就绪")

    return {
        "status": "healthy",
        "model": model_config.get("model_name", "unknown"),
        "timestamp": int(time.time())
    }

@app.get("/metrics")
async def get_metrics():
    """获取服务指标"""
    if not engine:
        raise HTTPException(status_code=503, detail="引擎未就绪")

    # 这里可以添加更详细的指标收集
    return {
        "model": model_config.get("model_name", "unknown"),
        "tensor_parallel_size": model_config.get("tensor_parallel_size", 1),
        "max_model_len": model_config.get("max_model_len", 4096),
        "dtype": model_config.get("dtype", "float16"),
        "uptime": int(time.time()),
        "status": "running"
    }

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="vLLM MoE API服务器")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务器地址")
    parser.add_argument("--port", type=int, default=8000, help="服务器端口")
    parser.add_argument("--workers", type=int, default=1, help="工作进程数")
    parser.add_argument("--log-level", type=str, default="info", help="日志级别")

    args = parser.parse_args()

    # 启动服务器
    uvicorn.run(
        "moe_api_server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level=args.log_level,
        access_log=True,
        reload=False
    )

if __name__ == "__main__":
    main()
