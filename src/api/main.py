"""
FastAPI主应用
多模态MoE推理服务的主入口
"""

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from .models import *
from .routes import monitoring, management
from .routes.inference_new import router as inference_router
from ..core.inference.manager import inference_manager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 全局变量
app_state = {
    "inference_manager": None,
    "feature_extractor": None,
    "start_time": time.time(),
    "request_count": 0,
    "error_count": 0
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化
    logger.info("🚀 初始化LocalMoE服务...")

    try:
        # 初始化推理管理器
        success = await inference_manager.initialize()
        if success:
            app_state["inference_manager"] = inference_manager
            logger.info("✅ 推理管理器初始化成功")
        else:
            logger.error("❌ 推理管理器初始化失败")
            raise RuntimeError("Failed to initialize inference manager")

        logger.info("🎉 LocalMoE服务启动成功")

    except Exception as e:
        logger.error(f"❌ 服务初始化失败: {e}")
        raise

    yield

    # 关闭时清理
    logger.info("🛑 关闭LocalMoE服务...")

    if inference_manager.is_available:
        await inference_manager.shutdown()

    logger.info("✅ LocalMoE服务已停止")


# 创建FastAPI应用
app = FastAPI(
    title="LocalMoE - 多模态MoE推理服务",
    description="基于DeepSpeed+vLLM的高性能多模态Mixture of Experts推理服务",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# 添加中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
# app.add_middleware(RequestLoggingMiddleware)  # 暂时注释掉
# app.add_middleware(RateLimitMiddleware, calls=100, period=60)  # 暂时注释掉


# 异常处理器
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """HTTP异常处理"""
    app_state["error_count"] += 1
    
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            error_code=f"HTTP_{exc.status_code}",
            request_id=getattr(request.state, "request_id", None)
        ).model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """通用异常处理"""
    app_state["error_count"] += 1
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            error_code="INTERNAL_ERROR",
            details={"exception_type": type(exc).__name__},
            request_id=getattr(request.state, "request_id", None)
        ).model_dump()
    )


# 根路径
@app.get("/", response_model=SuccessResponse)
async def root():
    """根路径"""
    return SuccessResponse(
        message="LocalMoE多模态MoE推理服务",
        data={
            "version": "0.1.0",
            "status": "running",
            "uptime_seconds": time.time() - app_state["start_time"],
            "docs_url": "/docs"
        }
    )


# 健康检查
@app.get("/health", response_model=HealthStatus)
async def health_check():
    """健康检查"""
    uptime = time.time() - app_state["start_time"]
    
    # 检查关键组件状态
    status = "healthy"
    if not app_state.get("inference_manager"):
        status = "unhealthy"
    if not app_state.get("feature_extractor"):
        status = "unhealthy"
    
    return HealthStatus(
        status=status,
        version="0.1.0",
        uptime_seconds=uptime
    )


# 就绪检查
@app.get("/ready")
async def readiness_check():
    """就绪检查"""
    if not app_state.get("inference_manager") or not app_state.get("feature_extractor"):
        raise HTTPException(status_code=503, detail="Service not ready")
    
    return {"status": "ready"}


# 包含路由
app.include_router(inference_router)

try:
    app.include_router(
        monitoring.router,
        prefix="/v1",
        tags=["monitoring"]
    )
except:
    logger.warning("监控路由加载失败，跳过")

try:
    app.include_router(
        management.router,
        prefix="/v1",
        tags=["management"]
    )
except:
    logger.warning("管理路由加载失败，跳过")


# 中间件函数
async def add_request_id(request, call_next):
    """添加请求ID"""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    
    return response


async def track_requests(request, call_next):
    """跟踪请求"""
    app_state["request_count"] += 1
    start_time = time.time()
    
    response = await call_next(request)
    
    # 记录请求时间
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    
    return response


# 添加中间件
app.middleware("http")(add_request_id)
app.middleware("http")(track_requests)


# 获取应用状态的依赖函数
def get_app_state():
    """获取应用状态"""
    return app_state


# 启动函数
def start_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    workers: int = 1,
    reload: bool = False,
    log_level: str = "info"
):
    """启动服务器"""
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        workers=workers,
        reload=reload,
        log_level=log_level,
        access_log=True
    )


if __name__ == "__main__":
    start_server()
