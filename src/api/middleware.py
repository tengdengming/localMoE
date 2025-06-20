"""
FastAPI中间件
提供请求日志、限流等功能
"""

import time
import logging
from typing import Dict, Any
from collections import defaultdict, deque
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """请求日志中间件"""
    
    def __init__(self, app, log_level: str = "INFO"):
        super().__init__(app)
        self.log_level = getattr(logging, log_level.upper())
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # 记录请求开始
        logger.log(
            self.log_level,
            f"Request started: {request.method} {request.url.path}"
        )
        
        # 处理请求
        response = await call_next(request)
        
        # 计算处理时间
        process_time = time.time() - start_time
        
        # 记录请求完成
        logger.log(
            self.log_level,
            f"Request completed: {request.method} {request.url.path} "
            f"- Status: {response.status_code} - Time: {process_time:.3f}s"
        )
        
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """限流中间件"""
    
    def __init__(self, app, calls: int = 100, period: int = 60):
        super().__init__(app)
        self.calls = calls  # 允许的调用次数
        self.period = period  # 时间窗口（秒）
        self.clients = defaultdict(lambda: deque())
    
    async def dispatch(self, request: Request, call_next):
        # 获取客户端IP
        client_ip = self._get_client_ip(request)
        
        # 检查限流
        if self._is_rate_limited(client_ip):
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "error_code": "RATE_LIMIT_EXCEEDED",
                    "details": {
                        "limit": self.calls,
                        "period": self.period,
                        "retry_after": self.period
                    }
                },
                headers={"Retry-After": str(self.period)}
            )
        
        # 记录请求
        self._record_request(client_ip)
        
        # 处理请求
        response = await call_next(request)
        
        # 添加限流头部
        remaining = self._get_remaining_calls(client_ip)
        response.headers["X-RateLimit-Limit"] = str(self.calls)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(time.time()) + self.period)
        
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """获取客户端IP"""
        # 检查代理头部
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # 使用客户端IP
        return request.client.host if request.client else "unknown"
    
    def _is_rate_limited(self, client_ip: str) -> bool:
        """检查是否超过限流"""
        now = time.time()
        client_requests = self.clients[client_ip]
        
        # 清理过期请求
        while client_requests and client_requests[0] <= now - self.period:
            client_requests.popleft()
        
        # 检查是否超过限制
        return len(client_requests) >= self.calls
    
    def _record_request(self, client_ip: str):
        """记录请求"""
        now = time.time()
        self.clients[client_ip].append(now)
    
    def _get_remaining_calls(self, client_ip: str) -> int:
        """获取剩余调用次数"""
        client_requests = self.clients[client_ip]
        return max(0, self.calls - len(client_requests))


class SecurityMiddleware(BaseHTTPMiddleware):
    """安全中间件"""
    
    def __init__(self, app):
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next):
        # 处理请求
        response = await call_next(request)
        
        # 添加安全头部
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        return response


class MetricsMiddleware(BaseHTTPMiddleware):
    """指标收集中间件"""
    
    def __init__(self, app):
        super().__init__(app)
        self.metrics = {
            "request_count": 0,
            "error_count": 0,
            "response_times": deque(maxlen=1000),
            "status_codes": defaultdict(int),
            "endpoints": defaultdict(int)
        }
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # 增加请求计数
        self.metrics["request_count"] += 1
        
        # 记录端点访问
        endpoint = f"{request.method} {request.url.path}"
        self.metrics["endpoints"][endpoint] += 1
        
        # 处理请求
        response = await call_next(request)
        
        # 计算响应时间
        response_time = time.time() - start_time
        self.metrics["response_times"].append(response_time)
        
        # 记录状态码
        self.metrics["status_codes"][response.status_code] += 1
        
        # 记录错误
        if response.status_code >= 400:
            self.metrics["error_count"] += 1
        
        # 添加指标头部
        response.headers["X-Request-Count"] = str(self.metrics["request_count"])
        response.headers["X-Response-Time"] = f"{response_time:.3f}"
        
        return response
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取指标"""
        response_times = list(self.metrics["response_times"])
        
        if response_times:
            import statistics
            avg_response_time = statistics.mean(response_times)
            p95_response_time = statistics.quantiles(response_times, n=20)[18]  # 95th percentile
        else:
            avg_response_time = 0
            p95_response_time = 0
        
        return {
            "request_count": self.metrics["request_count"],
            "error_count": self.metrics["error_count"],
            "error_rate": self.metrics["error_count"] / max(1, self.metrics["request_count"]),
            "avg_response_time": avg_response_time,
            "p95_response_time": p95_response_time,
            "status_codes": dict(self.metrics["status_codes"]),
            "top_endpoints": dict(sorted(
                self.metrics["endpoints"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:10])
        }


class CORSMiddleware(BaseHTTPMiddleware):
    """自定义CORS中间件"""
    
    def __init__(
        self,
        app,
        allow_origins: list = None,
        allow_methods: list = None,
        allow_headers: list = None,
        allow_credentials: bool = False
    ):
        super().__init__(app)
        self.allow_origins = allow_origins or ["*"]
        self.allow_methods = allow_methods or ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
        self.allow_headers = allow_headers or ["*"]
        self.allow_credentials = allow_credentials
    
    async def dispatch(self, request: Request, call_next):
        # 处理预检请求
        if request.method == "OPTIONS":
            response = Response()
            self._add_cors_headers(response, request)
            return response
        
        # 处理实际请求
        response = await call_next(request)
        self._add_cors_headers(response, request)
        
        return response
    
    def _add_cors_headers(self, response: Response, request: Request):
        """添加CORS头部"""
        origin = request.headers.get("origin")
        
        if self.allow_origins == ["*"] or origin in self.allow_origins:
            response.headers["Access-Control-Allow-Origin"] = origin or "*"
        
        response.headers["Access-Control-Allow-Methods"] = ", ".join(self.allow_methods)
        response.headers["Access-Control-Allow-Headers"] = ", ".join(self.allow_headers)
        
        if self.allow_credentials:
            response.headers["Access-Control-Allow-Credentials"] = "true"
