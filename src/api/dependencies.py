"""
FastAPI依赖注入
提供各种服务的依赖注入
"""

from fastapi import HTTPException, Request
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def get_app_state(request: Request) -> Dict[str, Any]:
    """获取应用状态"""
    return request.app.state.__dict__.get("app_state", {})


def get_inference_manager(request: Request):
    """获取推理管理器"""
    app_state = get_app_state(request)
    inference_manager = app_state.get("inference_manager")
    
    if inference_manager is None:
        raise HTTPException(
            status_code=503, 
            detail="Inference manager not available"
        )
    
    return inference_manager


def get_feature_extractor(request: Request):
    """获取特征提取器"""
    app_state = get_app_state(request)
    feature_extractor = app_state.get("feature_extractor")
    
    if feature_extractor is None:
        raise HTTPException(
            status_code=503, 
            detail="Feature extractor not available"
        )
    
    return feature_extractor


def get_config_manager(request: Request):
    """获取配置管理器"""
    # 简化实现，返回一个模拟的配置管理器
    class MockConfigManager:
        def __init__(self):
            self.config = {}
        
        def get_config(self, key: str):
            return self.config.get(key)
        
        def set_config(self, key: str, value: Any):
            self.config[key] = value
            return True
    
    return MockConfigManager()


def get_current_user(request: Request) -> Optional[Dict[str, Any]]:
    """获取当前用户（如果启用了认证）"""
    # 简化实现，实际应该从JWT token或session中获取
    return {
        "user_id": "anonymous",
        "permissions": ["read", "write"]
    }


def check_permission(permission: str):
    """检查权限装饰器"""
    def dependency(user: Dict[str, Any] = get_current_user):
        if user is None:
            raise HTTPException(status_code=401, detail="Authentication required")
        
        user_permissions = user.get("permissions", [])
        if permission not in user_permissions:
            raise HTTPException(
                status_code=403, 
                detail=f"Permission '{permission}' required"
            )
        
        return user
    
    return dependency


def validate_request_size(max_size_mb: int = 10):
    """验证请求大小"""
    def dependency(request: Request):
        content_length = request.headers.get("content-length")
        if content_length:
            size_mb = int(content_length) / (1024 * 1024)
            if size_mb > max_size_mb:
                raise HTTPException(
                    status_code=413,
                    detail=f"Request too large. Maximum size: {max_size_mb}MB"
                )
        return True
    
    return dependency
