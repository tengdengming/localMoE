"""
FastAPI服务接口模块
提供REST API接口，支持多模态MoE推理服务
"""

from .main import app
from .models import *
from .routes import *

__all__ = ["app"]
