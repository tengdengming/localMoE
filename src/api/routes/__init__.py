"""
API路由模块
"""

from . import inference_new as inference, monitoring, management

__all__ = ["inference", "monitoring", "management"]
