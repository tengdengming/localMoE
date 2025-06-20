"""
API数据模型定义
定义请求和响应的数据结构
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import time


class InferenceMode(str, Enum):
    """推理模式"""
    TEXT_ONLY = "text_only"
    CODE_ONLY = "code_only"
    MULTIMODAL = "multimodal"


class SamplingParams(BaseModel):
    """采样参数"""
    temperature: float = Field(default=0.8, ge=0.0, le=2.0, description="采样温度")
    top_p: float = Field(default=0.95, ge=0.0, le=1.0, description="nucleus sampling参数")
    top_k: int = Field(default=50, ge=1, le=100, description="top-k sampling参数")
    max_tokens: int = Field(default=512, ge=1, le=2048, description="最大生成token数")
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0, description="重复惩罚")
    do_sample: bool = Field(default=True, description="是否使用采样")
    stream: bool = Field(default=False, description="是否流式输出")


class ModelConfig(BaseModel):
    """模型配置"""
    use_moe: bool = Field(default=True, description="是否使用MoE")
    top_k_experts: int = Field(default=2, ge=1, le=8, description="激活的专家数量")
    expert_selection_strategy: str = Field(default="adaptive", description="专家选择策略")
    enable_caching: bool = Field(default=True, description="是否启用缓存")
    quantization: Optional[str] = Field(default=None, description="量化类型")


class InferenceRequest(BaseModel):
    """推理请求"""
    text: Optional[str] = Field(default=None, description="输入文本")
    code: Optional[str] = Field(default=None, description="输入代码")
    mode: InferenceMode = Field(default=InferenceMode.MULTIMODAL, description="推理模式")
    sampling_params: Optional[SamplingParams] = Field(default=None, description="采样参数")
    model_config: Optional[ModelConfig] = Field(default=None, description="模型配置")
    request_id: Optional[str] = Field(default=None, description="请求ID")
    
    @validator('text', 'code')
    def validate_input(cls, v, values):
        """验证输入"""
        if not values.get('text') and not values.get('code'):
            raise ValueError("至少需要提供text或code中的一个")
        return v
    
    @validator('mode')
    def validate_mode(cls, v, values):
        """验证推理模式"""
        text = values.get('text')
        code = values.get('code')
        
        if v == InferenceMode.TEXT_ONLY and not text:
            raise ValueError("TEXT_ONLY模式需要提供text")
        if v == InferenceMode.CODE_ONLY and not code:
            raise ValueError("CODE_ONLY模式需要提供code")
        if v == InferenceMode.MULTIMODAL and not (text and code):
            raise ValueError("MULTIMODAL模式需要同时提供text和code")
        
        return v


class ExpertInfo(BaseModel):
    """专家信息"""
    expert_id: int
    device_id: int
    load: float
    memory_usage: float
    active: bool


class InferenceResult(BaseModel):
    """推理结果"""
    generated_text: str = Field(description="生成的文本")
    input_tokens: int = Field(description="输入token数量")
    output_tokens: int = Field(description="输出token数量")
    inference_time_ms: float = Field(description="推理时间(毫秒)")
    experts_used: List[int] = Field(description="使用的专家ID列表")
    model_info: Dict[str, Any] = Field(description="模型信息")


class InferenceResponse(BaseModel):
    """推理响应"""
    success: bool = Field(description="是否成功")
    request_id: str = Field(description="请求ID")
    result: Optional[InferenceResult] = Field(default=None, description="推理结果")
    error: Optional[str] = Field(default=None, description="错误信息")
    timestamp: float = Field(default_factory=time.time, description="时间戳")


class BatchInferenceRequest(BaseModel):
    """批量推理请求"""
    requests: List[InferenceRequest] = Field(description="推理请求列表")
    batch_size: Optional[int] = Field(default=None, description="批次大小")
    
    @validator('requests')
    def validate_requests(cls, v):
        if not v:
            raise ValueError("请求列表不能为空")
        if len(v) > 100:
            raise ValueError("批量请求数量不能超过100")
        return v


class BatchInferenceResponse(BaseModel):
    """批量推理响应"""
    success: bool = Field(description="是否成功")
    results: List[InferenceResponse] = Field(description="推理结果列表")
    batch_info: Dict[str, Any] = Field(description="批次信息")
    total_time_ms: float = Field(description="总处理时间(毫秒)")


class HealthStatus(BaseModel):
    """健康状态"""
    status: str = Field(description="状态")
    timestamp: float = Field(default_factory=time.time, description="时间戳")
    version: str = Field(description="版本")
    uptime_seconds: float = Field(description="运行时间(秒)")


class SystemMetrics(BaseModel):
    """系统指标"""
    gpu_metrics: Dict[str, Any] = Field(description="GPU指标")
    memory_metrics: Dict[str, Any] = Field(description="内存指标")
    inference_metrics: Dict[str, Any] = Field(description="推理指标")
    expert_metrics: Dict[str, Any] = Field(description="专家指标")
    timestamp: float = Field(default_factory=time.time, description="时间戳")


class ExpertStatus(BaseModel):
    """专家状态"""
    expert_id: int = Field(description="专家ID")
    device_id: int = Field(description="设备ID")
    status: str = Field(description="状态")
    load: float = Field(description="负载")
    memory_usage_gb: float = Field(description="内存使用(GB)")
    request_count: int = Field(description="请求数量")
    avg_latency_ms: float = Field(description="平均延迟(毫秒)")


class ExpertStatusResponse(BaseModel):
    """专家状态响应"""
    experts: List[ExpertStatus] = Field(description="专家状态列表")
    total_experts: int = Field(description="专家总数")
    active_experts: int = Field(description="活跃专家数")
    timestamp: float = Field(default_factory=time.time, description="时间戳")


class ConfigUpdateRequest(BaseModel):
    """配置更新请求"""
    config_type: str = Field(description="配置类型")
    config_data: Dict[str, Any] = Field(description="配置数据")
    
    @validator('config_type')
    def validate_config_type(cls, v):
        allowed_types = ["sampling", "model", "system", "expert"]
        if v not in allowed_types:
            raise ValueError(f"配置类型必须是: {allowed_types}")
        return v


class ConfigUpdateResponse(BaseModel):
    """配置更新响应"""
    success: bool = Field(description="是否成功")
    message: str = Field(description="消息")
    old_config: Dict[str, Any] = Field(description="旧配置")
    new_config: Dict[str, Any] = Field(description="新配置")
    timestamp: float = Field(default_factory=time.time, description="时间戳")


class StreamResponse(BaseModel):
    """流式响应"""
    token: str = Field(description="生成的token")
    is_final: bool = Field(default=False, description="是否为最后一个token")
    request_id: str = Field(description="请求ID")
    timestamp: float = Field(default_factory=time.time, description="时间戳")


class ErrorResponse(BaseModel):
    """错误响应"""
    error: str = Field(description="错误信息")
    error_code: str = Field(description="错误代码")
    details: Optional[Dict[str, Any]] = Field(default=None, description="错误详情")
    request_id: Optional[str] = Field(default=None, description="请求ID")
    timestamp: float = Field(default_factory=time.time, description="时间戳")


# 常用的响应模型
class SuccessResponse(BaseModel):
    """成功响应"""
    success: bool = Field(default=True, description="是否成功")
    message: str = Field(description="消息")
    data: Optional[Dict[str, Any]] = Field(default=None, description="数据")
    timestamp: float = Field(default_factory=time.time, description="时间戳")
