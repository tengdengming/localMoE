"""
API数据模型定义
定义请求和响应的数据结构
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import time
from datetime import datetime


class InferenceMode(str, Enum):
    """推理模式"""
    TEXT_ONLY = "text_only"
    CODE_ONLY = "code_only"
    MULTIMODAL = "multimodal"
    AUTO = "auto"


class InferenceEngine(str, Enum):
    """推理引擎类型"""
    DEEPSPEED = "deepspeed"
    VLLM = "vllm"
    AUTO = "auto"


class SamplingParams(BaseModel):
    """采样参数"""
    temperature: float = Field(default=0.8, description="采样温度")
    top_p: float = Field(default=0.95, description="nucleus sampling参数")
    top_k: int = Field(default=50, description="top-k sampling参数")
    max_tokens: int = Field(default=512, description="最大生成token数")
    repetition_penalty: float = Field(default=1.1, description="重复惩罚")
    do_sample: bool = Field(default=True, description="是否使用采样")
    stream: bool = Field(default=False, description="是否流式输出")


class ModelConfig(BaseModel):
    """模型配置"""
    use_moe: bool = Field(default=True, description="是否使用MoE")
    top_k_experts: int = Field(default=2, description="激活的专家数量")
    expert_selection_strategy: str = Field(default="adaptive", description="专家选择策略")
    enable_caching: bool = Field(default=True, description="是否启用缓存")
    quantization: Optional[str] = Field(default=None, description="量化类型")


class InferenceRequest(BaseModel):
    """推理请求"""
    text: Optional[str] = None
    code: Optional[str] = None
    mode: InferenceMode = InferenceMode.AUTO
    sampling_params: Optional[SamplingParams] = None
    model_settings: Optional[ModelConfig] = None
    request_id: Optional[str] = None
    
    @model_validator(mode='after')
    def validate_input_and_mode(self):
        """验证输入和模式"""
        text = self.text
        code = self.code
        mode = self.mode
        model_settings = self.model_settings

        # 验证至少有一个输入
        if not text and not code:
            raise ValueError("至少需要提供text或code中的一个")

        # 验证模式匹配
        if mode == InferenceMode.TEXT_ONLY and not text:
            raise ValueError("TEXT_ONLY模式需要提供text")
        if mode == InferenceMode.CODE_ONLY and not code:
            raise ValueError("CODE_ONLY模式需要提供code")
        if mode == InferenceMode.MULTIMODAL and not (text and code):
            raise ValueError("MULTIMODAL模式需要同时提供text和code")
        # AUTO模式会自动根据输入选择合适的模式

        return self


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
    
    @field_validator('requests')
    @classmethod
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
    
    @field_validator('config_type')
    @classmethod
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


# 新增模型类
class ExpertInfo(BaseModel):
    """专家信息"""
    expert_id: int = Field(description="专家ID")
    device_id: int = Field(description="设备ID")
    load: float = Field(description="负载")
    memory_usage: float = Field(description="内存使用")
    active: bool = Field(description="是否激活")


class InferenceResult(BaseModel):
    """推理结果"""
    generated_text: str = Field(description="生成的文本")
    input_tokens: int = Field(description="输入token数")
    output_tokens: int = Field(description="输出token数")
    inference_time_ms: float = Field(description="推理时间(毫秒)")
    experts_used: List[int] = Field(default_factory=list, description="使用的专家")
    model_info: Dict[str, Any] = Field(default_factory=dict, description="模型信息")


class InferenceResponse(BaseModel):
    """推理响应"""
    success: bool = Field(description="是否成功")
    request_id: str = Field(description="请求ID")
    result: Optional[InferenceResult] = Field(default=None, description="推理结果")
    error: Optional[str] = Field(default=None, description="错误信息")
    timestamp: float = Field(default_factory=time.time, description="时间戳")


class BatchInferenceResponse(BaseModel):
    """批量推理响应"""
    success: bool = Field(description="是否成功")
    results: List[InferenceResponse] = Field(description="推理结果列表")
    batch_info: Dict[str, Any] = Field(description="批次信息")
    total_time_ms: float = Field(description="总时间(毫秒)")


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


class SystemMetrics(BaseModel):
    """系统指标"""
    cpu_usage: float = Field(description="CPU使用率")
    memory_usage: float = Field(description="内存使用率")
    gpu_usage: List[float] = Field(description="GPU使用率")
    disk_usage: float = Field(description="磁盘使用率")
