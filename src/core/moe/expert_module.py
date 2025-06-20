"""
专家模块实现 - 支持动态量化和计算图优化
基于DesignPlan.md的专家参数量化策略
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Union
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class QuantizationType(Enum):
    """量化类型枚举"""
    FP16 = "fp16"
    FP8 = "fp8"
    INT8 = "int8"
    INT4 = "int4"
    AWQ_3BIT = "awq_3bit"
    GPTQ_4BIT = "gptq_4bit"


@dataclass
class ExpertConfig:
    """专家配置"""
    expert_id: int
    hidden_size: int
    intermediate_size: int
    activation: str = "gelu"
    dropout: float = 0.1
    quantization: Optional[QuantizationType] = None
    device: torch.device = torch.device("cuda:0")


class QuantizedLinear(nn.Module):
    """量化线性层"""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        quantization_type: QuantizationType,
        bias: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quantization_type = quantization_type
        
        # 根据量化类型初始化权重
        if quantization_type == QuantizationType.FP8:
            self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.float8_e4m3fn))
        elif quantization_type == QuantizationType.INT8:
            self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.int8))
            self.scale = nn.Parameter(torch.ones(out_features))
        elif quantization_type == QuantizationType.INT4:
            # INT4需要特殊处理，这里简化为INT8
            self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.int8))
            self.scale = nn.Parameter(torch.ones(out_features))
        else:
            self.weight = nn.Parameter(torch.empty(out_features, in_features))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        if self.quantization_type in [QuantizationType.FP8, QuantizationType.FP16]:
            nn.init.normal_(self.weight, mean=0.0, std=0.02)
        else:
            # 量化权重需要特殊初始化
            with torch.no_grad():
                weight_fp32 = torch.randn(self.out_features, self.in_features) * 0.02
                self.weight.copy_(self._quantize_weight(weight_fp32))
        
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def _quantize_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """量化权重"""
        if self.quantization_type == QuantizationType.INT8:
            # 简单的对称量化
            scale = weight.abs().max() / 127.0
            quantized = torch.round(weight / scale).clamp(-128, 127).to(torch.int8)
            self.scale.data.fill_(scale)
            return quantized
        elif self.quantization_type == QuantizationType.INT4:
            # INT4量化（简化实现）
            scale = weight.abs().max() / 7.0
            quantized = torch.round(weight / scale).clamp(-8, 7).to(torch.int8)
            self.scale.data.fill_(scale)
            return quantized
        else:
            return weight
    
    def _dequantize_weight(self) -> torch.Tensor:
        """反量化权重"""
        if self.quantization_type in [QuantizationType.INT8, QuantizationType.INT4]:
            return self.weight.float() * self.scale.unsqueeze(1)
        else:
            return self.weight.float()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        if self.quantization_type in [QuantizationType.INT8, QuantizationType.INT4]:
            weight = self._dequantize_weight()
        else:
            weight = self.weight
        
        return F.linear(x, weight, self.bias)


class ExpertFFN(nn.Module):
    """
    专家前馈网络
    实现高效的FFN计算
    """
    
    def __init__(self, config: ExpertConfig):
        super().__init__()
        self.config = config
        
        # 第一层：up projection
        self.gate_proj = QuantizedLinear(
            config.hidden_size,
            config.intermediate_size,
            config.quantization or QuantizationType.FP16,
            bias=False
        )
        
        # 第二层：gate projection (用于SwiGLU激活)
        self.up_proj = QuantizedLinear(
            config.hidden_size,
            config.intermediate_size,
            config.quantization or QuantizationType.FP16,
            bias=False
        )
        
        # 第三层：down projection
        self.down_proj = QuantizedLinear(
            config.intermediate_size,
            config.hidden_size,
            config.quantization or QuantizationType.FP16,
            bias=False
        )
        
        # 激活函数
        self.activation = self._get_activation(config.activation)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
    
    def _get_activation(self, activation: str):
        """获取激活函数"""
        if activation == "gelu":
            return F.gelu
        elif activation == "relu":
            return F.relu
        elif activation == "silu" or activation == "swish":
            return F.silu
        elif activation == "swiglu":
            return lambda x: F.silu(x) * x  # SwiGLU激活
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: [batch_size, seq_len, hidden_size]
            
        Returns:
            torch.Tensor: [batch_size, seq_len, hidden_size]
        """
        if self.config.activation == "swiglu":
            # SwiGLU: gate * silu(up)
            gate = self.gate_proj(x)
            up = self.up_proj(x)
            intermediate = F.silu(gate) * up
        else:
            # 标准FFN
            intermediate = self.activation(self.gate_proj(x))
        
        # Down projection
        output = self.down_proj(intermediate)
        output = self.dropout(output)
        
        return output


class Expert(nn.Module):
    """
    单个专家模块
    包含完整的专家计算逻辑
    """
    
    def __init__(self, config: ExpertConfig):
        super().__init__()
        self.config = config
        self.expert_id = config.expert_id
        
        # 专家FFN
        self.ffn = ExpertFFN(config)
        
        # 层归一化
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        
        # 统计信息
        self.forward_count = 0
        self.total_tokens = 0
        
        logger.info(f"Expert {config.expert_id} initialized on {config.device}")
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        专家前向传播
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: 注意力掩码
            
        Returns:
            torch.Tensor: 专家输出
        """
        # 更新统计信息
        self.forward_count += 1
        self.total_tokens += hidden_states.numel() // hidden_states.size(-1)
        
        # 残差连接 + 层归一化
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # FFN计算
        ffn_output = self.ffn(hidden_states)
        
        # 残差连接
        hidden_states = residual + ffn_output
        
        # 后层归一化
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        return hidden_states
    
    def get_stats(self) -> Dict[str, Any]:
        """获取专家统计信息"""
        return {
            "expert_id": self.expert_id,
            "forward_count": self.forward_count,
            "total_tokens": self.total_tokens,
            "avg_tokens_per_forward": self.total_tokens / max(1, self.forward_count),
            "device": str(self.config.device),
            "quantization": self.config.quantization.value if self.config.quantization else None
        }
    
    def reset_stats(self):
        """重置统计信息"""
        self.forward_count = 0
        self.total_tokens = 0


# 使用torch.compile优化的专家前向传播
try:
    @torch.compile(
        mode='max-autotune',
        fullgraph=True,
        dynamic=True,
        options={
            'shape_padding': True,
            'triton.cudagraphs': True
        }
    )
    def expert_forward_compiled(expert: Expert, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        编译优化的专家前向传播
        基于DesignPlan.md的计算图优化
        """
        return expert(hidden_states)
except Exception as e:
    logger.warning(f"torch.compile not available: {e}")
    def expert_forward_compiled(expert: Expert, hidden_states: torch.Tensor) -> torch.Tensor:
        return expert(hidden_states)


class ExpertPool:
    """
    专家池管理器
    管理多个专家的生命周期和调度
    """
    
    def __init__(self, expert_configs: List[ExpertConfig]):
        self.experts: Dict[int, Expert] = {}
        self.expert_configs = {config.expert_id: config for config in expert_configs}
        
        # 初始化专家
        for config in expert_configs:
            expert = Expert(config)
            expert.to(config.device)
            self.experts[config.expert_id] = expert
        
        logger.info(f"ExpertPool initialized with {len(self.experts)} experts")
    
    def get_expert(self, expert_id: int) -> Optional[Expert]:
        """获取专家"""
        return self.experts.get(expert_id)
    
    def forward_expert(
        self, 
        expert_id: int, 
        hidden_states: torch.Tensor,
        use_compilation: bool = True
    ) -> Optional[torch.Tensor]:
        """
        执行专家前向传播
        
        Args:
            expert_id: 专家ID
            hidden_states: 输入状态
            use_compilation: 是否使用编译优化
            
        Returns:
            专家输出或None（如果专家不存在）
        """
        expert = self.get_expert(expert_id)
        if expert is None:
            logger.warning(f"Expert {expert_id} not found")
            return None
        
        # 确保输入在正确的设备上
        device = expert.config.device
        if hidden_states.device != device:
            hidden_states = hidden_states.to(device)
        
        # 执行前向传播
        if use_compilation:
            return expert_forward_compiled(expert, hidden_states)
        else:
            return expert(hidden_states)
    
    def get_all_stats(self) -> Dict[int, Dict[str, Any]]:
        """获取所有专家的统计信息"""
        return {expert_id: expert.get_stats() for expert_id, expert in self.experts.items()}
    
    def reset_all_stats(self):
        """重置所有专家的统计信息"""
        for expert in self.experts.values():
            expert.reset_stats()
    
    def move_expert_to_device(self, expert_id: int, device: torch.device):
        """将专家移动到指定设备"""
        if expert_id in self.experts:
            self.experts[expert_id].to(device)
            self.expert_configs[expert_id].device = device
            logger.info(f"Moved expert {expert_id} to {device}")
    
    def get_expert_memory_usage(self) -> Dict[int, Dict[str, float]]:
        """获取专家内存使用情况"""
        memory_usage = {}
        for expert_id, expert in self.experts.items():
            device = expert.config.device
            if device.type == 'cuda':
                torch.cuda.set_device(device)
                allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
                reserved = torch.cuda.memory_reserved(device) / 1024**3    # GB
                memory_usage[expert_id] = {
                    "allocated_gb": allocated,
                    "reserved_gb": reserved,
                    "device": str(device)
                }
        return memory_usage
