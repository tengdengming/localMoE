"""
多模态融合模块 - 融合文本和代码特征
实现多种融合策略和注意力机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class FusionStrategy(Enum):
    """融合策略枚举"""
    CONCATENATION = "concatenation"
    ATTENTION = "attention"
    CROSS_ATTENTION = "cross_attention"
    GATED_FUSION = "gated_fusion"
    MULTIHEAD_FUSION = "multihead_fusion"


@dataclass
class FusionConfig:
    """融合配置"""
    text_hidden_size: int = 768
    code_hidden_size: int = 768
    fusion_hidden_size: int = 768
    num_attention_heads: int = 12
    fusion_strategy: FusionStrategy = FusionStrategy.CROSS_ATTENTION
    dropout: float = 0.1
    num_fusion_layers: int = 2
    enable_residual: bool = True
    enable_layer_norm: bool = True
    temperature: float = 1.0


class CrossModalAttention(nn.Module):
    """跨模态注意力机制"""
    
    def __init__(self, config: FusionConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.head_dim = config.fusion_hidden_size // config.num_attention_heads
        self.scale = self.head_dim ** -0.5
        
        # 查询、键、值投影层
        self.q_proj = nn.Linear(config.text_hidden_size, config.fusion_hidden_size)
        self.k_proj = nn.Linear(config.code_hidden_size, config.fusion_hidden_size)
        self.v_proj = nn.Linear(config.code_hidden_size, config.fusion_hidden_size)
        
        # 输出投影
        self.out_proj = nn.Linear(config.fusion_hidden_size, config.fusion_hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(
        self,
        text_features: torch.Tensor,
        code_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        跨模态注意力前向传播
        
        Args:
            text_features: [batch_size, text_seq_len, text_hidden_size]
            code_features: [batch_size, code_seq_len, code_hidden_size]
            attention_mask: 注意力掩码
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (融合特征, 注意力权重)
        """
        batch_size, text_seq_len, _ = text_features.shape
        _, code_seq_len, _ = code_features.shape
        
        # 投影到查询、键、值
        Q = self.q_proj(text_features)  # [batch_size, text_seq_len, fusion_hidden_size]
        K = self.k_proj(code_features)  # [batch_size, code_seq_len, fusion_hidden_size]
        V = self.v_proj(code_features)  # [batch_size, code_seq_len, fusion_hidden_size]
        
        # 重塑为多头格式
        Q = Q.view(batch_size, text_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, code_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, code_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # 应用注意力掩码
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask.unsqueeze(1).unsqueeze(1)
        
        # 应用温度缩放
        attention_scores = attention_scores / self.config.temperature
        
        # Softmax
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力权重
        context = torch.matmul(attention_weights, V)
        
        # 重塑并投影输出
        context = context.transpose(1, 2).contiguous().view(
            batch_size, text_seq_len, self.config.fusion_hidden_size
        )
        output = self.out_proj(context)
        
        return output, attention_weights.mean(dim=1)  # 平均多头注意力权重


class GatedFusion(nn.Module):
    """门控融合机制"""
    
    def __init__(self, config: FusionConfig):
        super().__init__()
        self.config = config
        
        # 门控网络
        self.text_gate = nn.Linear(config.text_hidden_size, config.fusion_hidden_size)
        self.code_gate = nn.Linear(config.code_hidden_size, config.fusion_hidden_size)
        self.fusion_gate = nn.Linear(
            config.text_hidden_size + config.code_hidden_size, 
            config.fusion_hidden_size
        )
        
        # 特征投影
        self.text_proj = nn.Linear(config.text_hidden_size, config.fusion_hidden_size)
        self.code_proj = nn.Linear(config.code_hidden_size, config.fusion_hidden_size)
        
        # 激活函数
        self.activation = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
    def forward(
        self,
        text_features: torch.Tensor,
        code_features: torch.Tensor
    ) -> torch.Tensor:
        """
        门控融合前向传播
        
        Args:
            text_features: [batch_size, seq_len, text_hidden_size]
            code_features: [batch_size, seq_len, code_hidden_size]
            
        Returns:
            torch.Tensor: 融合特征
        """
        # 确保序列长度一致
        if text_features.size(1) != code_features.size(1):
            min_len = min(text_features.size(1), code_features.size(1))
            text_features = text_features[:, :min_len, :]
            code_features = code_features[:, :min_len, :]
        
        # 投影特征
        text_proj = self.text_proj(text_features)
        code_proj = self.code_proj(code_features)
        
        # 计算门控权重
        text_gate_weight = self.sigmoid(self.text_gate(text_features))
        code_gate_weight = self.sigmoid(self.code_gate(code_features))
        
        # 融合门控
        concat_features = torch.cat([text_features, code_features], dim=-1)
        fusion_gate_weight = self.sigmoid(self.fusion_gate(concat_features))
        
        # 门控融合
        gated_text = text_proj * text_gate_weight
        gated_code = code_proj * code_gate_weight
        
        # 最终融合
        fused_features = (gated_text + gated_code) * fusion_gate_weight
        fused_features = self.activation(fused_features)
        
        return fused_features


class MultiHeadFusion(nn.Module):
    """多头融合机制"""
    
    def __init__(self, config: FusionConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        
        # 多个融合头
        self.fusion_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.text_hidden_size + config.code_hidden_size, config.fusion_hidden_size),
                nn.ReLU(),
                nn.Dropout(config.dropout)
            )
            for _ in range(self.num_heads)
        ])
        
        # 头部权重
        self.head_weights = nn.Parameter(torch.ones(self.num_heads) / self.num_heads)
        
        # 输出投影
        self.output_proj = nn.Linear(config.fusion_hidden_size, config.fusion_hidden_size)
        
    def forward(
        self,
        text_features: torch.Tensor,
        code_features: torch.Tensor
    ) -> torch.Tensor:
        """
        多头融合前向传播
        
        Args:
            text_features: [batch_size, seq_len, text_hidden_size]
            code_features: [batch_size, seq_len, code_hidden_size]
            
        Returns:
            torch.Tensor: 融合特征
        """
        # 确保序列长度一致
        if text_features.size(1) != code_features.size(1):
            min_len = min(text_features.size(1), code_features.size(1))
            text_features = text_features[:, :min_len, :]
            code_features = code_features[:, :min_len, :]
        
        # 拼接特征
        concat_features = torch.cat([text_features, code_features], dim=-1)
        
        # 多头处理
        head_outputs = []
        for head in self.fusion_heads:
            head_output = head(concat_features)
            head_outputs.append(head_output)
        
        # 加权组合
        head_outputs = torch.stack(head_outputs, dim=0)  # [num_heads, batch_size, seq_len, hidden_size]
        weights = F.softmax(self.head_weights, dim=0).view(-1, 1, 1, 1)
        
        fused_features = (head_outputs * weights).sum(dim=0)
        fused_features = self.output_proj(fused_features)
        
        return fused_features


class MultimodalFusion(nn.Module):
    """
    多模态融合模块
    支持多种融合策略
    """
    
    def __init__(self, config: FusionConfig):
        super().__init__()
        self.config = config
        self.fusion_strategy = config.fusion_strategy
        
        # 输入投影层（确保维度一致）
        self.text_input_proj = nn.Linear(config.text_hidden_size, config.fusion_hidden_size)
        self.code_input_proj = nn.Linear(config.code_hidden_size, config.fusion_hidden_size)
        
        # 根据策略选择融合模块
        if config.fusion_strategy == FusionStrategy.CONCATENATION:
            self.fusion_layer = nn.Linear(
                config.text_hidden_size + config.code_hidden_size,
                config.fusion_hidden_size
            )
        elif config.fusion_strategy == FusionStrategy.ATTENTION:
            self.fusion_layer = nn.MultiheadAttention(
                embed_dim=config.fusion_hidden_size,
                num_heads=config.num_attention_heads,
                dropout=config.dropout,
                batch_first=True
            )
        elif config.fusion_strategy == FusionStrategy.CROSS_ATTENTION:
            self.fusion_layer = CrossModalAttention(config)
        elif config.fusion_strategy == FusionStrategy.GATED_FUSION:
            self.fusion_layer = GatedFusion(config)
        elif config.fusion_strategy == FusionStrategy.MULTIHEAD_FUSION:
            self.fusion_layer = MultiHeadFusion(config)
        else:
            raise ValueError(f"Unsupported fusion strategy: {config.fusion_strategy}")
        
        # 多层融合
        if config.num_fusion_layers > 1:
            self.fusion_layers = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=config.fusion_hidden_size,
                    nhead=config.num_attention_heads,
                    dim_feedforward=config.fusion_hidden_size * 4,
                    dropout=config.dropout,
                    activation="gelu",
                    batch_first=True
                )
                for _ in range(config.num_fusion_layers - 1)
            ])
        else:
            self.fusion_layers = None
        
        # 层归一化和残差连接
        if config.enable_layer_norm:
            self.layer_norm = nn.LayerNorm(config.fusion_hidden_size)
        
        # 输出投影
        self.output_proj = nn.Linear(config.fusion_hidden_size, config.fusion_hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        logger.info(f"MultimodalFusion initialized with strategy: {config.fusion_strategy.value}")
    
    def _align_sequences(
        self,
        text_features: torch.Tensor,
        code_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """对齐序列长度"""
        text_len = text_features.size(1)
        code_len = code_features.size(1)
        
        if text_len == code_len:
            return text_features, code_features
        
        # 使用较短的长度
        min_len = min(text_len, code_len)
        
        if text_len > min_len:
            # 截断或池化文本特征
            text_features = text_features[:, :min_len, :]
        
        if code_len > min_len:
            # 截断或池化代码特征
            code_features = code_features[:, :min_len, :]
        
        return text_features, code_features
    
    def forward(
        self,
        text_features: torch.Tensor,
        code_features: torch.Tensor,
        text_attention_mask: Optional[torch.Tensor] = None,
        code_attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        多模态融合前向传播
        
        Args:
            text_features: [batch_size, text_seq_len, text_hidden_size]
            code_features: [batch_size, code_seq_len, code_hidden_size]
            text_attention_mask: 文本注意力掩码
            code_attention_mask: 代码注意力掩码
            
        Returns:
            Dict[str, torch.Tensor]: 融合结果
        """
        # 输入投影
        text_proj = self.text_input_proj(text_features)
        code_proj = self.code_input_proj(code_features)
        
        # 序列对齐
        text_aligned, code_aligned = self._align_sequences(text_proj, code_proj)
        
        # 根据策略进行融合
        if self.fusion_strategy == FusionStrategy.CONCATENATION:
            # 拼接融合
            concat_features = torch.cat([text_aligned, code_aligned], dim=-1)
            fused_features = self.fusion_layer(concat_features)
            attention_weights = None
            
        elif self.fusion_strategy == FusionStrategy.ATTENTION:
            # 自注意力融合
            combined_features = text_aligned + code_aligned
            fused_features, attention_weights = self.fusion_layer(
                combined_features, combined_features, combined_features
            )
            
        elif self.fusion_strategy == FusionStrategy.CROSS_ATTENTION:
            # 跨模态注意力融合
            fused_features, attention_weights = self.fusion_layer(
                text_aligned, code_aligned
            )
            
        elif self.fusion_strategy == FusionStrategy.GATED_FUSION:
            # 门控融合
            fused_features = self.fusion_layer(text_aligned, code_aligned)
            attention_weights = None
            
        elif self.fusion_strategy == FusionStrategy.MULTIHEAD_FUSION:
            # 多头融合
            fused_features = self.fusion_layer(text_aligned, code_aligned)
            attention_weights = None
        
        # 残差连接
        if self.config.enable_residual:
            residual = (text_aligned + code_aligned) / 2
            fused_features = fused_features + residual
        
        # 层归一化
        if self.config.enable_layer_norm:
            fused_features = self.layer_norm(fused_features)
        
        # 多层融合
        if self.fusion_layers is not None:
            for layer in self.fusion_layers:
                fused_features = layer(fused_features)
        
        # Dropout
        fused_features = self.dropout(fused_features)
        
        # 输出投影
        output_features = self.output_proj(fused_features)
        
        # 池化输出
        pooled_output = output_features.mean(dim=1)  # 全局平均池化
        
        return {
            "fused_features": output_features,
            "pooled_output": pooled_output,
            "attention_weights": attention_weights,
            "text_features": text_aligned,
            "code_features": code_aligned
        }
    
    def get_fusion_statistics(
        self,
        text_features: torch.Tensor,
        code_features: torch.Tensor
    ) -> Dict[str, Any]:
        """获取融合统计信息"""
        with torch.no_grad():
            outputs = self.forward(text_features, code_features)
            
            stats = {
                "text_feature_norm": torch.norm(text_features, dim=-1).mean().item(),
                "code_feature_norm": torch.norm(code_features, dim=-1).mean().item(),
                "fused_feature_norm": torch.norm(outputs["fused_features"], dim=-1).mean().item(),
                "fusion_strategy": self.fusion_strategy.value
            }
            
            if outputs["attention_weights"] is not None:
                stats["attention_entropy"] = self._compute_attention_entropy(
                    outputs["attention_weights"]
                ).item()
            
            return stats
    
    def _compute_attention_entropy(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """计算注意力熵"""
        # 避免log(0)
        attention_weights = attention_weights + 1e-8
        entropy = -(attention_weights * torch.log(attention_weights)).sum(dim=-1)
        return entropy.mean()
