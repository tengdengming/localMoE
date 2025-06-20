"""
文本处理器 - 处理自然语言文本输入
支持多种tokenizer和预处理策略
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModel
import logging

logger = logging.getLogger(__name__)


@dataclass
class TextConfig:
    """文本处理配置"""
    model_name: str = "microsoft/codebert-base"
    max_length: int = 512
    padding: str = "max_length"
    truncation: bool = True
    return_attention_mask: bool = True
    hidden_size: int = 768
    num_attention_heads: int = 12
    num_hidden_layers: int = 12
    dropout: float = 0.1
    layer_norm_eps: float = 1e-12


class TextEmbedding(nn.Module):
    """文本嵌入层"""
    
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        
        # 词嵌入
        self.word_embeddings = nn.Embedding(
            config.vocab_size if hasattr(config, 'vocab_size') else 50265,
            config.hidden_size
        )
        
        # 位置嵌入
        self.position_embeddings = nn.Embedding(
            config.max_length,
            config.hidden_size
        )
        
        # 类型嵌入（用于区分不同类型的文本）
        self.token_type_embeddings = nn.Embedding(2, config.hidden_size)
        
        # 层归一化和dropout
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            input_ids: [batch_size, seq_len]
            token_type_ids: [batch_size, seq_len]
            position_ids: [batch_size, seq_len]
            
        Returns:
            torch.Tensor: [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # 词嵌入
        word_embeddings = self.word_embeddings(input_ids)
        
        # 位置嵌入
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.position_embeddings(position_ids)
        
        # 类型嵌入
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        # 组合嵌入
        embeddings = word_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings


class TextEncoder(nn.Module):
    """文本编码器"""
    
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        
        # Transformer层
        self.layers = nn.ModuleList([
            self._create_transformer_layer(config)
            for _ in range(config.num_hidden_layers)
        ])
        
        # 池化层
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_activation = nn.Tanh()
    
    def _create_transformer_layer(self, config: TextConfig) -> nn.Module:
        """创建Transformer层"""
        return nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.hidden_size * 4,
            dropout=config.dropout,
            activation="gelu",
            layer_norm_eps=config.layer_norm_eps,
            batch_first=True
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len]
            
        Returns:
            Dict[str, torch.Tensor]: 包含last_hidden_state和pooler_output
        """
        # 处理注意力掩码
        if attention_mask is not None:
            # 转换为Transformer期望的格式
            attention_mask = attention_mask.float()
            attention_mask = (1.0 - attention_mask) * -10000.0
        
        # 通过Transformer层
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                src_key_padding_mask=attention_mask
            )
        
        # 池化
        pooler_output = self.pooler(hidden_states[:, 0])  # 使用[CLS] token
        pooler_output = self.pooler_activation(pooler_output)
        
        return {
            "last_hidden_state": hidden_states,
            "pooler_output": pooler_output
        }


class TextProcessor(nn.Module):
    """
    文本处理器
    完整的文本处理流水线
    """
    
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        
        # 初始化tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            logger.warning(f"Failed to load tokenizer {config.model_name}: {e}")
            # 使用默认tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        
        # 更新配置中的vocab_size
        config.vocab_size = self.tokenizer.vocab_size
        
        # 文本嵌入层
        self.embeddings = TextEmbedding(config)
        
        # 文本编码器
        self.encoder = TextEncoder(config)
        
        # 特征投影层
        self.feature_projection = nn.Linear(config.hidden_size, config.hidden_size)
        
        logger.info(f"TextProcessor initialized with model: {config.model_name}")
    
    def tokenize(
        self,
        texts: Union[str, List[str]],
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        文本tokenization
        
        Args:
            texts: 输入文本
            **kwargs: tokenizer参数
            
        Returns:
            Dict[str, torch.Tensor]: tokenization结果
        """
        # 合并配置参数
        tokenize_kwargs = {
            "max_length": self.config.max_length,
            "padding": self.config.padding,
            "truncation": self.config.truncation,
            "return_tensors": "pt",
            "return_attention_mask": self.config.return_attention_mask,
            **kwargs
        }
        
        # 执行tokenization
        encoded = self.tokenizer(texts, **tokenize_kwargs)
        
        return encoded
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        texts: Optional[Union[str, List[str]]] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            input_ids: token IDs
            attention_mask: 注意力掩码
            token_type_ids: token类型IDs
            texts: 原始文本（如果提供，会先进行tokenization）
            
        Returns:
            Dict[str, torch.Tensor]: 处理结果
        """
        # 如果提供了原始文本，先进行tokenization
        if texts is not None:
            encoded = self.tokenize(texts, **kwargs)
            input_ids = encoded["input_ids"]
            attention_mask = encoded.get("attention_mask")
            token_type_ids = encoded.get("token_type_ids")
        
        if input_ids is None:
            raise ValueError("Either input_ids or texts must be provided")
        
        # 嵌入
        embeddings = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids
        )
        
        # 编码
        encoder_outputs = self.encoder(
            hidden_states=embeddings,
            attention_mask=attention_mask
        )
        
        # 特征投影
        last_hidden_state = encoder_outputs["last_hidden_state"]
        projected_features = self.feature_projection(last_hidden_state)
        
        return {
            "last_hidden_state": last_hidden_state,
            "pooler_output": encoder_outputs["pooler_output"],
            "projected_features": projected_features,
            "attention_mask": attention_mask,
            "input_ids": input_ids
        }
    
    def extract_features(
        self,
        texts: Union[str, List[str]],
        layer_index: int = -1
    ) -> torch.Tensor:
        """
        提取文本特征
        
        Args:
            texts: 输入文本
            layer_index: 提取特征的层索引
            
        Returns:
            torch.Tensor: 文本特征
        """
        with torch.no_grad():
            outputs = self.forward(texts=texts)
            
            if layer_index == -1:
                # 使用最后一层的pooler输出
                return outputs["pooler_output"]
            else:
                # 使用指定层的hidden state
                return outputs["last_hidden_state"].mean(dim=1)  # 平均池化
    
    def get_text_embeddings(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """获取文本嵌入向量"""
        return self.extract_features(texts, layer_index=-1)
    
    def batch_process(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> List[torch.Tensor]:
        """
        批量处理文本
        
        Args:
            texts: 文本列表
            batch_size: 批次大小
            
        Returns:
            List[torch.Tensor]: 处理结果列表
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_outputs = self.forward(texts=batch_texts)
            results.append(batch_outputs["projected_features"])
        
        return results
    
    def save_pretrained(self, save_directory: str):
        """保存模型"""
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        # 保存模型权重
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        
        # 保存配置
        import json
        config_dict = {
            "model_name": self.config.model_name,
            "max_length": self.config.max_length,
            "hidden_size": self.config.hidden_size,
            "num_attention_heads": self.config.num_attention_heads,
            "num_hidden_layers": self.config.num_hidden_layers,
            "vocab_size": self.config.vocab_size
        }
        
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"TextProcessor saved to {save_directory}")
    
    @classmethod
    def from_pretrained(cls, model_path: str) -> "TextProcessor":
        """从预训练模型加载"""
        import json
        import os
        
        # 加载配置
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        
        config = TextConfig(**config_dict)
        
        # 创建模型
        model = cls(config)
        
        # 加载权重
        model_path_bin = os.path.join(model_path, "pytorch_model.bin")
        state_dict = torch.load(model_path_bin, map_location="cpu")
        model.load_state_dict(state_dict)
        
        logger.info(f"TextProcessor loaded from {model_path}")
        return model
