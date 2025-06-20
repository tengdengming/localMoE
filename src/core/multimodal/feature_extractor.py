"""
特征提取器 - 统一的多模态特征提取接口
整合文本处理器、代码处理器和多模态融合
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
import logging

from .text_processor import TextProcessor, TextConfig
from .code_processor import CodeProcessor, CodeConfig
from .multimodal_fusion import MultimodalFusion, FusionConfig, FusionStrategy

logger = logging.getLogger(__name__)


@dataclass
class FeatureExtractorConfig:
    """特征提取器配置"""
    # 文本配置
    text_model_name: str = "microsoft/codebert-base"
    text_max_length: int = 512
    text_hidden_size: int = 768
    
    # 代码配置
    code_max_length: int = 1024
    code_hidden_size: int = 768
    code_vocab_size: int = 50000
    
    # 融合配置
    fusion_hidden_size: int = 768
    fusion_strategy: FusionStrategy = FusionStrategy.CROSS_ATTENTION
    num_attention_heads: int = 12
    num_fusion_layers: int = 2
    
    # 通用配置
    dropout: float = 0.1
    enable_caching: bool = True
    cache_size: int = 1000


class FeatureCache:
    """特征缓存"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = {}
        self.access_order = []
    
    def get(self, key: str) -> Optional[torch.Tensor]:
        """获取缓存特征"""
        if key in self.cache:
            # 更新访问顺序
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None
    
    def put(self, key: str, value: torch.Tensor):
        """存储特征到缓存"""
        if key in self.cache:
            # 更新现有项
            self.access_order.remove(key)
        elif len(self.cache) >= self.max_size:
            # 移除最久未使用的项
            oldest_key = self.access_order.pop(0)
            del self.cache[oldest_key]
        
        self.cache[key] = value.detach().clone()
        self.access_order.append(key)
    
    def clear(self):
        """清空缓存"""
        self.cache.clear()
        self.access_order.clear()
    
    def size(self) -> int:
        """获取缓存大小"""
        return len(self.cache)


class FeatureExtractor(nn.Module):
    """
    统一的多模态特征提取器
    支持文本、代码的独立处理和融合处理
    """
    
    def __init__(self, config: FeatureExtractorConfig):
        super().__init__()
        self.config = config
        
        # 创建子模块配置
        text_config = TextConfig(
            model_name=config.text_model_name,
            max_length=config.text_max_length,
            hidden_size=config.text_hidden_size,
            dropout=config.dropout
        )
        
        code_config = CodeConfig(
            max_length=config.code_max_length,
            hidden_size=config.code_hidden_size,
            vocab_size=config.code_vocab_size,
            dropout=config.dropout
        )
        
        fusion_config = FusionConfig(
            text_hidden_size=config.text_hidden_size,
            code_hidden_size=config.code_hidden_size,
            fusion_hidden_size=config.fusion_hidden_size,
            fusion_strategy=config.fusion_strategy,
            num_attention_heads=config.num_attention_heads,
            num_fusion_layers=config.num_fusion_layers,
            dropout=config.dropout
        )
        
        # 初始化处理器
        self.text_processor = TextProcessor(text_config)
        self.code_processor = CodeProcessor(code_config)
        self.multimodal_fusion = MultimodalFusion(fusion_config)
        
        # 特征缓存
        if config.enable_caching:
            self.feature_cache = FeatureCache(config.cache_size)
        else:
            self.feature_cache = None
        
        # 统计信息
        self.stats = {
            "text_processed": 0,
            "code_processed": 0,
            "multimodal_processed": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        logger.info(f"FeatureExtractor initialized with fusion strategy: {config.fusion_strategy.value}")
    
    def _generate_cache_key(self, text: str = None, code: str = None) -> str:
        """生成缓存键"""
        import hashlib
        
        content = ""
        if text:
            content += f"text:{text}"
        if code:
            content += f"code:{code}"
        
        return hashlib.md5(content.encode()).hexdigest()
    
    def extract_text_features(
        self,
        texts: Union[str, List[str]],
        use_cache: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        提取文本特征
        
        Args:
            texts: 输入文本
            use_cache: 是否使用缓存
            
        Returns:
            Dict[str, torch.Tensor]: 文本特征
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # 检查缓存
        if use_cache and self.feature_cache:
            cache_key = self._generate_cache_key(text=str(texts))
            cached_features = self.feature_cache.get(cache_key)
            if cached_features is not None:
                self.stats["cache_hits"] += 1
                return {"pooler_output": cached_features}
            else:
                self.stats["cache_misses"] += 1
        
        # 提取特征
        with torch.no_grad():
            outputs = self.text_processor(texts=texts)
        
        # 更新统计
        self.stats["text_processed"] += len(texts)
        
        # 缓存特征
        if use_cache and self.feature_cache:
            self.feature_cache.put(cache_key, outputs["pooler_output"])
        
        return outputs
    
    def extract_code_features(
        self,
        code: Union[str, List[str]],
        use_cache: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        提取代码特征
        
        Args:
            code: 输入代码
            use_cache: 是否使用缓存
            
        Returns:
            Dict[str, torch.Tensor]: 代码特征
        """
        if isinstance(code, str):
            code = [code]
        
        # 检查缓存
        if use_cache and self.feature_cache:
            cache_key = self._generate_cache_key(code=str(code))
            cached_features = self.feature_cache.get(cache_key)
            if cached_features is not None:
                self.stats["cache_hits"] += 1
                return {"pooler_output": cached_features}
            else:
                self.stats["cache_misses"] += 1
        
        # 提取特征
        with torch.no_grad():
            outputs = self.code_processor(code=code)
        
        # 更新统计
        self.stats["code_processed"] += len(code)
        
        # 缓存特征
        if use_cache and self.feature_cache:
            self.feature_cache.put(cache_key, outputs["pooler_output"])
        
        return outputs
    
    def extract_multimodal_features(
        self,
        texts: Union[str, List[str]],
        code: Union[str, List[str]],
        use_cache: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        提取多模态融合特征
        
        Args:
            texts: 输入文本
            code: 输入代码
            use_cache: 是否使用缓存
            
        Returns:
            Dict[str, torch.Tensor]: 融合特征
        """
        if isinstance(texts, str):
            texts = [texts]
        if isinstance(code, str):
            code = [code]
        
        # 检查缓存
        if use_cache and self.feature_cache:
            cache_key = self._generate_cache_key(text=str(texts), code=str(code))
            cached_features = self.feature_cache.get(cache_key)
            if cached_features is not None:
                self.stats["cache_hits"] += 1
                return {"pooled_output": cached_features}
            else:
                self.stats["cache_misses"] += 1
        
        # 提取文本特征
        text_outputs = self.text_processor(texts=texts)
        text_features = text_outputs["last_hidden_state"]
        text_attention_mask = text_outputs.get("attention_mask")
        
        # 提取代码特征
        code_outputs = self.code_processor(code=code)
        code_features = code_outputs["last_hidden_state"]
        code_attention_mask = code_outputs.get("attention_mask")
        
        # 多模态融合
        with torch.no_grad():
            fusion_outputs = self.multimodal_fusion(
                text_features=text_features,
                code_features=code_features,
                text_attention_mask=text_attention_mask,
                code_attention_mask=code_attention_mask
            )
        
        # 更新统计
        self.stats["multimodal_processed"] += len(texts)
        
        # 缓存特征
        if use_cache and self.feature_cache:
            self.feature_cache.put(cache_key, fusion_outputs["pooled_output"])
        
        return fusion_outputs
    
    def compute_similarity(
        self,
        texts1: Union[str, List[str]],
        code1: Union[str, List[str]],
        texts2: Union[str, List[str]],
        code2: Union[str, List[str]],
        similarity_type: str = "cosine"
    ) -> torch.Tensor:
        """
        计算多模态相似度
        
        Args:
            texts1, code1: 第一组输入
            texts2, code2: 第二组输入
            similarity_type: 相似度类型
            
        Returns:
            torch.Tensor: 相似度分数
        """
        # 提取特征
        features1 = self.extract_multimodal_features(texts1, code1)
        features2 = self.extract_multimodal_features(texts2, code2)
        
        feat1 = features1["pooled_output"]
        feat2 = features2["pooled_output"]
        
        # 计算相似度
        if similarity_type == "cosine":
            similarity = torch.cosine_similarity(feat1, feat2, dim=-1)
        elif similarity_type == "euclidean":
            similarity = -torch.norm(feat1 - feat2, dim=-1)
        elif similarity_type == "dot":
            similarity = torch.sum(feat1 * feat2, dim=-1)
        else:
            raise ValueError(f"Unsupported similarity type: {similarity_type}")
        
        return similarity
    
    def batch_extract_features(
        self,
        texts: List[str],
        codes: List[str],
        batch_size: int = 32,
        feature_type: str = "multimodal"
    ) -> List[torch.Tensor]:
        """
        批量提取特征
        
        Args:
            texts: 文本列表
            codes: 代码列表
            batch_size: 批次大小
            feature_type: 特征类型 ("text", "code", "multimodal")
            
        Returns:
            List[torch.Tensor]: 特征列表
        """
        if len(texts) != len(codes):
            raise ValueError("texts and codes must have the same length")
        
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_codes = codes[i:i + batch_size]
            
            if feature_type == "text":
                batch_outputs = self.extract_text_features(batch_texts)
                batch_features = batch_outputs["pooler_output"]
            elif feature_type == "code":
                batch_outputs = self.extract_code_features(batch_codes)
                batch_features = batch_outputs["pooler_output"]
            elif feature_type == "multimodal":
                batch_outputs = self.extract_multimodal_features(batch_texts, batch_codes)
                batch_features = batch_outputs["pooled_output"]
            else:
                raise ValueError(f"Unsupported feature type: {feature_type}")
            
            results.append(batch_features)
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self.stats.copy()
        
        if self.feature_cache:
            stats["cache_size"] = self.feature_cache.size()
            stats["cache_hit_rate"] = (
                self.stats["cache_hits"] / 
                max(1, self.stats["cache_hits"] + self.stats["cache_misses"])
            )
        
        return stats
    
    def clear_cache(self):
        """清空缓存"""
        if self.feature_cache:
            self.feature_cache.clear()
            logger.info("Feature cache cleared")
    
    def save_pretrained(self, save_directory: str):
        """保存模型"""
        import os
        import json
        
        os.makedirs(save_directory, exist_ok=True)
        
        # 保存模型权重
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        
        # 保存配置
        config_dict = {
            "text_model_name": self.config.text_model_name,
            "text_max_length": self.config.text_max_length,
            "text_hidden_size": self.config.text_hidden_size,
            "code_max_length": self.config.code_max_length,
            "code_hidden_size": self.config.code_hidden_size,
            "fusion_hidden_size": self.config.fusion_hidden_size,
            "fusion_strategy": self.config.fusion_strategy.value,
            "num_attention_heads": self.config.num_attention_heads,
            "num_fusion_layers": self.config.num_fusion_layers
        }
        
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"FeatureExtractor saved to {save_directory}")
    
    @classmethod
    def from_pretrained(cls, model_path: str) -> "FeatureExtractor":
        """从预训练模型加载"""
        import json
        import os
        
        # 加载配置
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        
        # 转换fusion_strategy
        config_dict["fusion_strategy"] = FusionStrategy(config_dict["fusion_strategy"])
        
        config = FeatureExtractorConfig(**config_dict)
        
        # 创建模型
        model = cls(config)
        
        # 加载权重
        model_path_bin = os.path.join(model_path, "pytorch_model.bin")
        state_dict = torch.load(model_path_bin, map_location="cpu")
        model.load_state_dict(state_dict)
        
        logger.info(f"FeatureExtractor loaded from {model_path}")
        return model
