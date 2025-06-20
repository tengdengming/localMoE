"""
L40S量化优化器
根据模型大小和硬件特点自动选择最佳量化策略
"""

import torch
import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import psutil

logger = logging.getLogger(__name__)


class ModelSize(Enum):
    """模型大小分类"""
    SMALL = "small"      # < 10B
    MEDIUM = "medium"    # 10B - 30B
    LARGE = "large"      # 30B - 70B
    XLARGE = "xlarge"    # > 70B


class UseCase(Enum):
    """使用场景"""
    CHAT = "chat"                    # 聊天对话
    CODE_GENERATION = "code"         # 代码生成
    BATCH_PROCESSING = "batch"       # 批处理
    REAL_TIME = "realtime"          # 实时交互
    HIGH_QUALITY = "quality"        # 高质量要求


@dataclass
class QuantizationRecommendation:
    """量化推荐结果"""
    quantization_type: Optional[str]
    kv_cache_dtype: Optional[str]
    gpu_memory_utilization: float
    tensor_parallel_size: int
    max_model_len: int
    max_num_batched_tokens: int
    reasoning: str
    expected_memory_gb: float
    expected_performance_ratio: float


class L40SQuantizationOptimizer:
    """L40S量化优化器"""
    
    def __init__(self):
        self.l40s_memory_gb = 48.0
        self.l40s_count = 4
        self.total_memory_gb = self.l40s_memory_gb * self.l40s_count
        
        # 量化方案性能特征
        self.quantization_specs = {
            None: {
                "memory_ratio": 1.0,
                "performance_ratio": 1.0,
                "quality_loss": 0.0,
                "l40s_optimized": False
            },
            "fp16": {
                "memory_ratio": 0.5,
                "performance_ratio": 1.0,
                "quality_loss": 0.005,
                "l40s_optimized": True
            },
            "fp8": {
                "memory_ratio": 0.25,
                "performance_ratio": 1.2,
                "quality_loss": 0.02,
                "l40s_optimized": True  # L40S原生支持
            },
            "awq": {
                "memory_ratio": 0.25,
                "performance_ratio": 1.1,
                "quality_loss": 0.03,
                "l40s_optimized": True
            },
            "gptq": {
                "memory_ratio": 0.25,
                "performance_ratio": 1.0,
                "quality_loss": 0.05,
                "l40s_optimized": True
            },
            "squeezellm": {
                "memory_ratio": 0.2,
                "performance_ratio": 0.9,
                "quality_loss": 0.08,
                "l40s_optimized": False
            }
        }
        
        # 模型大小估算 (参数数量 -> 内存需求GB)
        self.model_memory_estimates = {
            "7b": 14,
            "13b": 26,
            "30b": 60,
            "70b": 140,
            "175b": 350
        }
    
    def estimate_model_size(self, model_name: str) -> Tuple[ModelSize, float]:
        """估算模型大小"""
        model_name_lower = model_name.lower()
        
        # 从模型名称推断大小
        if "7b" in model_name_lower or "6b" in model_name_lower:
            return ModelSize.SMALL, self.model_memory_estimates["7b"]
        elif "13b" in model_name_lower or "12b" in model_name_lower:
            return ModelSize.MEDIUM, self.model_memory_estimates["13b"]
        elif "30b" in model_name_lower or "34b" in model_name_lower:
            return ModelSize.LARGE, self.model_memory_estimates["30b"]
        elif "70b" in model_name_lower or "65b" in model_name_lower:
            return ModelSize.LARGE, self.model_memory_estimates["70b"]
        elif "175b" in model_name_lower or "180b" in model_name_lower:
            return ModelSize.XLARGE, self.model_memory_estimates["175b"]
        else:
            # 默认估算为中等模型
            logger.warning(f"Cannot estimate size for model {model_name}, assuming medium")
            return ModelSize.MEDIUM, self.model_memory_estimates["13b"]
    
    def get_use_case_requirements(self, use_case: UseCase) -> Dict[str, Any]:
        """获取使用场景要求"""
        requirements = {
            UseCase.CHAT: {
                "max_quality_loss": 0.03,
                "min_performance_ratio": 1.0,
                "preferred_sequence_length": 4096,
                "batch_size_priority": "medium"
            },
            UseCase.CODE_GENERATION: {
                "max_quality_loss": 0.02,
                "min_performance_ratio": 1.0,
                "preferred_sequence_length": 8192,
                "batch_size_priority": "low"
            },
            UseCase.BATCH_PROCESSING: {
                "max_quality_loss": 0.05,
                "min_performance_ratio": 0.9,
                "preferred_sequence_length": 2048,
                "batch_size_priority": "high"
            },
            UseCase.REAL_TIME: {
                "max_quality_loss": 0.03,
                "min_performance_ratio": 1.1,
                "preferred_sequence_length": 2048,
                "batch_size_priority": "low"
            },
            UseCase.HIGH_QUALITY: {
                "max_quality_loss": 0.01,
                "min_performance_ratio": 0.9,
                "preferred_sequence_length": 4096,
                "batch_size_priority": "medium"
            }
        }
        return requirements[use_case]
    
    def recommend_quantization(
        self,
        model_name: str,
        use_case: UseCase = UseCase.CHAT,
        custom_requirements: Optional[Dict[str, Any]] = None
    ) -> QuantizationRecommendation:
        """推荐量化配置"""
        
        # 估算模型大小
        model_size, base_memory_gb = self.estimate_model_size(model_name)
        
        # 获取使用场景要求
        requirements = self.get_use_case_requirements(use_case)
        if custom_requirements:
            requirements.update(custom_requirements)
        
        # 评估量化方案
        best_quantization = None
        best_score = -1
        best_config = None
        
        for quant_type, specs in self.quantization_specs.items():
            # 计算内存需求
            memory_needed = base_memory_gb * specs["memory_ratio"]
            
            # 检查是否能装下
            if memory_needed > self.total_memory_gb * 0.9:  # 留10%余量
                continue
            
            # 检查质量要求
            if specs["quality_loss"] > requirements["max_quality_loss"]:
                continue
            
            # 检查性能要求
            if specs["performance_ratio"] < requirements["min_performance_ratio"]:
                continue
            
            # 计算评分
            score = self._calculate_score(specs, requirements, memory_needed)
            
            if score > best_score:
                best_score = score
                best_quantization = quant_type
                best_config = self._generate_config(
                    quant_type, specs, requirements, model_size, memory_needed
                )
        
        if best_config is None:
            # 如果没有找到合适的配置，使用最激进的量化
            logger.warning("No suitable quantization found, using aggressive quantization")
            best_quantization = "gptq"
            specs = self.quantization_specs["gptq"]
            memory_needed = base_memory_gb * specs["memory_ratio"]
            best_config = self._generate_config(
                "gptq", specs, requirements, model_size, memory_needed
            )
        
        return best_config
    
    def _calculate_score(
        self, 
        specs: Dict[str, Any], 
        requirements: Dict[str, Any],
        memory_needed: float
    ) -> float:
        """计算量化方案评分"""
        score = 0
        
        # 性能加分
        score += specs["performance_ratio"] * 30
        
        # 质量加分 (质量损失越小越好)
        score += (1 - specs["quality_loss"]) * 40
        
        # L40S优化加分
        if specs["l40s_optimized"]:
            score += 20
        
        # 内存利用率加分 (适中最好)
        memory_utilization = memory_needed / self.total_memory_gb
        if 0.6 <= memory_utilization <= 0.8:
            score += 10
        elif 0.4 <= memory_utilization <= 0.9:
            score += 5
        
        return score
    
    def _generate_config(
        self,
        quant_type: Optional[str],
        specs: Dict[str, Any],
        requirements: Dict[str, Any],
        model_size: ModelSize,
        memory_needed: float
    ) -> QuantizationRecommendation:
        """生成配置推荐"""
        
        # 确定并行策略
        if model_size in [ModelSize.SMALL]:
            tensor_parallel_size = 2
        else:
            tensor_parallel_size = 4
        
        # 确定内存使用率
        memory_per_gpu = memory_needed / tensor_parallel_size
        if memory_per_gpu < self.l40s_memory_gb * 0.7:
            gpu_memory_utilization = 0.9
        elif memory_per_gpu < self.l40s_memory_gb * 0.85:
            gpu_memory_utilization = 0.85
        else:
            gpu_memory_utilization = 0.8
        
        # 确定序列长度
        max_model_len = requirements["preferred_sequence_length"]
        
        # 确定批处理大小
        batch_priority = requirements["batch_size_priority"]
        if batch_priority == "high":
            max_num_batched_tokens = 32768
        elif batch_priority == "medium":
            max_num_batched_tokens = 16384
        else:
            max_num_batched_tokens = 8192
        
        # KV缓存类型
        kv_cache_dtype = None
        if quant_type == "fp8":
            kv_cache_dtype = "fp8_e5m2"
        
        # 生成推理说明
        reasoning = self._generate_reasoning(
            quant_type, model_size, specs, memory_needed, tensor_parallel_size
        )
        
        return QuantizationRecommendation(
            quantization_type=quant_type,
            kv_cache_dtype=kv_cache_dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            reasoning=reasoning,
            expected_memory_gb=memory_needed,
            expected_performance_ratio=specs["performance_ratio"]
        )
    
    def _generate_reasoning(
        self,
        quant_type: Optional[str],
        model_size: ModelSize,
        specs: Dict[str, Any],
        memory_needed: float,
        tensor_parallel_size: int
    ) -> str:
        """生成推理说明"""
        reasons = []
        
        if quant_type is None:
            reasons.append("使用FP16精度，追求最高质量")
        elif quant_type == "fp8":
            reasons.append("使用FP8量化，L40S原生支持，性能最佳")
        elif quant_type == "awq":
            reasons.append("使用AWQ量化，质量和性能平衡最佳")
        elif quant_type == "gptq":
            reasons.append("使用GPTQ量化，兼容性好")
        
        reasons.append(f"模型大小: {model_size.value}")
        reasons.append(f"预计内存需求: {memory_needed:.1f}GB")
        reasons.append(f"GPU并行度: {tensor_parallel_size}")
        reasons.append(f"预期性能比: {specs['performance_ratio']:.1f}x")
        
        if specs["l40s_optimized"]:
            reasons.append("针对L40S架构优化")
        
        return "; ".join(reasons)
    
    def validate_configuration(self, config: Dict[str, Any]) -> Tuple[bool, str]:
        """验证配置是否适合L40S"""
        issues = []
        
        # 检查GPU数量
        tensor_parallel_size = config.get("tensor_parallel_size", 1)
        if tensor_parallel_size > self.l40s_count:
            issues.append(f"张量并行度 {tensor_parallel_size} 超过GPU数量 {self.l40s_count}")
        
        # 检查内存使用率
        gpu_memory_utilization = config.get("gpu_memory_utilization", 0.9)
        if gpu_memory_utilization > 0.95:
            issues.append("GPU内存使用率过高，可能导致OOM")
        
        # 检查量化类型
        quantization = config.get("quantization")
        if quantization and quantization not in self.quantization_specs:
            issues.append(f"不支持的量化类型: {quantization}")
        
        if issues:
            return False, "; ".join(issues)
        else:
            return True, "配置验证通过"


# 使用示例
def get_optimized_config(model_name: str, use_case: str = "chat") -> Dict[str, Any]:
    """获取优化配置"""
    optimizer = L40SQuantizationOptimizer()
    use_case_enum = UseCase(use_case)
    
    recommendation = optimizer.recommend_quantization(model_name, use_case_enum)
    
    config = {
        "quantization": recommendation.quantization_type,
        "kv_cache_dtype": recommendation.kv_cache_dtype,
        "gpu_memory_utilization": recommendation.gpu_memory_utilization,
        "tensor_parallel_size": recommendation.tensor_parallel_size,
        "max_model_len": recommendation.max_model_len,
        "max_num_batched_tokens": recommendation.max_num_batched_tokens
    }
    
    logger.info(f"推荐配置: {recommendation.reasoning}")
    
    return config
