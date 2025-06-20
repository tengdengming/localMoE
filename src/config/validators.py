"""
配置验证器 - 验证配置的有效性和一致性
提供详细的验证规则和错误报告
"""

import re
import os
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
import logging

from .settings import Settings

logger = logging.getLogger(__name__)


@dataclass
class ValidationError:
    """验证错误"""
    field: str
    value: Any
    message: str
    severity: str = "error"  # error, warning, info


@dataclass
class ValidationResult:
    """验证结果"""
    is_valid: bool
    errors: List[ValidationError]
    warnings: List[ValidationError]
    
    def add_error(self, field: str, value: Any, message: str):
        """添加错误"""
        self.errors.append(ValidationError(field, value, message, "error"))
        self.is_valid = False
    
    def add_warning(self, field: str, value: Any, message: str):
        """添加警告"""
        self.warnings.append(ValidationError(field, value, message, "warning"))


class ConfigValidator:
    """
    配置验证器
    验证配置的有效性、一致性和安全性
    """
    
    def __init__(self):
        self.validation_rules = self._build_validation_rules()
    
    def _build_validation_rules(self) -> Dict[str, Any]:
        """构建验证规则"""
        return {
            "gpu": {
                "device_count": {
                    "type": int,
                    "min": 1,
                    "max": 8,
                    "required": True
                },
                "memory_limit_gb": {
                    "type": (int, float),
                    "min": 1.0,
                    "max": 80.0,
                    "required": True
                },
                "utilization_threshold": {
                    "type": (int, float),
                    "min": 0.1,
                    "max": 1.0,
                    "required": True
                },
                "temperature_threshold": {
                    "type": (int, float),
                    "min": 50.0,
                    "max": 100.0,
                    "required": True
                },
                "monitoring_interval": {
                    "type": (int, float),
                    "min": 0.1,
                    "max": 60.0,
                    "required": True
                }
            },
            "model": {
                "num_experts": {
                    "type": int,
                    "min": 1,
                    "max": 32,
                    "required": True
                },
                "top_k_experts": {
                    "type": int,
                    "min": 1,
                    "max": 8,
                    "required": True
                },
                "hidden_size": {
                    "type": int,
                    "min": 128,
                    "max": 8192,
                    "required": True
                },
                "max_sequence_length": {
                    "type": int,
                    "min": 128,
                    "max": 8192,
                    "required": True
                },
                "quantization_type": {
                    "type": str,
                    "choices": ["fp16", "fp8", "int8", "int4", "awq_3bit", "gptq_4bit", None],
                    "required": False
                }
            },
            "inference": {
                "preferred_engine": {
                    "type": str,
                    "choices": ["deepspeed", "vllm", "auto"],
                    "required": True
                },
                "max_concurrent_requests": {
                    "type": int,
                    "min": 1,
                    "max": 1000,
                    "required": True
                },
                "request_timeout": {
                    "type": (int, float),
                    "min": 1.0,
                    "max": 300.0,
                    "required": True
                },
                "max_batch_size": {
                    "type": int,
                    "min": 1,
                    "max": 128,
                    "required": True
                },
                "batch_timeout": {
                    "type": (int, float),
                    "min": 0.01,
                    "max": 10.0,
                    "required": True
                }
            },
            # DeepSpeed验证规则已注释掉
            # "deepspeed": {
            #     "zero_stage": {
            #         "type": int,
            #         "choices": [0, 1, 2, 3],
            #         "required": True
            #     },
            #     "expert_shard_size": {
            #         "type": int,
            #         "min": 1,
            #         "max": 16,
            #         "required": True
            #     },
            #     "quantization_bits": {
            #         "type": int,
            #         "choices": [4, 8, 16],
            #         "required": False
            #     },
            #     "tensor_parallel_size": {
            #         "type": int,
            #         "min": 1,
            #         "max": 8,
            #         "required": True
            #     }
            # },
            "vllm": {
                "tensor_parallel_size": {
                    "type": int,
                    "min": 1,
                    "max": 8,
                    "required": True
                },
                "gpu_memory_utilization": {
                    "type": (int, float),
                    "min": 0.1,
                    "max": 1.0,
                    "required": True
                },
                "max_model_len": {
                    "type": int,
                    "min": 128,
                    "max": 8192,
                    "required": True
                },
                "block_size": {
                    "type": int,
                    "min": 8,
                    "max": 64,
                    "required": True
                },
                "swap_space": {
                    "type": int,
                    "min": 0,
                    "max": 32,
                    "required": True
                }
            },
            "api": {
                "host": {
                    "type": str,
                    "pattern": r"^(\d{1,3}\.){3}\d{1,3}$|^localhost$|^0\.0\.0\.0$",
                    "required": True
                },
                "port": {
                    "type": int,
                    "min": 1024,
                    "max": 65535,
                    "required": True
                },
                "workers": {
                    "type": int,
                    "min": 1,
                    "max": 16,
                    "required": True
                },
                "log_level": {
                    "type": str,
                    "choices": ["debug", "info", "warning", "error", "critical"],
                    "required": True
                },
                "rate_limit_calls": {
                    "type": int,
                    "min": 1,
                    "max": 10000,
                    "required": True
                },
                "rate_limit_period": {
                    "type": int,
                    "min": 1,
                    "max": 3600,
                    "required": True
                }
            },
            "monitoring": {
                "metrics_interval": {
                    "type": (int, float),
                    "min": 1.0,
                    "max": 300.0,
                    "required": True
                },
                "prometheus_port": {
                    "type": int,
                    "min": 1024,
                    "max": 65535,
                    "required": False
                },
                "log_max_size": {
                    "type": str,
                    "pattern": r"^\d+[KMGT]?B$",
                    "required": False
                },
                "log_backup_count": {
                    "type": int,
                    "min": 1,
                    "max": 100,
                    "required": False
                }
            },
            "security": {
                "jwt_algorithm": {
                    "type": str,
                    "choices": ["HS256", "HS384", "HS512", "RS256", "RS384", "RS512"],
                    "required": False
                },
                "jwt_expiration_hours": {
                    "type": int,
                    "min": 1,
                    "max": 168,  # 1 week
                    "required": False
                },
                "max_request_size_mb": {
                    "type": int,
                    "min": 1,
                    "max": 100,
                    "required": True
                }
            }
        }
    
    def validate_settings(self, settings: Settings) -> ValidationResult:
        """
        验证设置对象
        
        Args:
            settings: 设置对象
            
        Returns:
            ValidationResult: 验证结果
        """
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        # 转换为字典进行验证
        settings_dict = settings.to_dict()
        
        # 验证各个部分
        for section_name, section_rules in self.validation_rules.items():
            if section_name in settings_dict:
                section_data = settings_dict[section_name]
                self._validate_section(section_name, section_data, section_rules, result)
        
        # 跨字段验证
        self._validate_cross_field_constraints(settings_dict, result)
        
        # 资源一致性验证
        self._validate_resource_consistency(settings_dict, result)
        
        # 安全性验证
        self._validate_security_settings(settings_dict, result)
        
        return result
    
    def _validate_section(
        self, 
        section_name: str, 
        section_data: Dict[str, Any], 
        section_rules: Dict[str, Any], 
        result: ValidationResult
    ):
        """验证配置段"""
        for field_name, field_rules in section_rules.items():
            field_path = f"{section_name}.{field_name}"
            
            if field_name not in section_data:
                if field_rules.get("required", False):
                    result.add_error(field_path, None, f"Required field missing")
                continue
            
            field_value = section_data[field_name]
            self._validate_field(field_path, field_value, field_rules, result)
    
    def _validate_field(
        self, 
        field_path: str, 
        field_value: Any, 
        field_rules: Dict[str, Any], 
        result: ValidationResult
    ):
        """验证单个字段"""
        # 类型验证
        expected_type = field_rules.get("type")
        if expected_type and not isinstance(field_value, expected_type):
            result.add_error(
                field_path, 
                field_value, 
                f"Expected type {expected_type}, got {type(field_value)}"
            )
            return
        
        # 数值范围验证
        if isinstance(field_value, (int, float)):
            min_val = field_rules.get("min")
            max_val = field_rules.get("max")
            
            if min_val is not None and field_value < min_val:
                result.add_error(
                    field_path, 
                    field_value, 
                    f"Value {field_value} is below minimum {min_val}"
                )
            
            if max_val is not None and field_value > max_val:
                result.add_error(
                    field_path, 
                    field_value, 
                    f"Value {field_value} is above maximum {max_val}"
                )
        
        # 选择值验证
        choices = field_rules.get("choices")
        if choices and field_value not in choices:
            result.add_error(
                field_path, 
                field_value, 
                f"Value {field_value} not in allowed choices: {choices}"
            )
        
        # 正则表达式验证
        pattern = field_rules.get("pattern")
        if pattern and isinstance(field_value, str):
            if not re.match(pattern, field_value):
                result.add_error(
                    field_path, 
                    field_value, 
                    f"Value does not match pattern: {pattern}"
                )
        
        # 路径验证
        if field_path.endswith("_dir") and isinstance(field_value, str):
            if not os.path.isabs(field_value):
                result.add_warning(
                    field_path, 
                    field_value, 
                    "Relative path may cause issues in different environments"
                )
    
    def _validate_cross_field_constraints(self, settings_dict: Dict[str, Any], result: ValidationResult):
        """验证跨字段约束"""
        # 验证top_k_experts不超过num_experts
        model_config = settings_dict.get("model", {})
        num_experts = model_config.get("num_experts", 8)
        top_k_experts = model_config.get("top_k_experts", 2)
        
        if top_k_experts > num_experts:
            result.add_error(
                "model.top_k_experts", 
                top_k_experts, 
                f"top_k_experts ({top_k_experts}) cannot exceed num_experts ({num_experts})"
            )
        
        # 验证tensor_parallel_size不超过GPU数量
        gpu_config = settings_dict.get("gpu", {})
        device_count = gpu_config.get("device_count", 4)
        
        deepspeed_config = settings_dict.get("deepspeed", {})
        ds_tensor_parallel = deepspeed_config.get("tensor_parallel_size", 4)
        
        vllm_config = settings_dict.get("vllm", {})
        vllm_tensor_parallel = vllm_config.get("tensor_parallel_size", 4)
        
        if ds_tensor_parallel > device_count:
            result.add_error(
                "deepspeed.tensor_parallel_size", 
                ds_tensor_parallel, 
                f"tensor_parallel_size ({ds_tensor_parallel}) cannot exceed device_count ({device_count})"
            )
        
        if vllm_tensor_parallel > device_count:
            result.add_error(
                "vllm.tensor_parallel_size", 
                vllm_tensor_parallel, 
                f"tensor_parallel_size ({vllm_tensor_parallel}) cannot exceed device_count ({device_count})"
            )
    
    def _validate_resource_consistency(self, settings_dict: Dict[str, Any], result: ValidationResult):
        """验证资源一致性"""
        gpu_config = settings_dict.get("gpu", {})
        model_config = settings_dict.get("model", {})
        inference_config = settings_dict.get("inference", {})
        
        device_count = gpu_config.get("device_count", 4)
        memory_limit = gpu_config.get("memory_limit_gb", 40.0)
        num_experts = model_config.get("num_experts", 8)
        max_batch_size = inference_config.get("max_batch_size", 32)
        
        # 估算内存需求
        estimated_memory_per_expert = 4.0  # GB
        total_memory_needed = num_experts * estimated_memory_per_expert
        total_available_memory = device_count * memory_limit
        
        if total_memory_needed > total_available_memory:
            result.add_warning(
                "resource.memory", 
                total_memory_needed, 
                f"Estimated memory requirement ({total_memory_needed:.1f}GB) may exceed "
                f"available memory ({total_available_memory:.1f}GB)"
            )
        
        # 检查批处理大小是否合理
        if max_batch_size > device_count * 8:
            result.add_warning(
                "inference.max_batch_size", 
                max_batch_size, 
                f"Large batch size ({max_batch_size}) may cause memory issues"
            )
    
    def _validate_security_settings(self, settings_dict: Dict[str, Any], result: ValidationResult):
        """验证安全设置"""
        security_config = settings_dict.get("security", {})
        api_config = settings_dict.get("api", {})
        
        # 检查生产环境的安全设置
        environment = settings_dict.get("environment", "development")
        if environment == "production":
            enable_auth = security_config.get("enable_auth", False)
            if not enable_auth:
                result.add_warning(
                    "security.enable_auth", 
                    enable_auth, 
                    "Authentication should be enabled in production"
                )
            
            jwt_secret = security_config.get("jwt_secret_key")
            if enable_auth and not jwt_secret:
                result.add_error(
                    "security.jwt_secret_key", 
                    jwt_secret, 
                    "JWT secret key is required when authentication is enabled"
                )
            
            # 检查API主机设置
            api_host = api_config.get("host", "0.0.0.0")
            if api_host == "0.0.0.0":
                result.add_warning(
                    "api.host", 
                    api_host, 
                    "Binding to 0.0.0.0 in production may be a security risk"
                )
    
    def validate_config_dict(self, config_dict: Dict[str, Any]) -> ValidationResult:
        """
        验证配置字典
        
        Args:
            config_dict: 配置字典
            
        Returns:
            ValidationResult: 验证结果
        """
        try:
            settings = Settings.from_dict(config_dict)
            return self.validate_settings(settings)
        except Exception as e:
            result = ValidationResult(is_valid=False, errors=[], warnings=[])
            result.add_error("config", config_dict, f"Failed to parse config: {str(e)}")
            return result
    
    def get_validation_schema(self) -> Dict[str, Any]:
        """获取验证模式"""
        return self.validation_rules.copy()
