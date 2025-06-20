"""
系统设置和配置
定义所有配置项和默认值
"""

import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
import yaml
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class GPUSettings:
    """GPU配置设置"""
    device_count: int = 4
    memory_limit_gb: float = 40.0
    utilization_threshold: float = 0.85
    temperature_threshold: float = 85.0
    enable_monitoring: bool = True
    monitoring_interval: float = 1.0


@dataclass
class ModelSettings:
    """模型配置设置"""
    num_experts: int = 8
    top_k_experts: int = 2
    hidden_size: int = 768
    intermediate_size: int = 3072
    max_sequence_length: int = 2048
    quantization_type: Optional[str] = "fp16"
    enable_compilation: bool = True


@dataclass
class InferenceSettings:
    """推理配置设置"""
    preferred_engine: str = "auto"
    enable_fallback: bool = True
    enable_load_balancing: bool = True
    max_concurrent_requests: int = 100
    request_timeout: float = 30.0
    enable_batching: bool = True
    max_batch_size: int = 32
    batch_timeout: float = 0.1


# DeepSpeed配置已注释掉
# @dataclass
# class DeepSpeedSettings:
#     """DeepSpeed配置设置"""
#     zero_stage: int = 3
#     enable_expert_sharding: bool = True
#     expert_shard_size: int = 4
#     cpu_offload: bool = True
#     nvme_offload: bool = False
#     nvme_offload_dir: str = "/tmp/deepspeed_nvme"
#     enable_quantization: bool = True
#     quantization_bits: int = 8
#     tensor_parallel_size: int = 4


@dataclass
class VLLMSettings:
    """vLLM配置设置"""
    model_name: str = "microsoft/DialoGPT-medium"
    tensor_parallel_size: int = 4
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 2048
    block_size: int = 16
    swap_space: int = 4
    max_num_batched_tokens: int = 8192
    enable_prefix_caching: bool = True
    quantization: Optional[str] = None


@dataclass
class APISettings:
    """API配置设置"""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False
    log_level: str = "info"
    enable_cors: bool = True
    enable_gzip: bool = True
    rate_limit_calls: int = 100
    rate_limit_period: int = 60


@dataclass
class MonitoringSettings:
    """监控配置设置"""
    enable_metrics: bool = True
    metrics_interval: float = 10.0
    enable_prometheus: bool = False
    prometheus_port: int = 9090
    enable_logging: bool = True
    log_file: Optional[str] = None
    log_rotation: bool = True
    log_max_size: str = "100MB"
    log_backup_count: int = 5


@dataclass
class SecuritySettings:
    """安全配置设置"""
    enable_auth: bool = False
    jwt_secret_key: Optional[str] = None
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    enable_rate_limiting: bool = True
    enable_request_validation: bool = True
    max_request_size_mb: int = 10


@dataclass
class Settings:
    """主配置类"""
    gpu: GPUSettings = field(default_factory=GPUSettings)
    model: ModelSettings = field(default_factory=ModelSettings)
    inference: InferenceSettings = field(default_factory=InferenceSettings)
    # deepspeed: DeepSpeedSettings = field(default_factory=DeepSpeedSettings)  # DeepSpeed已注释掉
    vllm: VLLMSettings = field(default_factory=VLLMSettings)
    api: APISettings = field(default_factory=APISettings)
    monitoring: MonitoringSettings = field(default_factory=MonitoringSettings)
    security: SecuritySettings = field(default_factory=SecuritySettings)
    
    # 环境配置
    environment: str = "development"
    debug: bool = False
    
    # 路径配置
    model_cache_dir: str = "./models"
    log_dir: str = "./logs"
    config_dir: str = "./configs"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {}
        for key, value in self.__dict__.items():
            if hasattr(value, '__dict__'):
                result[key] = value.__dict__
            else:
                result[key] = value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Settings":
        """从字典创建"""
        settings = cls()
        
        for key, value in data.items():
            if hasattr(settings, key):
                attr = getattr(settings, key)
                if hasattr(attr, '__dict__') and isinstance(value, dict):
                    # 更新嵌套对象
                    for sub_key, sub_value in value.items():
                        if hasattr(attr, sub_key):
                            setattr(attr, sub_key, sub_value)
                else:
                    setattr(settings, key, value)
        
        return settings


def load_settings(config_file: Optional[str] = None) -> Settings:
    """
    加载配置设置
    
    Args:
        config_file: 配置文件路径
        
    Returns:
        Settings: 配置对象
    """
    settings = Settings()
    
    # 从环境变量加载
    settings = _load_from_env(settings)
    
    # 从配置文件加载
    if config_file:
        settings = _load_from_file(settings, config_file)
    else:
        # 尝试默认配置文件
        default_files = [
            "config.yaml",
            "config.yml", 
            "config.json",
            "configs/config.yaml",
            "configs/config.yml",
            "configs/config.json"
        ]
        
        for file_path in default_files:
            if os.path.exists(file_path):
                settings = _load_from_file(settings, file_path)
                break
    
    # 创建必要的目录
    _ensure_directories(settings)
    
    logger.info(f"Settings loaded (environment: {settings.environment})")
    return settings


def _load_from_env(settings: Settings) -> Settings:
    """从环境变量加载配置"""
    env_mappings = {
        # API设置
        "LOCALMOE_HOST": ("api", "host"),
        "LOCALMOE_PORT": ("api", "port", int),
        "LOCALMOE_WORKERS": ("api", "workers", int),
        "LOCALMOE_LOG_LEVEL": ("api", "log_level"),
        
        # GPU设置
        "LOCALMOE_GPU_COUNT": ("gpu", "device_count", int),
        "LOCALMOE_GPU_MEMORY_LIMIT": ("gpu", "memory_limit_gb", float),
        
        # 模型设置
        "LOCALMOE_NUM_EXPERTS": ("model", "num_experts", int),
        "LOCALMOE_TOP_K_EXPERTS": ("model", "top_k_experts", int),
        
        # 推理设置
        "LOCALMOE_PREFERRED_ENGINE": ("inference", "preferred_engine"),
        "LOCALMOE_MAX_CONCURRENT": ("inference", "max_concurrent_requests", int),
        
        # 环境设置
        "LOCALMOE_ENVIRONMENT": ("environment",),
        "LOCALMOE_DEBUG": ("debug", bool),
        "LOCALMOE_MODEL_CACHE_DIR": ("model_cache_dir",),
        "LOCALMOE_LOG_DIR": ("log_dir",),
    }
    
    for env_var, config_path in env_mappings.items():
        value = os.getenv(env_var)
        if value is not None:
            try:
                # 类型转换
                if len(config_path) > 2:
                    converter = config_path[2]
                    if converter == bool:
                        value = value.lower() in ("true", "1", "yes", "on")
                    else:
                        value = converter(value)
                
                # 设置值
                if len(config_path) == 1:
                    setattr(settings, config_path[0], value)
                else:
                    obj = getattr(settings, config_path[0])
                    setattr(obj, config_path[1], value)
                    
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid environment variable {env_var}={value}: {e}")
    
    return settings


def _load_from_file(settings: Settings, config_file: str) -> Settings:
    """从配置文件加载"""
    try:
        file_path = Path(config_file)
        if not file_path.exists():
            logger.warning(f"Config file not found: {config_file}")
            return settings
        
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.suffix.lower() in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
            elif file_path.suffix.lower() == '.json':
                data = json.load(f)
            else:
                logger.warning(f"Unsupported config file format: {config_file}")
                return settings
        
        if data:
            settings = Settings.from_dict(data)
            logger.info(f"Loaded config from: {config_file}")
        
    except Exception as e:
        logger.error(f"Failed to load config file {config_file}: {e}")
    
    return settings


def _ensure_directories(settings: Settings):
    """确保必要的目录存在"""
    directories = [
        settings.model_cache_dir,
        settings.log_dir,
        settings.config_dir
    ]
    
    for directory in directories:
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"Failed to create directory {directory}: {e}")


def save_settings(settings: Settings, config_file: str):
    """
    保存配置到文件
    
    Args:
        settings: 配置对象
        config_file: 配置文件路径
    """
    try:
        file_path = Path(config_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = settings.to_dict()
        
        with open(file_path, 'w', encoding='utf-8') as f:
            if file_path.suffix.lower() in ['.yaml', '.yml']:
                yaml.dump(data, f, default_flow_style=False, indent=2)
            elif file_path.suffix.lower() == '.json':
                json.dump(data, f, indent=2, ensure_ascii=False)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        logger.info(f"Settings saved to: {config_file}")
        
    except Exception as e:
        logger.error(f"Failed to save settings to {config_file}: {e}")
        raise


# 全局设置实例
_global_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """获取全局设置实例"""
    global _global_settings
    if _global_settings is None:
        _global_settings = load_settings()
    return _global_settings


def update_settings(new_settings: Settings):
    """更新全局设置"""
    global _global_settings
    _global_settings = new_settings
