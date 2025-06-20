"""
配置管理器 - 动态配置管理和热更新
支持配置验证、版本控制和回滚
"""

import os
import json
import yaml
import threading
import time
import logging
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import hashlib

from .settings import Settings, save_settings
from .validators import ConfigValidator

logger = logging.getLogger(__name__)


@dataclass
class ConfigSchema:
    """配置模式定义"""
    name: str
    version: str
    schema: Dict[str, Any]
    required_fields: List[str]
    optional_fields: List[str]
    validation_rules: Dict[str, Any]


@dataclass
class ConfigVersion:
    """配置版本信息"""
    version_id: str
    timestamp: datetime
    config_data: Dict[str, Any]
    checksum: str
    description: Optional[str] = None
    author: Optional[str] = None


class ConfigManager:
    """
    配置管理器
    提供配置的动态管理、验证和热更新功能
    """
    
    def __init__(
        self,
        config_file: Optional[str] = None,
        enable_hot_reload: bool = True,
        enable_versioning: bool = True,
        max_versions: int = 10
    ):
        self.config_file = config_file or "config.yaml"
        self.enable_hot_reload = enable_hot_reload
        self.enable_versioning = enable_versioning
        self.max_versions = max_versions
        
        # 当前配置
        self.current_config: Optional[Settings] = None
        self.config_lock = threading.RLock()
        
        # 配置版本历史
        self.config_versions: List[ConfigVersion] = []
        self.version_file = "config_versions.json"
        
        # 配置变更回调
        self.change_callbacks: List[Callable[[str, Any, Any], None]] = []
        
        # 配置验证器
        self.validator = ConfigValidator()
        
        # 文件监控
        self.file_watcher_thread = None
        self.is_watching = False
        self.last_modified = 0
        
        # 初始化
        self._load_initial_config()
        self._load_version_history()
        
        if self.enable_hot_reload:
            self._start_file_watcher()
        
        logger.info(f"ConfigManager initialized (file: {self.config_file})")
    
    def _load_initial_config(self):
        """加载初始配置"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    if self.config_file.endswith(('.yaml', '.yml')):
                        data = yaml.safe_load(f)
                    else:
                        data = json.load(f)
                
                self.current_config = Settings.from_dict(data or {})
                self.last_modified = os.path.getmtime(self.config_file)
            else:
                self.current_config = Settings()
                self._save_config()
            
            # 创建初始版本
            if self.enable_versioning:
                self._create_version("Initial configuration", "system")
                
        except Exception as e:
            logger.error(f"Failed to load initial config: {e}")
            self.current_config = Settings()
    
    def _load_version_history(self):
        """加载版本历史"""
        if not self.enable_versioning:
            return
        
        try:
            if os.path.exists(self.version_file):
                with open(self.version_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self.config_versions = []
                for version_data in data:
                    version = ConfigVersion(
                        version_id=version_data["version_id"],
                        timestamp=datetime.fromisoformat(version_data["timestamp"]),
                        config_data=version_data["config_data"],
                        checksum=version_data["checksum"],
                        description=version_data.get("description"),
                        author=version_data.get("author")
                    )
                    self.config_versions.append(version)
                
                logger.info(f"Loaded {len(self.config_versions)} config versions")
                
        except Exception as e:
            logger.error(f"Failed to load version history: {e}")
            self.config_versions = []
    
    def _save_version_history(self):
        """保存版本历史"""
        if not self.enable_versioning:
            return
        
        try:
            data = []
            for version in self.config_versions:
                data.append({
                    "version_id": version.version_id,
                    "timestamp": version.timestamp.isoformat(),
                    "config_data": version.config_data,
                    "checksum": version.checksum,
                    "description": version.description,
                    "author": version.author
                })
            
            with open(self.version_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Failed to save version history: {e}")
    
    def _create_version(self, description: str = None, author: str = None) -> str:
        """创建配置版本"""
        if not self.enable_versioning or not self.current_config:
            return ""
        
        config_data = self.current_config.to_dict()
        checksum = self._calculate_checksum(config_data)
        
        # 检查是否有变化
        if self.config_versions and self.config_versions[-1].checksum == checksum:
            return self.config_versions[-1].version_id
        
        version_id = f"v{len(self.config_versions) + 1}_{int(time.time())}"
        
        version = ConfigVersion(
            version_id=version_id,
            timestamp=datetime.now(),
            config_data=config_data,
            checksum=checksum,
            description=description,
            author=author
        )
        
        self.config_versions.append(version)
        
        # 限制版本数量
        if len(self.config_versions) > self.max_versions:
            self.config_versions = self.config_versions[-self.max_versions:]
        
        self._save_version_history()
        
        logger.info(f"Created config version: {version_id}")
        return version_id
    
    def _calculate_checksum(self, data: Dict[str, Any]) -> str:
        """计算配置数据的校验和"""
        json_str = json.dumps(data, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(json_str.encode()).hexdigest()
    
    def _save_config(self):
        """保存当前配置到文件"""
        if self.current_config:
            save_settings(self.current_config, self.config_file)
            self.last_modified = os.path.getmtime(self.config_file)
    
    def _start_file_watcher(self):
        """启动文件监控"""
        if self.is_watching:
            return
        
        self.is_watching = True
        self.file_watcher_thread = threading.Thread(
            target=self._file_watcher_loop, daemon=True
        )
        self.file_watcher_thread.start()
        
        logger.info("Started config file watcher")
    
    def _stop_file_watcher(self):
        """停止文件监控"""
        self.is_watching = False
        if self.file_watcher_thread:
            self.file_watcher_thread.join(timeout=5.0)
    
    def _file_watcher_loop(self):
        """文件监控循环"""
        while self.is_watching:
            try:
                if os.path.exists(self.config_file):
                    current_modified = os.path.getmtime(self.config_file)
                    if current_modified > self.last_modified:
                        logger.info("Config file changed, reloading...")
                        self._reload_config()
                        self.last_modified = current_modified
                
                time.sleep(1.0)  # 检查间隔
                
            except Exception as e:
                logger.error(f"File watcher error: {e}")
                time.sleep(5.0)
    
    def _reload_config(self):
        """重新加载配置"""
        try:
            with self.config_lock:
                old_config = self.current_config.to_dict() if self.current_config else {}
                
                # 加载新配置
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    if self.config_file.endswith(('.yaml', '.yml')):
                        data = yaml.safe_load(f)
                    else:
                        data = json.load(f)
                
                new_config = Settings.from_dict(data or {})
                
                # 验证配置
                validation_result = self.validator.validate_settings(new_config)
                if not validation_result.is_valid:
                    logger.error(f"Config validation failed: {validation_result.errors}")
                    return
                
                # 应用新配置
                self.current_config = new_config
                
                # 创建版本
                if self.enable_versioning:
                    self._create_version("Hot reload", "file_watcher")
                
                # 通知变更
                new_config_dict = new_config.to_dict()
                self._notify_config_change("config_reload", old_config, new_config_dict)
                
                logger.info("Config reloaded successfully")
                
        except Exception as e:
            logger.error(f"Failed to reload config: {e}")
    
    def get_config(self) -> Settings:
        """获取当前配置"""
        with self.config_lock:
            return self.current_config
    
    def update_config(
        self, 
        updates: Dict[str, Any], 
        description: str = None, 
        author: str = None,
        validate: bool = True
    ) -> bool:
        """
        更新配置
        
        Args:
            updates: 更新的配置项
            description: 更新描述
            author: 更新作者
            validate: 是否验证配置
            
        Returns:
            bool: 是否成功更新
        """
        try:
            with self.config_lock:
                if not self.current_config:
                    return False
                
                old_config = self.current_config.to_dict()
                
                # 应用更新
                new_config_data = old_config.copy()
                self._deep_update(new_config_data, updates)
                
                # 创建新配置对象
                new_config = Settings.from_dict(new_config_data)
                
                # 验证配置
                if validate:
                    validation_result = self.validator.validate_settings(new_config)
                    if not validation_result.is_valid:
                        logger.error(f"Config validation failed: {validation_result.errors}")
                        return False
                
                # 应用配置
                self.current_config = new_config
                
                # 保存到文件
                self._save_config()
                
                # 创建版本
                if self.enable_versioning:
                    self._create_version(description or "Manual update", author)
                
                # 通知变更
                self._notify_config_change("config_update", old_config, new_config_data)
                
                logger.info(f"Config updated successfully: {list(updates.keys())}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to update config: {e}")
            return False
    
    def _deep_update(self, target: Dict[str, Any], source: Dict[str, Any]):
        """深度更新字典"""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_update(target[key], value)
            else:
                target[key] = value
    
    def rollback_to_version(self, version_id: str) -> bool:
        """
        回滚到指定版本
        
        Args:
            version_id: 版本ID
            
        Returns:
            bool: 是否成功回滚
        """
        try:
            with self.config_lock:
                # 查找版本
                target_version = None
                for version in self.config_versions:
                    if version.version_id == version_id:
                        target_version = version
                        break
                
                if not target_version:
                    logger.error(f"Version {version_id} not found")
                    return False
                
                old_config = self.current_config.to_dict() if self.current_config else {}
                
                # 恢复配置
                self.current_config = Settings.from_dict(target_version.config_data)
                
                # 保存到文件
                self._save_config()
                
                # 创建回滚版本
                if self.enable_versioning:
                    self._create_version(f"Rollback to {version_id}", "system")
                
                # 通知变更
                self._notify_config_change("config_rollback", old_config, target_version.config_data)
                
                logger.info(f"Rolled back to version: {version_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to rollback to version {version_id}: {e}")
            return False
    
    def get_version_history(self) -> List[ConfigVersion]:
        """获取版本历史"""
        return self.config_versions.copy()
    
    def add_change_callback(self, callback: Callable[[str, Any, Any], None]):
        """添加配置变更回调"""
        self.change_callbacks.append(callback)
    
    def remove_change_callback(self, callback: Callable[[str, Any, Any], None]):
        """移除配置变更回调"""
        if callback in self.change_callbacks:
            self.change_callbacks.remove(callback)
    
    def _notify_config_change(self, change_type: str, old_config: Any, new_config: Any):
        """通知配置变更"""
        for callback in self.change_callbacks:
            try:
                callback(change_type, old_config, new_config)
            except Exception as e:
                logger.error(f"Config change callback error: {e}")
    
    def export_config(self, file_path: str, format: str = "yaml") -> bool:
        """
        导出配置
        
        Args:
            file_path: 导出文件路径
            format: 导出格式 (yaml/json)
            
        Returns:
            bool: 是否成功导出
        """
        try:
            with self.config_lock:
                if not self.current_config:
                    return False
                
                data = self.current_config.to_dict()
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    if format.lower() == 'yaml':
                        yaml.dump(data, f, default_flow_style=False, indent=2)
                    else:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Config exported to: {file_path}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to export config: {e}")
            return False
    
    def import_config(self, file_path: str, validate: bool = True) -> bool:
        """
        导入配置
        
        Args:
            file_path: 导入文件路径
            validate: 是否验证配置
            
        Returns:
            bool: 是否成功导入
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.endswith(('.yaml', '.yml')):
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)
            
            return self.update_config(
                data, 
                description=f"Import from {file_path}", 
                author="import",
                validate=validate
            )
            
        except Exception as e:
            logger.error(f"Failed to import config from {file_path}: {e}")
            return False
    
    def get_config_diff(self, version_id1: str, version_id2: str) -> Dict[str, Any]:
        """
        获取两个版本之间的差异
        
        Args:
            version_id1: 版本1 ID
            version_id2: 版本2 ID
            
        Returns:
            Dict[str, Any]: 差异信息
        """
        try:
            version1 = None
            version2 = None
            
            for version in self.config_versions:
                if version.version_id == version_id1:
                    version1 = version
                if version.version_id == version_id2:
                    version2 = version
            
            if not version1 or not version2:
                return {"error": "Version not found"}
            
            # 简化的差异计算
            diff = {
                "version1": version_id1,
                "version2": version_id2,
                "changes": []
            }
            
            # 这里可以实现更复杂的差异算法
            if version1.checksum != version2.checksum:
                diff["changes"].append("Configuration changed")
            
            return diff
            
        except Exception as e:
            logger.error(f"Failed to get config diff: {e}")
            return {"error": str(e)}
    
    def cleanup(self):
        """清理资源"""
        self._stop_file_watcher()
        self.change_callbacks.clear()
        
        logger.info("ConfigManager cleaned up")
