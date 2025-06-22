#!/usr/bin/env python3
"""
LocalMoE 增强模型下载器
支持 aria2c 高速下载、断点续传、进度显示
"""

import os
import sys
import json
import argparse
import subprocess
import requests
from pathlib import Path
from typing import List, Dict, Optional
import time
from urllib.parse import urljoin

class ModelDownloader:
    def __init__(self, target_dir: str = "/data/models", use_mirror: bool = False, 
                 mirror_url: str = "https://hf-mirror.com", max_connections: int = 16):
        self.target_dir = Path(target_dir)
        self.use_mirror = use_mirror
        self.mirror_url = mirror_url
        self.max_connections = max_connections
        self.base_url = mirror_url if use_mirror else "https://huggingface.co"
        
        # 创建目标目录
        self.target_dir.mkdir(parents=True, exist_ok=True)
        
    def log(self, level: str, message: str):
        """彩色日志输出"""
        colors = {
            'INFO': '\033[0;34m',
            'SUCCESS': '\033[0;32m', 
            'WARNING': '\033[1;33m',
            'ERROR': '\033[0;31m',
            'PROGRESS': '\033[0;36m',
            'DOWNLOAD': '\033[0;35m'
        }
        reset = '\033[0m'
        print(f"{colors.get(level, '')}[{level}]{reset} {message}")
        
    def check_dependencies(self) -> bool:
        """检查依赖工具"""
        self.log("INFO", "Checking dependencies...")
        
        # 检查 aria2c
        try:
            subprocess.run(["aria2c", "--version"], capture_output=True, check=True)
            self.log("SUCCESS", "aria2c is available")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.log("ERROR", "aria2c is not installed")
            self.log("INFO", "Install with: sudo apt-get install aria2")
            return False
            
    def get_model_files(self, model_name: str) -> List[Dict]:
        """获取模型文件列表"""
        self.log("INFO", f"Getting file list for {model_name}...")
        
        # HuggingFace API URL
        api_url = f"https://huggingface.co/api/models/{model_name}/tree/main"
        if self.use_mirror:
            # 对于镜像，我们需要不同的方法
            return self._get_files_from_mirror(model_name)
            
        try:
            response = requests.get(api_url, timeout=30)
            response.raise_for_status()
            files = response.json()
            
            # 过滤出需要下载的文件
            download_files = []
            for item in files:
                if item.get('type') == 'file':
                    file_path = item.get('path', '')
                    # 下载模型权重、配置文件等
                    if any(file_path.endswith(ext) for ext in 
                          ['.bin', '.safetensors', '.json', '.txt', '.py', '.md', '.model']):
                        download_files.append({
                            'path': file_path,
                            'size': item.get('size', 0),
                            'url': f"{self.base_url}/{model_name}/resolve/main/{file_path}"
                        })
                        
            self.log("SUCCESS", f"Found {len(download_files)} files to download")
            return download_files
            
        except Exception as e:
            self.log("ERROR", f"Failed to get file list: {e}")
            return []
            
    def _get_files_from_mirror(self, model_name: str) -> List[Dict]:
        """从镜像站获取文件列表（简化版）"""
        # 常见的模型文件
        common_files = [
            'config.json', 'tokenizer.json', 'tokenizer_config.json',
            'special_tokens_map.json', 'vocab.txt', 'merges.txt',
            'pytorch_model.bin', 'model.safetensors', 'README.md'
        ]
        
        files = []
        for filename in common_files:
            url = f"{self.base_url}/{model_name}/resolve/main/{filename}"
            # 检查文件是否存在
            try:
                response = requests.head(url, timeout=10)
                if response.status_code == 200:
                    size = int(response.headers.get('content-length', 0))
                    files.append({
                        'path': filename,
                        'size': size,
                        'url': url
                    })
            except:
                continue
                
        return files
        
    def download_file_aria2(self, file_info: Dict, output_dir: Path) -> bool:
        """使用 aria2c 下载单个文件"""
        file_path = file_info['path']
        file_url = file_info['url']
        file_size = file_info.get('size', 0)
        
        self.log("DOWNLOAD", f"Downloading {file_path} ({self._format_size(file_size)})")
        
        # aria2c 参数
        aria2_cmd = [
            "aria2c",
            "--continue=true",
            f"--max-connection-per-server={self.max_connections}",
            "--max-concurrent-downloads=1",
            "--timeout=300",
            "--retry-wait=3", 
            "--max-tries=5",
            "--split=16",
            "--min-split-size=1M",
            "--file-allocation=falloc",
            "--summary-interval=1",
            "--console-log-level=notice",
            "--show-console-readout=true",
            "--human-readable=true",
            f"--dir={output_dir}",
            f"--out={file_path}",
            file_url
        ]
        
        try:
            # 创建子目录
            file_output_path = output_dir / file_path
            file_output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 执行下载
            result = subprocess.run(aria2_cmd, capture_output=False)
            
            if result.returncode == 0:
                self.log("SUCCESS", f"Downloaded {file_path}")
                return True
            else:
                self.log("ERROR", f"Failed to download {file_path}")
                return False
                
        except Exception as e:
            self.log("ERROR", f"Error downloading {file_path}: {e}")
            return False
            
    def _format_size(self, size_bytes: int) -> str:
        """格式化文件大小"""
        if size_bytes == 0:
            return "Unknown"
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f}{unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f}TB"
        
    def download_model(self, model_name: str) -> bool:
        """下载完整模型"""
        self.log("INFO", f"Starting download of {model_name}")
        self.log("INFO", f"Target directory: {self.target_dir}")
        self.log("INFO", f"Using mirror: {self.use_mirror}")
        
        if not self.check_dependencies():
            return False
            
        # 获取文件列表
        files = self.get_model_files(model_name)
        if not files:
            self.log("ERROR", "No files found to download")
            return False
            
        # 创建模型目录
        model_dir = self.target_dir / model_name.split('/')[-1]
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # 下载所有文件
        success_count = 0
        total_files = len(files)
        
        self.log("PROGRESS", f"Downloading {total_files} files...")
        
        for i, file_info in enumerate(files, 1):
            self.log("PROGRESS", f"[{i}/{total_files}] Processing {file_info['path']}")
            
            if self.download_file_aria2(file_info, model_dir):
                success_count += 1
            else:
                self.log("WARNING", f"Failed to download {file_info['path']}")
                
        # 下载结果
        if success_count == total_files:
            self.log("SUCCESS", f"All {total_files} files downloaded successfully!")
            return True
        elif success_count > 0:
            self.log("WARNING", f"Downloaded {success_count}/{total_files} files")
            return True
        else:
            self.log("ERROR", "No files were downloaded successfully")
            return False

def main():
    parser = argparse.ArgumentParser(description="LocalMoE Enhanced Model Downloader with aria2c")
    parser.add_argument("-m", "--model", required=True, help="Model name (e.g., Qwen/Qwen1.5-MoE-A2.7B-Chat)")
    parser.add_argument("-d", "--dir", default="/data/models", help="Target directory (default: /data/models)")
    parser.add_argument("--use-mirror", action="store_true", help="Use HuggingFace mirror")
    parser.add_argument("--mirror-url", default="https://hf-mirror.com", help="Mirror URL")
    parser.add_argument("--max-connections", type=int, default=16, help="Max connections per server")
    
    args = parser.parse_args()
    
    downloader = ModelDownloader(
        target_dir=args.dir,
        use_mirror=args.use_mirror,
        mirror_url=args.mirror_url,
        max_connections=args.max_connections
    )
    
    success = downloader.download_model(args.model)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
