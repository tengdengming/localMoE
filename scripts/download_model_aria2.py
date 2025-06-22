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

        # 对于镜像，使用专门的方法
        if self.use_mirror:
            return self._get_files_from_mirror(model_name)

        # HuggingFace API URL
        api_url = f"https://huggingface.co/api/models/{model_name}/tree/main"

        try:
            response = requests.get(api_url, timeout=30)
            response.raise_for_status()
            files = response.json()

            # 过滤出需要下载的文件
            download_files = []
            for item in files:
                if item.get('type') == 'file':
                    file_path = item.get('path', '')
                    # 下载模型权重、配置文件等，包括分片文件
                    if any(file_path.endswith(ext) for ext in
                          ['.bin', '.safetensors', '.json', '.txt', '.py', '.md', '.model']) or \
                       'model-' in file_path or 'pytorch_model' in file_path:
                        download_files.append({
                            'path': file_path,
                            'size': item.get('size', 0),
                            'url': f"{self.base_url}/{model_name}/resolve/main/{file_path}"
                        })

            self.log("SUCCESS", f"Found {len(download_files)} files to download")

            # 显示文件大小统计
            total_size = sum(f.get('size', 0) for f in download_files)
            self.log("INFO", f"Total download size: {self._format_size(total_size)}")

            return download_files

        except Exception as e:
            self.log("ERROR", f"Failed to get file list: {e}")
            self.log("INFO", "Falling back to mirror method...")
            return self._get_files_from_mirror(model_name)
            
    def _get_files_from_mirror(self, model_name: str) -> List[Dict]:
        """从镜像站获取文件列表（增强版）"""
        self.log("INFO", "Getting file list from mirror...")

        files = []

        # 首先尝试获取 index.json 来了解模型结构
        index_url = f"{self.base_url}/{model_name}/resolve/main/model.safetensors.index.json"
        try:
            response = requests.get(index_url, timeout=30)
            if response.status_code == 200:
                index_data = response.json()
                weight_map = index_data.get('weight_map', {})
                # 获取所有权重文件
                weight_files = set(weight_map.values())
                self.log("SUCCESS", f"Found {len(weight_files)} weight files from index")

                for weight_file in weight_files:
                    url = f"{self.base_url}/{model_name}/resolve/main/{weight_file}"
                    try:
                        head_resp = requests.head(url, timeout=10)
                        if head_resp.status_code == 200:
                            size = int(head_resp.headers.get('content-length', 0))
                            files.append({
                                'path': weight_file,
                                'size': size,
                                'url': url
                            })
                            self.log("INFO", f"Added weight file: {weight_file} ({self._format_size(size)})")
                    except Exception as e:
                        self.log("WARNING", f"Failed to check {weight_file}: {e}")
                        continue

                # 添加 index 文件本身
                files.append({
                    'path': 'model.safetensors.index.json',
                    'size': len(response.content),
                    'url': index_url
                })

        except Exception as e:
            self.log("WARNING", f"Could not get model index: {e}")
            # 如果没有 index，尝试猜测分片文件
            self.log("INFO", "Trying to detect model files by pattern...")

            # 尝试常见的分片模式
            for i in range(1, 21):  # 尝试 1-20 个分片
                for pattern in [f"model-{i:05d}-of-*.safetensors", f"model-{i:05d}-of-*.bin",
                              f"model-{i:05d}-of-*.safetensors", f"model-{i:05d}-of-*.bin",
                              f"model-{i:05d}-of-*.safetensors", f"model-{i:05d}-of-*.bin"]:
                    # 先尝试一些常见的总数
                    for total in [10, 15, 20, 25, 30, 35, 40, 45, 50]:
                        if pattern.endswith('.safetensors'):
                            filename = f"model-{i:05d}-of-{total:05d}.safetensors"
                        else:
                            filename = f"model-{i:05d}-of-{total:05d}.bin"

                        url = f"{self.base_url}/{model_name}/resolve/main/{filename}"
                        try:
                            head_resp = requests.head(url, timeout=5)
                            if head_resp.status_code == 200:
                                size = int(head_resp.headers.get('content-length', 0))
                                files.append({
                                    'path': filename,
                                    'size': size,
                                    'url': url
                                })
                                self.log("SUCCESS", f"Found model file: {filename}")
                                # 找到一个分片后，继续查找同一总数的其他分片
                                for j in range(i+1, total+1):
                                    if pattern.endswith('.safetensors'):
                                        next_filename = f"model-{j:05d}-of-{total:05d}.safetensors"
                                    else:
                                        next_filename = f"model-{j:05d}-of-{total:05d}.bin"
                                    next_url = f"{self.base_url}/{model_name}/resolve/main/{next_filename}"
                                    try:
                                        next_resp = requests.head(next_url, timeout=5)
                                        if next_resp.status_code == 200:
                                            next_size = int(next_resp.headers.get('content-length', 0))
                                            files.append({
                                                'path': next_filename,
                                                'size': next_size,
                                                'url': next_url
                                            })
                                            self.log("SUCCESS", f"Found model file: {next_filename}")
                                    except:
                                        continue
                                break
                        except:
                            continue
                    if files:  # 如果找到了文件，跳出循环
                        break
                if files:
                    break

        # 添加配置和其他必要文件
        essential_files = [
            'config.json', 'tokenizer.json', 'tokenizer_config.json',
            'special_tokens_map.json', 'vocab.txt', 'merges.txt', 'vocab.json',
            'generation_config.json', 'README.md'
        ]

        for filename in essential_files:
            url = f"{self.base_url}/{model_name}/resolve/main/{filename}"
            try:
                response = requests.head(url, timeout=10)
                if response.status_code == 200:
                    size = int(response.headers.get('content-length', 0))
                    # 避免重复添加
                    if not any(f['path'] == filename for f in files):
                        files.append({
                            'path': filename,
                            'size': size,
                            'url': url
                        })
                        self.log("INFO", f"Added essential file: {filename}")
            except:
                continue

        # 如果仍然没有找到权重文件，尝试单个模型文件
        weight_files_found = any('safetensors' in f['path'] or 'bin' in f['path']
                               for f in files if 'index' not in f['path'])

        if not weight_files_found:
            self.log("WARNING", "No weight files found, trying single model files...")
            single_model_files = ['model.safetensors', 'pytorch_model.bin', 
                                'model.safetensors', 'pytorch_model.bin',
                                'model-00001-of-00001.safetensors']
            for filename in single_model_files:
                url = f"{self.base_url}/{model_name}/resolve/main/{filename}"
                try:
                    response = requests.head(url, timeout=10)
                    if response.status_code == 200:
                        size = int(response.headers.get('content-length', 0))
                        files.append({
                            'path': filename,
                            'size': size,
                            'url': url
                        })
                        self.log("SUCCESS", f"Found single model file: {filename}")
                        break
                except:
                    continue

        total_size = sum(f.get('size', 0) for f in files)
        self.log("SUCCESS", f"Total files found: {len(files)}, Total size: {self._format_size(total_size)}")

        return files
        
    def download_file_aria2(self, file_info: Dict, output_dir: Path) -> bool:
        """使用 aria2c 下载单个文件"""
        file_path = file_info['path']
        file_url = file_info['url']
        file_size = file_info.get('size', 0)
        
        # 特殊处理Qwen模型的URL
        if 'Qwen' in file_url and not file_url.startswith('http'):
            file_url = f"https://huggingface.co/{file_url.split('resolve/main/')[0]}/resolve/main/{file_path}"
            
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

        # 按文件大小排序，先下载小文件，再下载大文件
        files_sorted = sorted(files, key=lambda x: x.get('size', 0))

        for i, file_info in enumerate(files_sorted, 1):
            file_path = file_info['path']
            file_size = file_info.get('size', 0)

            self.log("PROGRESS", f"[{i}/{total_files}] {file_path} ({self._format_size(file_size)})")

            # 检查文件是否已存在且大小正确
            local_file = model_dir / file_path
            if local_file.exists():
                local_size = local_file.stat().st_size
                if local_size == file_size:
                    self.log("SUCCESS", f"File already exists and complete: {file_path}")
                    success_count += 1
                    continue
                else:
                    self.log("WARNING", f"File exists but size mismatch: {file_path} (local: {local_size}, expected: {file_size})")

            # 下载文件
            if self.download_file_aria2(file_info, model_dir):
                success_count += 1

                # 验证下载的文件大小
                if local_file.exists():
                    actual_size = local_file.stat().st_size
                    if actual_size != file_size and file_size > 0:
                        self.log("WARNING", f"Downloaded file size mismatch: {file_path}")
                        self.log("WARNING", f"Expected: {file_size}, Got: {actual_size}")
                    else:
                        self.log("SUCCESS", f"Downloaded and verified: {file_path}")
            else:
                self.log("ERROR", f"Failed to download {file_path}")

        # 下载结果
        if success_count == total_files:
            self.log("SUCCESS", f"All {total_files} files downloaded successfully!")
            return True
        elif success_count > 0:
            self.log("WARNING", f"Downloaded {success_count}/{total_files} files")
            # 如果下载了大部分文件，仍然认为是成功的
            if success_count / total_files >= 0.8:
                self.log("INFO", "Download mostly successful (80%+ files)")
                return True
            return False
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
