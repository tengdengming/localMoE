#!/usr/bin/env python3
"""
下载缺失的模型权重文件
基于已有的 model.safetensors.index.json 文件
"""

import json
import subprocess
import requests
from pathlib import Path
import sys

def log(level: str, message: str):
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

def format_size(size_bytes: int) -> str:
    """格式化文件大小"""
    if size_bytes == 0:
        return "Unknown"
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f}TB"

def download_file_aria2(file_url: str, output_dir: Path, filename: str, max_connections: int = 16) -> bool:
    """使用 aria2c 下载单个文件"""
    log("DOWNLOAD", f"Downloading {filename} with aria2c...")
    
    # aria2c 参数
    aria2_cmd = [
        "aria2c",
        "--continue=true",
        f"--max-connection-per-server={max_connections}",
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
        f"--out={filename}",
        file_url
    ]
    
    try:
        # 创建子目录
        file_output_path = output_dir / filename
        file_output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 执行下载
        result = subprocess.run(aria2_cmd, capture_output=False)
        
        if result.returncode == 0:
            log("SUCCESS", f"Downloaded {filename}")
            return True
        else:
            log("ERROR", f"Failed to download {filename}")
            return False
            
    except Exception as e:
        log("ERROR", f"Error downloading {filename}: {e}")
        return False

def download_missing_weights(model_dir: str, model_name: str, use_mirror: bool = False):
    """下载缺失的权重文件"""
    
    model_path = Path(model_dir)
    if not model_path.exists():
        log("ERROR", f"Model directory not found: {model_path}")
        return False
    
    # 检查 index 文件
    index_file = model_path / "model.safetensors.index.json"
    if not index_file.exists():
        log("ERROR", f"Index file not found: {index_file}")
        return False
    
    log("INFO", f"Reading index file: {index_file}")
    
    # 读取 index 文件
    try:
        with open(index_file, 'r') as f:
            index_data = json.load(f)
    except Exception as e:
        log("ERROR", f"Failed to read index file: {e}")
        return False
    
    # 获取权重文件列表
    weight_map = index_data.get('weight_map', {})
    if not weight_map:
        log("ERROR", "No weight_map found in index file")
        return False
    
    weight_files = set(weight_map.values())
    log("INFO", f"Found {len(weight_files)} weight files in index")
    
    # 确定下载URL
    if use_mirror:
        base_url = "https://hf-mirror.com"
        log("INFO", "Using mirror: https://hf-mirror.com")
    else:
        base_url = "https://huggingface.co"
    
    # 检查哪些文件缺失
    missing_files = []
    existing_files = []
    
    for weight_file in weight_files:
        local_file = model_path / weight_file
        if local_file.exists():
            existing_files.append(weight_file)
            log("SUCCESS", f"Already exists: {weight_file}")
        else:
            missing_files.append(weight_file)
            log("WARNING", f"Missing: {weight_file}")
    
    if not missing_files:
        log("SUCCESS", "All weight files already exist!")
        return True
    
    log("INFO", f"Need to download {len(missing_files)} missing files")
    log("INFO", f"Already have {len(existing_files)} files")
    
    # 获取文件大小信息
    files_to_download = []
    total_size = 0
    
    for weight_file in missing_files:
        file_url = f"{base_url}/{model_name}/resolve/main/{weight_file}"
        
        # 获取文件大小
        try:
            response = requests.head(file_url, timeout=30)
            if response.status_code == 200:
                size = int(response.headers.get('content-length', 0))
                files_to_download.append({
                    'filename': weight_file,
                    'url': file_url,
                    'size': size
                })
                total_size += size
                log("INFO", f"Will download: {weight_file} ({format_size(size)})")
            else:
                log("ERROR", f"Cannot access file: {weight_file} (status: {response.status_code})")
        except Exception as e:
            log("ERROR", f"Error checking file {weight_file}: {e}")
    
    if not files_to_download:
        log("ERROR", "No files can be downloaded")
        return False
    
    log("INFO", f"Total download size: {format_size(total_size)}")
    
    # 开始下载
    success_count = 0
    
    for i, file_info in enumerate(files_to_download, 1):
        filename = file_info['filename']
        file_url = file_info['url']
        file_size = file_info['size']
        
        log("PROGRESS", f"[{i}/{len(files_to_download)}] {filename} ({format_size(file_size)})")
        
        if download_file_aria2(file_url, model_path, filename):
            success_count += 1
            
            # 验证文件大小
            local_file = model_path / filename
            if local_file.exists():
                actual_size = local_file.stat().st_size
                if actual_size == file_size:
                    log("SUCCESS", f"Downloaded and verified: {filename}")
                else:
                    log("WARNING", f"Size mismatch: {filename} (expected: {file_size}, got: {actual_size})")
        else:
            log("ERROR", f"Failed to download: {filename}")
    
    # 结果统计
    if success_count == len(files_to_download):
        log("SUCCESS", f"All {len(files_to_download)} files downloaded successfully!")
        return True
    elif success_count > 0:
        log("WARNING", f"Downloaded {success_count}/{len(files_to_download)} files")
        return True
    else:
        log("ERROR", "No files were downloaded successfully")
        return False

def main():
    if len(sys.argv) < 3:
        print("Usage: python download_missing_weights.py <model_dir> <model_name> [--use-mirror]")
        print("Example: python download_missing_weights.py /data/models/Qwen3-30B-A3B Qwen/Qwen3-30B-A3B --use-mirror")
        sys.exit(1)
    
    model_dir = sys.argv[1]
    model_name = sys.argv[2]
    use_mirror = "--use-mirror" in sys.argv
    
    print("🚀 LocalMoE Missing Weights Downloader")
    print("=====================================")
    print(f"📁 Model directory: {model_dir}")
    print(f"🏷️  Model name: {model_name}")
    print(f"🌐 Use mirror: {use_mirror}")
    print()
    
    success = download_missing_weights(model_dir, model_name, use_mirror)
    
    if success:
        print()
        print("✅ Download completed successfully!")
        print("🔧 You can now use the model with LocalMoE")
    else:
        print()
        print("❌ Download failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
