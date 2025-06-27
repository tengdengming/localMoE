#!/usr/bin/env python3
"""
Qwen2.5-VL-72B-Instruct模型下载脚本
使用huggingface_hub从HuggingFace下载模型文件
"""

import os
from huggingface_hub import snapshot_download
from tqdm import tqdm

# 模型配置
MODEL_NAME = "Qwen/Qwen2.5-VL-72B-Instruct"
LOCAL_DIR = "../models/Qwen2.5-VL-72B-Instruct"
CACHE_DIR = "../models/huggingface_cache"
TOKEN = os.getenv("HF_TOKEN")  # 从环境变量获取HuggingFace token

def download_model():
    """下载模型主函数"""
    os.makedirs(LOCAL_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    print(f"开始下载模型 {MODEL_NAME}...")
    print(f"本地目录: {os.path.abspath(LOCAL_DIR)}")
    print(f"缓存目录: {os.path.abspath(CACHE_DIR)}")
    
    try:
        snapshot_download(
            repo_id=MODEL_NAME,
            local_dir=LOCAL_DIR,
            cache_dir=CACHE_DIR,
            token=TOKEN,
            resume_download=True,
            allow_patterns=[
                "*.json",
                "*.bin",
                "*.model",
                "*.py",
                "*.md",
                "*.txt",
                "*.safetensors"
            ],
            ignore_patterns=[
                "*.h5",
                "*.ot",
                "*.tflite",
                "*.msgpack",
                "*.onnx"
            ]
        )
        print("\n模型下载完成!")
    except Exception as e:
        print(f"\n下载失败: {str(e)}")
        raise

if __name__ == "__main__":
    download_model()
