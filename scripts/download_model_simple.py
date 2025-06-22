#!/usr/bin/env python3
"""
LocalMoE 简单可靠的模型下载器
使用 huggingface-cli 或 git clone，确保下载完整模型
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

class SimpleModelDownloader:
    def __init__(self, target_dir: str = "/data/models", use_mirror: bool = False):
        self.target_dir = Path(target_dir)
        self.use_mirror = use_mirror
        self.mirror_url = "https://hf-mirror.com"
        
        # 创建目标目录
        self.target_dir.mkdir(parents=True, exist_ok=True)
        
    def log(self, level: str, message: str):
        """彩色日志输出"""
        colors = {
            'INFO': '\033[0;34m',
            'SUCCESS': '\033[0;32m', 
            'WARNING': '\033[1;33m',
            'ERROR': '\033[0;31m',
            'PROGRESS': '\033[0;36m'
        }
        reset = '\033[0m'
        print(f"{colors.get(level, '')}[{level}]{reset} {message}")
        
    def check_dependencies(self):
        """检查可用的下载工具"""
        tools = {}
        
        # 检查 huggingface-cli
        try:
            result = subprocess.run(['huggingface-cli', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                tools['hf_cli'] = True
                self.log("SUCCESS", "huggingface-cli is available")
            else:
                tools['hf_cli'] = False
        except:
            tools['hf_cli'] = False
            
        # 检查 git
        try:
            result = subprocess.run(['git', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                tools['git'] = True
                self.log("SUCCESS", "git is available")
            else:
                tools['git'] = False
        except:
            tools['git'] = False
            
        # 检查 git-lfs
        try:
            result = subprocess.run(['git', 'lfs', 'version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                tools['git_lfs'] = True
                self.log("SUCCESS", "git-lfs is available")
            else:
                tools['git_lfs'] = False
        except:
            tools['git_lfs'] = False
            
        return tools
        
    def download_with_hf_cli(self, model_name: str) -> bool:
        """使用 huggingface-cli 下载"""
        self.log("INFO", f"Downloading {model_name} with huggingface-cli...")
        
        model_dir = self.target_dir / model_name.split('/')[-1]
        
        cmd = [
            'huggingface-cli', 'download',
            model_name,
            '--local-dir', str(model_dir),
            '--local-dir-use-symlinks', 'False',
            '--resume-download'
        ]
        
        # 设置环境变量
        env = os.environ.copy()
        if self.use_mirror:
            env['HF_ENDPOINT'] = self.mirror_url
            self.log("INFO", f"Using mirror: {self.mirror_url}")
            
        try:
            self.log("PROGRESS", "Starting download...")
            result = subprocess.run(cmd, env=env, text=True)
            
            if result.returncode == 0:
                self.log("SUCCESS", "Download completed with huggingface-cli")
                return True
            else:
                self.log("ERROR", f"huggingface-cli failed with code {result.returncode}")
                return False
                
        except Exception as e:
            self.log("ERROR", f"Error running huggingface-cli: {e}")
            return False
            
    def download_with_git(self, model_name: str) -> bool:
        """使用 git clone 下载"""
        self.log("INFO", f"Downloading {model_name} with git clone...")
        
        # 确定URL
        if self.use_mirror:
            base_url = self.mirror_url
            self.log("INFO", f"Using mirror: {self.mirror_url}")
        else:
            base_url = "https://huggingface.co"
            
        repo_url = f"{base_url}/{model_name}"
        model_dir = self.target_dir / model_name.split('/')[-1]
        
        try:
            # 切换到目标目录
            os.chdir(self.target_dir)
            
            # 初始化 git lfs
            subprocess.run(['git', 'lfs', 'install'], check=False)
            
            # 克隆仓库
            if model_dir.exists():
                self.log("INFO", "Model directory exists, pulling updates...")
                os.chdir(model_dir)
                result = subprocess.run(['git', 'pull'], text=True)
            else:
                self.log("PROGRESS", "Cloning repository...")
                cmd = ['git', 'clone', repo_url, model_dir.name]
                result = subprocess.run(cmd, text=True)
                
            if result.returncode == 0:
                self.log("SUCCESS", "Download completed with git clone")
                return True
            else:
                self.log("ERROR", f"git clone failed with code {result.returncode}")
                return False
                
        except Exception as e:
            self.log("ERROR", f"Error running git clone: {e}")
            return False
            
    def verify_download(self, model_name: str) -> bool:
        """验证下载完整性"""
        model_dir = self.target_dir / model_name.split('/')[-1]
        
        if not model_dir.exists():
            self.log("ERROR", f"Model directory not found: {model_dir}")
            return False
            
        # 检查必要文件
        required_files = ['config.json']
        for file in required_files:
            if not (model_dir / file).exists():
                self.log("ERROR", f"Required file missing: {file}")
                return False
                
        # 检查模型权重文件
        weight_files = list(model_dir.glob('*.safetensors')) + \
                     list(model_dir.glob('*.bin')) + \
                     list(model_dir.glob('model-*.safetensors'))
                     
        if not weight_files:
            self.log("WARNING", "No model weight files found")
            return False
            
        # 计算总大小
        total_size = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file())
        size_gb = total_size / (1024**3)
        
        self.log("SUCCESS", f"Download verified! Total size: {size_gb:.1f} GB")
        self.log("INFO", f"Found {len(weight_files)} weight files")
        
        return True
        
    def download_model(self, model_name: str) -> bool:
        """下载模型主函数"""
        self.log("INFO", f"Starting download of {model_name}")
        self.log("INFO", f"Target directory: {self.target_dir}")
        
        # 检查依赖
        tools = self.check_dependencies()
        
        # 选择下载方法
        success = False
        
        if tools['hf_cli']:
            self.log("INFO", "Using huggingface-cli (recommended)")
            success = self.download_with_hf_cli(model_name)
        elif tools['git'] and tools['git_lfs']:
            self.log("INFO", "Using git clone with LFS")
            success = self.download_with_git(model_name)
        else:
            self.log("ERROR", "No suitable download tool found!")
            self.log("INFO", "Please install: pip install huggingface_hub")
            self.log("INFO", "Or install: git and git-lfs")
            return False
            
        if success:
            return self.verify_download(model_name)
        else:
            return False

def main():
    parser = argparse.ArgumentParser(description="Simple and reliable model downloader")
    parser.add_argument("-m", "--model", required=True, 
                       help="Model name (e.g., Qwen/Qwen3-30B-A3B)")
    parser.add_argument("-d", "--dir", default="/data/models", 
                       help="Target directory (default: /data/models)")
    parser.add_argument("--use-mirror", action="store_true", 
                       help="Use HuggingFace mirror for faster download")
    
    args = parser.parse_args()
    
    print("🚀 LocalMoE Simple Model Downloader")
    print("===================================")
    print()
    
    downloader = SimpleModelDownloader(
        target_dir=args.dir,
        use_mirror=args.use_mirror
    )
    
    success = downloader.download_model(args.model)
    
    if success:
        print()
        print("✅ Download completed successfully!")
        print(f"📁 Model location: {args.dir}/{args.model.split('/')[-1]}")
        print()
        print("🔧 Next steps:")
        print("1. Update your LocalMoE configuration")
        print("2. Test the model loading")
        print("3. Start your LocalMoE service")
    else:
        print()
        print("❌ Download failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
