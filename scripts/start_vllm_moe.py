#!/usr/bin/env python3
"""
vLLM MoE模型启动脚本
支持多种MoE模型的自动配置和启动
针对L40S GPU优化
"""

import os
import sys
import yaml
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, List

class VLLMMoELauncher:
    def __init__(self, config_path: str = "configs/moe_models.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
        self.validate_environment()
    
    def load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"❌ 配置文件未找到: {self.config_path}")
            sys.exit(1)
        except yaml.YAMLError as e:
            print(f"❌ 配置文件格式错误: {e}")
            sys.exit(1)
    
    def validate_environment(self):
        """验证环境"""
        print("🔍 验证环境...")
        
        # 检查vLLM
        try:
            import vllm
            print(f"✅ vLLM版本: {vllm.__version__}")
        except ImportError:
            print("❌ vLLM未安装")
            sys.exit(1)
        
        # 检查GPU
        try:
            import torch
            if not torch.cuda.is_available():
                print("❌ CUDA不可用")
                sys.exit(1)
            
            gpu_count = torch.cuda.device_count()
            print(f"✅ 检测到 {gpu_count} 个GPU")
            
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
                
        except Exception as e:
            print(f"❌ GPU检查失败: {e}")
            sys.exit(1)
    
    def list_models(self):
        """列出可用模型"""
        print("\n📋 可用的MoE模型:")
        print("=" * 60)
        
        for model_key, model_config in self.config['models'].items():
            print(f"\n🤖 {model_key}")
            print(f"  模型: {model_config['model_name']}")
            print(f"  大小: {model_config['model_size']}")
            print(f"  推荐GPU数: {model_config['recommended_gpus']}")
            print(f"  描述: {model_config['description']}")
    
    def get_model_config(self, model_key: str) -> Dict[str, Any]:
        """获取模型配置"""
        if model_key not in self.config['models']:
            print(f"❌ 未知模型: {model_key}")
            self.list_models()
            sys.exit(1)
        
        return self.config['models'][model_key]
    
    def build_vllm_command(self, model_key: str, scenario: str = "production", 
                          custom_args: Dict[str, Any] = None) -> List[str]:
        """构建vLLM启动命令"""
        model_config = self.get_model_config(model_key)
        scenario_config = self.config['deployment_scenarios'].get(scenario, {})
        global_config = self.config['global_settings']
        
        # 基础命令
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_config['model_name']
        ]
        
        # 并行配置
        if model_config.get('tensor_parallel_size', 1) > 1:
            cmd.extend(["--tensor-parallel-size", str(model_config['tensor_parallel_size'])])
        
        if model_config.get('pipeline_parallel_size', 1) > 1:
            cmd.extend(["--pipeline-parallel-size", str(model_config['pipeline_parallel_size'])])
        
        # 内存和性能配置
        gpu_memory_util = scenario_config.get('gpu_memory_utilization', 
                                            model_config.get('gpu_memory_utilization', 0.85))
        cmd.extend(["--gpu-memory-utilization", str(gpu_memory_util)])
        
        max_model_len = scenario_config.get('max_model_len', 
                                          model_config.get('max_model_len', 4096))
        cmd.extend(["--max-model-len", str(max_model_len)])
        
        max_num_seqs = scenario_config.get('max_num_seqs', 
                                         global_config.get('max_num_seqs', 256))
        cmd.extend(["--max-num-seqs", str(max_num_seqs)])
        
        # 数据类型
        if model_config.get('dtype'):
            cmd.extend(["--dtype", model_config['dtype']])
        
        # 量化
        if model_config.get('quantization'):
            cmd.extend(["--quantization", model_config['quantization']])
        
        # 性能优化
        if global_config.get('enable_chunked_prefill'):
            cmd.append("--enable-chunked-prefill")
        
        if global_config.get('max_num_batched_tokens'):
            cmd.extend(["--max-num-batched-tokens", str(global_config['max_num_batched_tokens'])])
        
        # API配置
        api_config = self.config['api_settings']
        cmd.extend([
            "--host", api_config['host'],
            "--port", str(api_config['port'])
        ])
        
        # 自定义参数
        if custom_args:
            for key, value in custom_args.items():
                if value is not None:
                    cmd.extend([f"--{key.replace('_', '-')}", str(value)])
        
        return cmd
    
    def start_model(self, model_key: str, scenario: str = "production", 
                   custom_args: Dict[str, Any] = None, dry_run: bool = False):
        """启动模型"""
        print(f"\n🚀 启动模型: {model_key}")
        print(f"📋 部署场景: {scenario}")
        
        model_config = self.get_model_config(model_key)
        print(f"🤖 模型名称: {model_config['model_name']}")
        print(f"💾 模型大小: {model_config['model_size']}")
        print(f"🎮 推荐GPU数: {model_config['recommended_gpus']}")
        
        # 构建命令
        cmd = self.build_vllm_command(model_key, scenario, custom_args)
        
        print(f"\n📝 启动命令:")
        print(" ".join(cmd))
        
        if dry_run:
            print("\n🔍 干运行模式，不实际启动服务")
            return
        
        # 设置环境变量
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, range(model_config['recommended_gpus'])))
        
        print(f"\n🌍 环境变量:")
        print(f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}")
        
        try:
            print(f"\n🎯 启动vLLM服务...")
            print("=" * 60)
            subprocess.run(cmd, env=env, check=True)
        except KeyboardInterrupt:
            print("\n\n⚠️ 服务被用户中断")
        except subprocess.CalledProcessError as e:
            print(f"\n❌ 启动失败: {e}")
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="vLLM MoE模型启动器")
    parser.add_argument("--model", "-m", type=str, help="模型名称")
    parser.add_argument("--scenario", "-s", type=str, default="production",
                       choices=["development", "production", "high_throughput", "long_context"],
                       help="部署场景")
    parser.add_argument("--list", "-l", action="store_true", help="列出可用模型")
    parser.add_argument("--dry-run", action="store_true", help="干运行，只显示命令不执行")
    parser.add_argument("--config", "-c", type=str, default="configs/moe_models.yaml",
                       help="配置文件路径")
    
    # 自定义参数
    parser.add_argument("--max-model-len", type=int, help="最大模型长度")
    parser.add_argument("--max-num-seqs", type=int, help="最大序列数")
    parser.add_argument("--gpu-memory-utilization", type=float, help="GPU内存利用率")
    parser.add_argument("--tensor-parallel-size", type=int, help="张量并行大小")
    parser.add_argument("--pipeline-parallel-size", type=int, help="流水线并行大小")
    
    args = parser.parse_args()
    
    # 创建启动器
    launcher = VLLMMoELauncher(args.config)
    
    if args.list:
        launcher.list_models()
        return
    
    if not args.model:
        print("❌ 请指定模型名称")
        launcher.list_models()
        return
    
    # 构建自定义参数
    custom_args = {}
    for arg_name in ['max_model_len', 'max_num_seqs', 'gpu_memory_utilization',
                     'tensor_parallel_size', 'pipeline_parallel_size']:
        value = getattr(args, arg_name.replace('-', '_'))
        if value is not None:
            custom_args[arg_name] = value
    
    # 启动模型
    launcher.start_model(args.model, args.scenario, custom_args, args.dry_run)


if __name__ == "__main__":
    main()
