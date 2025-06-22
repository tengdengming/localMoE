#!/bin/bash

# vLLM MoE模型部署依赖包安装脚本
# 针对L40S GPU优化

echo "📦 安装vLLM MoE模型部署依赖包"
echo "============================="

# 安装系统依赖
echo ""
echo "🛠️ 安装系统依赖..."
apt-get update
apt-get install -y git git-lfs wget curl

# 检查Git LFS
echo ""
echo "🔍 检查Git LFS..."
git lfs install

# 激活虚拟环境
if [ -n "$VIRTUAL_ENV" ]; then
    echo "✅ 虚拟环境已激活: $VIRTUAL_ENV"
else
    echo "⚠️  请先激活虚拟环境:"
    echo "source /home/emoney/localMoE/venv/bin/activate"
    exit 1
fi

# 检查CUDA环境
echo "检查CUDA环境..."
python -c "import torch; print(f'PyTorch CUDA可用: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA设备数量: {torch.cuda.device_count()}')" 2>/dev/null

# 安装DeepSpeed (MoE模型推荐)
echo ""
echo "🚀 安装DeepSpeed..."
pip install deepspeed

# 安装Flash Attention (性能优化)
echo ""
echo "⚡ 安装Flash Attention..."
pip install flash-attn --no-build-isolation

# 安装其他优化库
echo ""
echo "🔧 安装其他优化库..."
pip install xformers
pip install triton
pip install accelerate
pip install bitsandbytes

# 安装API相关依赖
echo ""
echo "🌐 安装API服务依赖..."
pip install uvicorn[standard]
pip install pydantic
pip install sse-starlette

# 安装监控和调试工具
echo ""
echo "📊 安装监控工具..."
pip install wandb
pip install tensorboard
pip install gpustat
pip install py3nvml

# 验证安装
echo ""
echo "✅ 验证关键包安装..."
python -c "
import torch
import vllm
import transformers
import fastapi
import deepspeed
import flash_attn
import xformers

print('✅ 所有关键包安装成功!')
print(f'PyTorch: {torch.__version__}')
print(f'vLLM: {vllm.__version__}')
print(f'Transformers: {transformers.__version__}')
print(f'DeepSpeed: {deepspeed.__version__}')
print(f'Flash Attention: {flash_attn.__version__}')
print(f'xFormers: {xformers.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
print(f'GPU数量: {torch.cuda.device_count()}')
"

echo ""
echo "🎉 依赖包安装完成!"
