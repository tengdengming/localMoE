#!/bin/bash

# vLLM MoE模型部署环境一键配置脚本
# 适用于L40S GPU服务器

set -e  # 遇到错误立即退出

echo "🚀 vLLM MoE模型部署环境一键配置"
echo "================================"

# 检查是否为root用户
if [ "$EUID" -ne 0 ]; then
    echo "❌ 请以root用户运行此脚本"
    echo "sudo $0"
    exit 1
fi

# 1. 系统更新
echo "📦 更新系统包..."
apt update && apt upgrade -y

# 2. 安装CUDA Toolkit
echo ""
echo "🔧 安装CUDA Toolkit 12.6..."
if ! command -v nvcc &> /dev/null; then
    # 下载CUDA keyring
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    dpkg -i cuda-keyring_1.1-1_all.deb
    apt-get update
    
    # 安装CUDA toolkit
    apt-get install -y cuda-toolkit-12-6
    
    # 设置环境变量
    echo 'export CUDA_HOME=/usr/local/cuda' >> /etc/environment
    echo 'export PATH=/usr/local/cuda/bin:$PATH' >> /etc/environment
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> /etc/environment
    
    # 创建符号链接
    ln -sf /usr/local/cuda-12.6 /usr/local/cuda
    
    echo "✅ CUDA安装完成，请重新登录以加载环境变量"
else
    echo "✅ CUDA已安装: $(nvcc --version | grep release)"
fi

# 3. 配置HuggingFace环境
echo ""
echo "🌐 配置HuggingFace环境..."
# 创建HuggingFace缓存目录
mkdir -p /data/huggingface_cache
chown -R emoney:emoney /data/huggingface_cache

# 添加环境变量到用户配置
cat >> /home/emoney/.bashrc << 'EOF'

# vLLM MoE环境配置
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# HuggingFace配置
export HF_HOME=/data/huggingface_cache
export HF_ENDPOINT=https://hf-mirror.com
export TRANSFORMERS_CACHE=/data/huggingface_cache
export HF_DATASETS_CACHE=/data/huggingface_cache

# vLLM优化配置
export VLLM_USE_MODELSCOPE=False
export VLLM_ATTENTION_BACKEND=FLASHINFER
export CUDA_VISIBLE_DEVICES=0,1,2,3
EOF

# 4. 安装系统级依赖
echo ""
echo "🔧 安装系统级依赖..."
apt install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    htop \
    tmux \
    screen \
    vim \
    tree \
    unzip \
    software-properties-common

# 5. 创建便捷脚本
echo ""
echo "📝 创建便捷脚本..."

# 创建激活环境脚本
cat > /home/emoney/activate_vllm.sh << 'EOF'
#!/bin/bash
# 激活vLLM环境

echo "🚀 激活vLLM MoE环境..."
source /home/emoney/localMoE/venv/bin/activate
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export HF_HOME=/data/huggingface_cache
export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "✅ 环境已激活"
echo "Python: $(python --version)"
echo "CUDA: $(nvcc --version | grep release || echo '未安装')"
echo "GPU数量: $(python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo '0')"
EOF

chmod +x /home/emoney/activate_vllm.sh
chown emoney:emoney /home/emoney/activate_vllm.sh

# 创建GPU监控脚本
cat > /home/emoney/monitor_gpu.sh << 'EOF'
#!/bin/bash
# GPU监控脚本

echo "🖥️ GPU状态监控"
echo "=============="
nvidia-smi
echo ""
echo "GPU温度和功耗:"
nvidia-smi --query-gpu=index,name,temperature.gpu,power.draw,memory.used,memory.total --format=csv
EOF

chmod +x /home/emoney/monitor_gpu.sh
chown emoney:emoney /home/emoney/monitor_gpu.sh

# 6. 设置文件权限
chown -R emoney:emoney /home/emoney/localMoE

echo ""
echo "🎉 环境配置完成!"
echo ""
echo "📋 下一步操作:"
echo "1. 重新登录或运行: source ~/.bashrc"
echo "2. 激活环境: source /home/emoney/activate_vllm.sh"
echo "3. 安装Python依赖: bash scripts/install_moe_dependencies.sh"
echo "4. 验证环境: python scripts/environment_scan.py"
echo ""
echo "🔧 便捷脚本:"
echo "- 激活环境: ~/activate_vllm.sh"
echo "- GPU监控: ~/monitor_gpu.sh"
echo ""
echo "⚠️  重要提醒: 请重新登录以加载CUDA环境变量!"
