#!/bin/bash

# CUDA环境修复脚本
# 适用于NVIDIA L40S GPU + Ubuntu系统

echo "🔧 修复CUDA环境配置"
echo "===================="

# 检查当前CUDA状态
echo "检查当前CUDA状态..."
if command -v nvcc &> /dev/null; then
    echo "✅ CUDA编译器已安装: $(nvcc --version | grep release)"
else
    echo "❌ CUDA编译器未安装"
fi

# 检查CUDA库路径
echo "检查CUDA库路径..."
if [ -d "/usr/local/cuda" ]; then
    echo "✅ CUDA安装目录存在: /usr/local/cuda"
    ls -la /usr/local/cuda/bin/nvcc 2>/dev/null && echo "✅ nvcc编译器存在" || echo "❌ nvcc编译器缺失"
else
    echo "❌ CUDA安装目录不存在"
fi

# 检查环境变量
echo "检查环境变量..."
echo "CUDA_HOME: ${CUDA_HOME:-未设置}"
echo "PATH中的CUDA: $(echo $PATH | grep -o '/usr/local/cuda[^:]*' || echo '未设置')"
echo "LD_LIBRARY_PATH中的CUDA: $(echo $LD_LIBRARY_PATH | grep -o '/usr/local/cuda[^:]*' || echo '未设置')"

# 推荐的修复方案
echo ""
echo "🛠️ 推荐修复方案:"
echo "1. 安装CUDA Toolkit 12.6 (匹配PyTorch cu126):"
echo "   wget https://developer.download.nvidia.com/compute/cuda/12.6.0/local_installers/cuda_12.6.0_560.28.03_linux.run"
echo "   sudo sh cuda_12.6.0_560.28.03_linux.run"
echo ""
echo "2. 或者使用包管理器安装:"
echo "   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb"
echo "   sudo dpkg -i cuda-keyring_1.1-1_all.deb"
echo "   sudo apt-get update"
echo "   sudo apt-get -y install cuda-toolkit-12-6"
echo ""
echo "3. 设置环境变量 (添加到 ~/.bashrc):"
echo "   export CUDA_HOME=/usr/local/cuda"
echo "   export PATH=\$CUDA_HOME/bin:\$PATH"
echo "   export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH"
echo ""
echo "4. 重新加载环境:"
echo "   source ~/.bashrc"
echo "   nvcc --version"
