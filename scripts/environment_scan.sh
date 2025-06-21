#!/bin/bash

# vLLM MoE模型部署环境扫描脚本 (Bash版本)
# 用于快速检查服务器环境是否满足部署要求

echo "🚀 vLLM MoE模型部署环境快速扫描"
echo "=================================="

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 检查函数
check_command() {
    if command -v "$1" &> /dev/null; then
        echo -e "${GREEN}✅${NC} $1 已安装"
        return 0
    else
        echo -e "${RED}❌${NC} $1 未安装"
        return 1
    fi
}

print_section() {
    echo -e "\n${BLUE}📋 $1${NC}"
    echo "----------------------------------------"
}

# 系统基本信息
print_section "系统信息"
echo "操作系统: $(uname -s)"
echo "内核版本: $(uname -r)"
echo "架构: $(uname -m)"
echo "主机名: $(hostname)"
echo "当前用户: $(whoami)"
echo "系统启动时间: $(uptime -s 2>/dev/null || echo "无法获取")"

# 硬件信息
print_section "硬件信息"
echo "CPU信息:"
if [ -f /proc/cpuinfo ]; then
    cpu_model=$(grep "model name" /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)
    cpu_cores=$(grep "processor" /proc/cpuinfo | wc -l)
    echo "  型号: $cpu_model"
    echo "  核心数: $cpu_cores"
    
    # 检查CPU特性
    cpu_flags=$(grep "flags" /proc/cpuinfo | head -1 | cut -d: -f2)
    important_flags=("avx" "avx2" "avx512f" "fma" "sse4_1" "sse4_2")
    echo "  支持的重要特性:"
    for flag in "${important_flags[@]}"; do
        if echo "$cpu_flags" | grep -q "$flag"; then
            echo -e "    ${GREEN}✅${NC} $flag"
        else
            echo -e "    ${RED}❌${NC} $flag"
        fi
    done
fi

# 内存信息
echo "内存信息:"
if [ -f /proc/meminfo ]; then
    total_mem=$(grep "MemTotal" /proc/meminfo | awk '{print int($2/1024/1024)}')
    available_mem=$(grep "MemAvailable" /proc/meminfo | awk '{print int($2/1024/1024)}')
    echo "  总内存: ${total_mem}GB"
    echo "  可用内存: ${available_mem}GB"
    
    if [ "$total_mem" -lt 32 ]; then
        echo -e "  ${RED}⚠️  内存不足，建议至少32GB${NC}"
    elif [ "$total_mem" -lt 64 ]; then
        echo -e "  ${YELLOW}⚠️  内存偏低，推荐64GB+${NC}"
    else
        echo -e "  ${GREEN}✅ 内存充足${NC}"
    fi
fi

# GPU信息
print_section "GPU信息"
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA驱动信息:"
    nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1 | xargs -I {} echo "  驱动版本: {}"
    
    echo "GPU详细信息:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free,temperature.gpu --format=csv,noheader,nounits | while IFS=, read -r index name memory_total memory_free temp; do
        memory_total_gb=$((memory_total / 1024))
        memory_free_gb=$((memory_free / 1024))
        echo "  GPU $index: $name"
        echo "    显存: ${memory_total_gb}GB (可用: ${memory_free_gb}GB)"
        echo "    温度: ${temp}°C"
    done
    
    # 检查总显存
    total_vram=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | awk '{sum+=$1} END {print int(sum/1024)}')
    gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
    echo "  GPU总数: $gpu_count"
    echo "  总显存: ${total_vram}GB"
    
    if [ "$total_vram" -lt 48 ]; then
        echo -e "  ${YELLOW}⚠️  显存偏低，大型MoE模型建议48GB+${NC}"
    else
        echo -e "  ${GREEN}✅ 显存充足${NC}"
    fi
else
    echo -e "${RED}❌ 未检测到NVIDIA GPU或驱动${NC}"
fi

# CUDA信息
print_section "CUDA环境"
if command -v nvcc &> /dev/null; then
    cuda_version=$(nvcc --version | grep "release" | awk '{print $6}' | cut -d, -f1)
    echo -e "${GREEN}✅${NC} CUDA版本: $cuda_version"
else
    echo -e "${RED}❌ CUDA未安装${NC}"
fi

# Python环境
print_section "Python环境"
if command -v python3 &> /dev/null; then
    python_version=$(python3 --version)
    echo -e "${GREEN}✅${NC} $python_version"
    echo "Python路径: $(which python3)"
    
    if [ -n "$VIRTUAL_ENV" ]; then
        echo "虚拟环境: $VIRTUAL_ENV"
    fi
    
    if [ -n "$CONDA_DEFAULT_ENV" ]; then
        echo "Conda环境: $CONDA_DEFAULT_ENV"
    fi
    
    # 检查pip
    if python3 -m pip --version &> /dev/null; then
        pip_version=$(python3 -m pip --version)
        echo -e "${GREEN}✅${NC} $pip_version"
    else
        echo -e "${RED}❌ pip未安装${NC}"
    fi
else
    echo -e "${RED}❌ Python3未安装${NC}"
fi

# 关键依赖包检查
print_section "关键依赖包"
key_packages=("torch" "vllm" "transformers" "fastapi" "deepspeed" "flash_attn")

for package in "${key_packages[@]}"; do
    if python3 -c "import $package; print($package.__version__)" 2>/dev/null; then
        version=$(python3 -c "import $package; print($package.__version__)" 2>/dev/null)
        echo -e "${GREEN}✅${NC} $package: $version"
    else
        echo -e "${RED}❌${NC} $package: 未安装"
    fi
done

# 网络检查
print_section "网络连接"
if ping -c 1 8.8.8.8 &> /dev/null; then
    echo -e "${GREEN}✅${NC} 互联网连接正常"
else
    echo -e "${RED}❌${NC} 互联网连接异常"
fi

if ping -c 1 huggingface.co &> /dev/null; then
    echo -e "${GREEN}✅${NC} HuggingFace连接正常"
else
    echo -e "${YELLOW}⚠️${NC} HuggingFace连接异常"
fi

# 存储检查
print_section "存储空间"
df -h | grep -E "^/dev" | while read filesystem size used avail use_percent mount; do
    echo "$mount: $avail 可用 (总计: $size)"
done

# 检查临时目录空间
tmp_space=$(df -h /tmp | tail -1 | awk '{print $4}')
echo "/tmp: $tmp_space 可用"

# 系统工具检查
print_section "系统工具"
tools=("git" "wget" "curl" "htop" "tmux" "screen")
for tool in "${tools[@]}"; do
    check_command "$tool"
done

echo -e "\n${BLUE}🔍 扫描完成！${NC}"
echo "建议运行Python版本的详细扫描脚本获取更多信息："
echo "python3 scripts/environment_scan.py"
