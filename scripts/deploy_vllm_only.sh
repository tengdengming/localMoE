#!/bin/bash
# LocalMoE Linux服务器部署脚本 (仅vLLM引擎)
# 适用于内网服务器环境 (376GB RAM, 128 CPU, 4x L40S GPU)
# 注意：DeepSpeed已被注释掉，只使用vLLM引擎
#
# 使用方法:
#   chmod +x scripts/deploy_vllm_only.sh
#   ./scripts/deploy_vllm_only.sh
#   ./scripts/deploy_vllm_only.sh start

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

echo "🚀 LocalMoE 简化部署 (仅vLLM引擎)"
echo "适用于: 376GB RAM, 128 CPU, 4x L40S GPU"
echo "注意: DeepSpeed已被注释掉"
echo ""

# 检查系统要求
check_requirements() {
    log_info "检查系统要求..."
    
    # 检查GPU
    if ! command -v nvidia-smi &> /dev/null; then
        log_error "未检测到NVIDIA GPU驱动"
        exit 1
    fi
    
    # 检查GPU数量和型号
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    log_info "检测到 $GPU_COUNT 张GPU"
    nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -4
    
    # 检查内存
    TOTAL_MEM=$(free -g | awk '/^Mem:/{print $2}')
    log_info "系统内存: ${TOTAL_MEM}GB"
    
    # 检查Python
    if ! command -v python3 &> /dev/null; then
        log_error "未检测到Python3"
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 --version)
    log_info "Python版本: $PYTHON_VERSION"
    
    log_success "系统要求检查完成"
}

# 创建Python虚拟环境
setup_python_env() {
    log_info "设置Python环境..."
    
    # 创建虚拟环境
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        log_success "虚拟环境创建完成"
    else
        log_info "虚拟环境已存在"
    fi
    
    # 激活虚拟环境
    source venv/bin/activate
    
    # 升级pip
    pip install --upgrade pip
    log_success "pip升级完成"
}

# 安装项目依赖
install_dependencies() {
    log_info "安装项目依赖..."
    
    source venv/bin/activate
    
    # 安装PyTorch (CUDA 12.1)
    log_info "安装PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    
    # 安装基础依赖
    log_info "安装基础依赖..."
    pip install -r requirements.txt
    
    # 安装vLLM (跳过DeepSpeed)
    log_info "安装vLLM引擎..."
    pip install vllm>=0.2.0
    
    # 安装优化库
    log_info "安装优化库..."
    pip install \
        flash-attn \
        xformers \
        triton \
        pynvml \
        zstandard \
        fastapi \
        uvicorn \
        pydantic
    
    log_success "依赖安装完成"
}

# 配置环境
setup_environment() {
    log_info "配置环境..."
    
    # 创建必要目录
    mkdir -p logs models checkpoints data
    
    # 创建环境变量文件
    cat > .env << EOF
# LocalMoE 环境变量 (仅vLLM)
LOCALMOE_ENVIRONMENT=production
LOCALMOE_HOST=0.0.0.0
LOCALMOE_PORT=8000
LOCALMOE_WORKERS=1
LOCALMOE_GPU_COUNT=4
LOCALMOE_PREFERRED_ENGINE=vllm
LOCALMOE_MAX_CONCURRENT=100
LOCALMOE_LOG_LEVEL=info

# CUDA设置
CUDA_VISIBLE_DEVICES=0,1,2,3
NVIDIA_VISIBLE_DEVICES=all

# Python路径
PYTHONPATH=$(pwd)
EOF
    
    log_success "环境配置完成"
}

# 测试安装
test_installation() {
    log_info "测试安装..."
    
    source venv/bin/activate
    
    # 测试CUDA
    log_info "测试CUDA..."
    python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
"
    
    # 测试vLLM
    log_info "测试vLLM..."
    python3 -c "
try:
    import vllm
    print('✅ vLLM导入成功')
except ImportError as e:
    print(f'❌ vLLM导入失败: {e}')
"
    
    # 测试项目模块
    log_info "测试项目模块..."
    export PYTHONPATH=$(pwd)
    python3 -c "
try:
    from src.config.settings import load_settings
    print('✅ 配置模块导入成功')
    from src.core.inference import VLLMInferenceEngine
    print('✅ vLLM引擎模块导入成功')
except ImportError as e:
    print(f'❌ 模块导入失败: {e}')
"
    
    log_success "安装测试完成"
}

# 启动服务
start_service() {
    log_info "启动LocalMoE服务..."
    
    source venv/bin/activate
    export PYTHONPATH=$(pwd)
    
    # 加载环境变量
    if [ -f .env ]; then
        export $(cat .env | grep -v '^#' | xargs)
    fi
    
    log_info "服务启动中..."
    log_info "访问地址: http://localhost:8000"
    log_info "API文档: http://localhost:8000/docs"
    
    # 启动服务
    python -m src.api.main
}

# 显示使用说明
show_usage() {
    echo ""
    log_success "🎉 部署完成！"
    echo ""
    echo "手动启动服务:"
    echo "  source venv/bin/activate"
    echo "  export PYTHONPATH=\$(pwd)"
    echo "  python -m src.api.main"
    echo ""
    echo "或者运行:"
    echo "  ./scripts/deploy_vllm_only.sh start"
    echo ""
    echo "服务地址:"
    echo "  - API服务: http://localhost:8000"
    echo "  - API文档: http://localhost:8000/docs"
    echo "  - 健康检查: http://localhost:8000/health"
}

# 主函数
main() {
    check_requirements
    setup_python_env
    install_dependencies
    setup_environment
    test_installation
    show_usage
}

# 处理命令行参数
case "${1:-}" in
    "start")
        start_service
        ;;
    "test")
        source venv/bin/activate
        export PYTHONPATH=$(pwd)
        test_installation
        ;;
    *)
        main
        ;;
esac
