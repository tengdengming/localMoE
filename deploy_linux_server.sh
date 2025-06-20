#!/bin/bash
# LocalMoE Linux服务器一键部署脚本
# 适用于: Ubuntu 20.04+, CentOS 8+, 376GB RAM, 128 CPU, 4x L40S GPU
# 注意: DeepSpeed已被注释掉，使用vLLM+L40S优化配置

set -e

# 配置变量
PROJECT_NAME="LocalMoE"
PYTHON_VERSION="3.10"
CUDA_VERSION="12.1"
VENV_NAME="venv"
SERVICE_PORT="8000"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# 日志函数
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "${PURPLE}[STEP]${NC} $1"; }

# 显示横幅
show_banner() {
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                    LocalMoE 服务器部署                        ║"
    echo "║                                                              ║"
    echo "║  🚀 vLLM引擎 + L40S GPU优化                                   ║"
    echo "║  ⚠️  DeepSpeed已注释掉                                        ║"
    echo "║  🎯 目标: 376GB RAM, 128 CPU, 4x L40S                        ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# 检测操作系统
detect_os() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS=$NAME
        VER=$VERSION_ID
    else
        log_error "无法检测操作系统"
        exit 1
    fi
    
    log_info "检测到操作系统: $OS $VER"
}

# 检查系统要求
check_requirements() {
    log_step "检查系统要求..."
    
    # 检查是否为root用户
    if [[ $EUID -eq 0 ]]; then
        log_warning "检测到root用户，建议使用普通用户运行"
    fi
    
    # 检查GPU
    if command -v nvidia-smi &> /dev/null; then
        GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
        log_info "检测到 $GPU_COUNT 张GPU"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -4
    else
        log_error "未检测到NVIDIA GPU驱动，请先安装NVIDIA驱动"
        exit 1
    fi
    
    # 检查内存
    TOTAL_MEM=$(free -g | awk '/^Mem:/{print $2}')
    log_info "系统内存: ${TOTAL_MEM}GB"
    if [ "$TOTAL_MEM" -lt 300 ]; then
        log_warning "内存不足300GB，建议至少376GB"
    fi
    
    # 检查CPU
    CPU_CORES=$(nproc)
    log_info "CPU核心数: $CPU_CORES"
    if [ "$CPU_CORES" -lt 64 ]; then
        log_warning "CPU核心数不足64，建议至少128核心"
    fi
    
    # 检查磁盘空间
    DISK_SPACE=$(df -h . | awk 'NR==2 {print $4}')
    log_info "可用磁盘空间: $DISK_SPACE"
    
    log_success "系统要求检查完成"
}

# 安装系统依赖
install_system_deps() {
    log_step "安装系统依赖..."
    
    if [[ "$OS" == *"Ubuntu"* ]] || [[ "$OS" == *"Debian"* ]]; then
        sudo apt-get update
        sudo apt-get install -y \
            build-essential \
            python3-dev \
            python3-pip \
            python3-venv \
            git \
            wget \
            curl \
            htop \
            nvtop \
            tmux \
            vim \
            unzip \
            software-properties-common
    elif [[ "$OS" == *"CentOS"* ]] || [[ "$OS" == *"Red Hat"* ]]; then
        sudo yum update -y
        sudo yum groupinstall -y "Development Tools"
        sudo yum install -y \
            python3-devel \
            python3-pip \
            git \
            wget \
            curl \
            htop \
            tmux \
            vim \
            unzip
    else
        log_error "不支持的操作系统: $OS"
        exit 1
    fi
    
    log_success "系统依赖安装完成"
}

# 设置Python环境
setup_python_env() {
    log_step "设置Python环境..."
    
    # 创建虚拟环境
    if [ ! -d "$VENV_NAME" ]; then
        python3 -m venv $VENV_NAME
        log_success "虚拟环境创建完成"
    else
        log_info "虚拟环境已存在"
    fi
    
    # 激活虚拟环境
    source $VENV_NAME/bin/activate
    
    # 升级pip
    pip install --upgrade pip
    
    log_success "Python环境设置完成"
}

# 安装PyTorch
install_pytorch() {
    log_step "安装PyTorch (CUDA $CUDA_VERSION)..."
    
    source $VENV_NAME/bin/activate
    
    # 安装PyTorch with CUDA支持
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    
    # 验证CUDA
    python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'GPU数量: {torch.cuda.device_count()}')"
    
    log_success "PyTorch安装完成"
}

# 安装项目依赖
install_project_deps() {
    log_step "安装项目依赖..."
    
    source $VENV_NAME/bin/activate
    
    # 安装基础依赖
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    else
        log_warning "requirements.txt不存在，安装基础依赖"
        pip install fastapi uvicorn pydantic
    fi
    
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
        transformers \
        accelerate
    
    log_success "项目依赖安装完成"
}

# 配置环境
setup_environment() {
    log_step "配置环境..."
    
    # 创建必要目录
    mkdir -p logs models checkpoints data
    
    # 创建环境变量文件
    cat > .env << EOF
# LocalMoE 环境变量 (Linux服务器)
LOCALMOE_ENVIRONMENT=production
LOCALMOE_HOST=0.0.0.0
LOCALMOE_PORT=$SERVICE_PORT
LOCALMOE_WORKERS=1
LOCALMOE_GPU_COUNT=4
LOCALMOE_PREFERRED_ENGINE=vllm
LOCALMOE_MAX_CONCURRENT=200
LOCALMOE_LOG_LEVEL=info

# CUDA设置
CUDA_VISIBLE_DEVICES=0,1,2,3
NVIDIA_VISIBLE_DEVICES=all
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# 性能优化
OMP_NUM_THREADS=32
NCCL_DEBUG=WARN
NCCL_IB_DISABLE=1

# Python路径
PYTHONPATH=$(pwd)
EOF
    
    # 设置权限
    chmod 644 .env
    
    log_success "环境配置完成"
}

# 测试安装
test_installation() {
    log_step "测试安装..."
    
    source $VENV_NAME/bin/activate
    export PYTHONPATH=$(pwd)
    
    # 测试CUDA
    log_info "测试CUDA..."
    python -c "
import torch
print(f'CUDA可用: {torch.cuda.is_available()}')
print(f'GPU数量: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
        print(f'  显存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f}GB')
"
    
    # 测试vLLM
    log_info "测试vLLM..."
    python -c "
try:
    import vllm
    print('✅ vLLM导入成功')
    print(f'vLLM版本: {vllm.__version__}')
except ImportError as e:
    print(f'❌ vLLM导入失败: {e}')
"
    
    # 测试项目模块
    log_info "测试项目模块..."
    python -c "
try:
    from src.config.settings import load_settings
    print('✅ 配置模块导入成功')
except ImportError as e:
    print(f'⚠️  配置模块导入失败: {e}')
    print('这是正常的，如果这是首次部署')

try:
    from src.core.inference import VLLMInferenceEngine
    print('✅ vLLM引擎模块导入成功')
except ImportError as e:
    print(f'⚠️  vLLM引擎模块导入失败: {e}')
"
    
    log_success "安装测试完成"
}

# 创建systemd服务
create_systemd_service() {
    log_step "创建systemd服务..."
    
    SERVICE_FILE="/etc/systemd/system/localmoe.service"
    
    sudo tee $SERVICE_FILE > /dev/null << EOF
[Unit]
Description=LocalMoE Inference Service
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
Environment=PATH=$(pwd)/$VENV_NAME/bin
EnvironmentFile=$(pwd)/.env
ExecStart=$(pwd)/$VENV_NAME/bin/python -m src.api.main
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
    
    sudo systemctl daemon-reload
    sudo systemctl enable localmoe
    
    log_success "systemd服务创建完成"
    log_info "使用以下命令管理服务:"
    log_info "  sudo systemctl start localmoe    # 启动服务"
    log_info "  sudo systemctl stop localmoe     # 停止服务"
    log_info "  sudo systemctl status localmoe   # 查看状态"
    log_info "  sudo journalctl -u localmoe -f   # 查看日志"
}

# 启动服务
start_service() {
    log_step "启动LocalMoE服务..."
    
    source $VENV_NAME/bin/activate
    export PYTHONPATH=$(pwd)
    
    # 加载环境变量
    if [ -f .env ]; then
        export $(cat .env | grep -v '^#' | xargs)
    fi
    
    log_info "服务启动中..."
    log_info "访问地址: http://localhost:$SERVICE_PORT"
    log_info "API文档: http://localhost:$SERVICE_PORT/docs"
    log_info "按 Ctrl+C 停止服务"
    
    # 启动服务
    python -m src.api.main
}

# 显示使用说明
show_usage() {
    echo ""
    log_success "🎉 LocalMoE部署完成！"
    echo ""
    echo "📋 服务管理命令:"
    echo "  # 手动启动 (前台运行)"
    echo "  ./deploy_linux_server.sh start"
    echo ""
    echo "  # 使用systemd管理 (后台运行)"
    echo "  sudo systemctl start localmoe"
    echo "  sudo systemctl status localmoe"
    echo "  sudo journalctl -u localmoe -f"
    echo ""
    echo "🌐 服务地址:"
    echo "  - API服务: http://localhost:$SERVICE_PORT"
    echo "  - API文档: http://localhost:$SERVICE_PORT/docs"
    echo "  - 健康检查: http://localhost:$SERVICE_PORT/health"
    echo ""
    echo "🔧 配置文件:"
    echo "  - 主配置: configs/config.yaml"
    echo "  - L40S优化: configs/l40s_quantization_configs.yaml"
    echo "  - 环境变量: .env"
    echo ""
    echo "📊 监控命令:"
    echo "  - GPU监控: watch -n 1 nvidia-smi"
    echo "  - 系统监控: htop"
    echo "  - 服务日志: tail -f logs/localmoe.log"
}

# 主函数
main() {
    show_banner
    detect_os
    check_requirements
    install_system_deps
    setup_python_env
    install_pytorch
    install_project_deps
    setup_environment
    test_installation
    create_systemd_service
    show_usage
}

# 处理命令行参数
case "${1:-}" in
    "start")
        start_service
        ;;
    "test")
        source $VENV_NAME/bin/activate
        export PYTHONPATH=$(pwd)
        test_installation
        ;;
    *)
        main
        ;;
esac
