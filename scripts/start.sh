#!/bin/bash

# LocalMoE 启动脚本
# 用于开发和生产环境的服务启动

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 默认配置
ENVIRONMENT="development"
CONFIG_FILE="configs/config.yaml"
HOST="0.0.0.0"
PORT=8000
WORKERS=1
LOG_LEVEL="info"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -h|--host)
            HOST="$2"
            shift 2
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -w|--workers)
            WORKERS="$2"
            shift 2
            ;;
        -l|--log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  -e, --environment ENV    Set environment (development/production)"
            echo "  -c, --config FILE        Configuration file path"
            echo "  -h, --host HOST          Host to bind to (default: 0.0.0.0)"
            echo "  -p, --port PORT          Port to bind to (default: 8000)"
            echo "  -w, --workers NUM        Number of workers (default: 1)"
            echo "  -l, --log-level LEVEL    Log level (default: info)"
            echo "  --help                   Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

log_info "Starting LocalMoE service..."
log_info "Environment: $ENVIRONMENT"
log_info "Config: $CONFIG_FILE"
log_info "Host: $HOST"
log_info "Port: $PORT"

# 检查Python环境
check_python_env() {
    log_info "Checking Python environment..."
    
    if ! command -v python &> /dev/null; then
        log_error "Python is not installed"
        exit 1
    fi
    
    # 检查Python版本
    python_version=$(python --version 2>&1 | cut -d' ' -f2)
    log_info "Python version: $python_version"
    
    # 检查必要的包
    required_packages=("torch" "fastapi" "uvicorn")
    for package in "${required_packages[@]}"; do
        if ! python -c "import $package" &> /dev/null; then
            log_error "Required package '$package' is not installed"
            log_info "Please run: pip install -r requirements.txt"
            exit 1
        fi
    done
    
    log_success "Python environment check passed"
}

# 检查GPU环境
check_gpu_env() {
    log_info "Checking GPU environment..."
    
    if ! command -v nvidia-smi &> /dev/null; then
        log_warning "nvidia-smi not found, GPU support may not be available"
        return
    fi
    
    # 检查GPU数量
    gpu_count=$(nvidia-smi --list-gpus | wc -l)
    log_info "Detected $gpu_count GPU(s)"
    
    if [[ $gpu_count -eq 0 ]]; then
        log_warning "No GPUs detected"
    else
        # 显示GPU信息
        nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits | \
        while IFS=, read -r name memory_total memory_used; do
            log_info "GPU: $name, Memory: ${memory_used}MB/${memory_total}MB"
        done
    fi
}

# 检查配置文件
check_config() {
    log_info "Checking configuration..."
    
    if [[ ! -f "$CONFIG_FILE" ]]; then
        log_error "Configuration file not found: $CONFIG_FILE"
        exit 1
    fi
    
    # 验证配置文件格式
    if ! python -c "import yaml; yaml.safe_load(open('$CONFIG_FILE'))" &> /dev/null; then
        log_error "Invalid YAML configuration file: $CONFIG_FILE"
        exit 1
    fi
    
    log_success "Configuration file is valid"
}

# 创建必要的目录
create_directories() {
    log_info "Creating necessary directories..."
    
    directories=("logs" "models" "cache")
    for dir in "${directories[@]}"; do
        if [[ ! -d "$dir" ]]; then
            mkdir -p "$dir"
            log_info "Created directory: $dir"
        fi
    done
}

# 设置环境变量
setup_environment() {
    log_info "Setting up environment variables..."
    
    export PYTHONPATH="${PYTHONPATH}:$(pwd)"
    export LOCALMOE_ENVIRONMENT="$ENVIRONMENT"
    export LOCALMOE_CONFIG_FILE="$CONFIG_FILE"
    export LOCALMOE_HOST="$HOST"
    export LOCALMOE_PORT="$PORT"
    export LOCALMOE_LOG_LEVEL="$LOG_LEVEL"
    
    # GPU相关环境变量
    if [[ -n "$CUDA_VISIBLE_DEVICES" ]]; then
        log_info "Using CUDA devices: $CUDA_VISIBLE_DEVICES"
    else
        export CUDA_VISIBLE_DEVICES="0,1,2,3"
        log_info "Set CUDA devices: $CUDA_VISIBLE_DEVICES"
    fi
    
    # 优化环境变量
    export OMP_NUM_THREADS=8
    export TOKENIZERS_PARALLELISM=false
    
    log_success "Environment variables set"
}

# 启动服务
start_service() {
    log_info "Starting LocalMoE service..."
    
    # 构建启动命令
    cmd="python -m uvicorn src.api.main:app"
    cmd="$cmd --host $HOST"
    cmd="$cmd --port $PORT"
    cmd="$cmd --workers $WORKERS"
    cmd="$cmd --log-level $LOG_LEVEL"
    
    if [[ "$ENVIRONMENT" == "development" ]]; then
        cmd="$cmd --reload"
        log_info "Development mode: auto-reload enabled"
    fi
    
    log_info "Starting command: $cmd"
    
    # 启动服务
    exec $cmd
}

# 信号处理
cleanup() {
    log_info "Shutting down LocalMoE service..."
    exit 0
}

trap cleanup SIGINT SIGTERM

# 主函数
main() {
    check_python_env
    check_gpu_env
    check_config
    create_directories
    setup_environment
    start_service
}

# 执行主函数
main
