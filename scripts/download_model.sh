#!/bin/bash

# LocalMoE 模型下载脚本
# 支持从HuggingFace Hub下载模型

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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
MODEL_NAME="deepseek-ai/deepseek-32b-chat"
TARGET_DIR="models"
RESUME_DOWNLOAD=true
SKIP_LFS=false

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            MODEL_NAME="$2"
            shift 2
            ;;
        -d|--dir)
            TARGET_DIR="$2"
            shift 2
            ;;
        --no-resume)
            RESUME_DOWNLOAD=false
            shift
            ;;
        --skip-lfs)
            SKIP_LFS=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  -m, --model MODEL    Model name (default: deepseek-ai/deepseek-32b-chat)"
            echo "  -d, --dir DIR        Target directory (default: models)"
            echo "  --no-resume          Disable resume download"
            echo "  --skip-lfs           Skip LFS files"
            echo "  -h, --help           Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# 检查依赖
check_dependencies() {
    log_info "Checking dependencies..."
    
    # 检查git
    if ! command -v git &> /dev/null; then
        log_error "Git is not installed"
        exit 1
    fi
    
    # 检查git-lfs
    if [[ "$SKIP_LFS" = false ]] && ! command -v git-lfs &> /dev/null; then
        log_error "Git LFS is not installed"
        exit 1
    fi
    
    log_success "Dependencies check passed"
}

# 下载模型
download_model() {
    log_info "Starting model download..."
    echo "Model: ${MODEL_NAME}"
    echo "Target directory: ${TARGET_DIR}/${MODEL_NAME#*/}"
    
    mkdir -p "${TARGET_DIR}"
    cd "${TARGET_DIR}"
    
    # 初始化git lfs
    if [[ "$SKIP_LFS" = false ]]; then
        git lfs install
    fi
    
    # 下载参数
    local GIT_ARGS=""
    if [[ "$RESUME_DOWNLOAD" = true ]]; then
        GIT_ARGS+=" --depth 1"
    fi
    
    # 克隆仓库
    if [[ ! -d "${MODEL_NAME#*/}" ]]; then
        log_info "Cloning repository..."
        git clone ${GIT_ARGS} "https://huggingface.co/${MODEL_NAME}"
    else
        log_info "Resuming download..."
        cd "${MODEL_NAME#*/}"
        git pull
    fi
    
    log_success "Model downloaded successfully"
}

# 主函数
main() {
    check_dependencies
    download_model
    
    log_success "Download completed!"
    echo ""
    echo "=== Model Information ==="
    echo "Model path: ${TARGET_DIR}/${MODEL_NAME#*/}"
    echo ""
    echo "You can now use this model with LocalMoE by updating configs/config.yaml"
}

# 错误处理
trap 'log_error "Download failed at line $LINENO"' ERR

# 执行主函数
main
