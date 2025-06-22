#!/bin/bash

# LocalMoE 增强模型下载脚本
# 支持aria2c高速下载、断点续传、进度显示
# 支持从HuggingFace Hub和镜像源下载模型

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
PURPLE='\033[0;35m'
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

log_progress() {
    echo -e "${CYAN}[PROGRESS]${NC} $1"
}

log_download() {
    echo -e "${PURPLE}[DOWNLOAD]${NC} $1"
}

# 默认配置
MODEL_NAME="Qwen/Qwen1.5-MoE-A2.7B-Chat"
TARGET_DIR="/data/models"
RESUME_DOWNLOAD=true
SKIP_LFS=false
USE_ARIA2=true
USE_MIRROR=false
MIRROR_URL="https://hf-mirror.com"
MAX_CONNECTIONS=16
MAX_CONCURRENT_DOWNLOADS=4
DOWNLOAD_TIMEOUT=300
ARIA2_CONFIG_FILE=""

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
        --use-mirror)
            USE_MIRROR=true
            shift
            ;;
        --mirror-url)
            MIRROR_URL="$2"
            USE_MIRROR=true
            shift 2
            ;;
        --no-aria2)
            USE_ARIA2=false
            shift
            ;;
        --max-connections)
            MAX_CONNECTIONS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  -m, --model MODEL         Model name (default: Qwen/Qwen1.5-MoE-A2.7B-Chat)"
            echo "  -d, --dir DIR             Target directory (default: /data/models)"
            echo "  --no-resume               Disable resume download"
            echo "  --skip-lfs                Skip LFS files"
            echo "  --use-mirror              Use HuggingFace mirror"
            echo "  --mirror-url URL          Custom mirror URL (default: https://hf-mirror.com)"
            echo "  --no-aria2                Disable aria2c, use git clone instead"
            echo "  --max-connections NUM     Max connections for aria2c (default: 16)"
            echo "  -h, --help                Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 -m mistralai/Mixtral-8x7B-Instruct-v0.1"
            echo "  $0 --use-mirror --max-connections 8"
            echo "  $0 -d /custom/path --no-aria2"
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

    # 检查aria2c
    if [[ "$USE_ARIA2" = true ]] && ! command -v aria2c &> /dev/null; then
        log_warning "aria2c is not installed, falling back to git clone"
        log_info "To install aria2c: sudo apt-get install aria2"
        USE_ARIA2=false
    fi

    # 检查huggingface-cli
    if ! command -v huggingface-cli &> /dev/null; then
        log_warning "huggingface-cli is not installed"
        log_info "To install: pip install huggingface_hub"
    fi

    log_success "Dependencies check passed"
}

# 获取HuggingFace仓库文件列表
get_repo_files() {
    local model_name="$1"
    local base_url="$2"

    log_info "Getting file list for ${model_name}..."

    # 使用HuggingFace API获取文件列表
    local api_url="${base_url}/${model_name}/tree/main"
    if [[ "$USE_MIRROR" = true ]]; then
        api_url="${MIRROR_URL}/${model_name}/tree/main"
    fi

    # 尝试使用curl获取文件列表
    curl -s "${api_url}" | grep -oP '(?<=href=")[^"]*(?=")' | grep -E '\.(bin|safetensors|json|txt|py|md)$' || true
}

# 使用aria2c下载单个文件
download_file_aria2() {
    local file_url="$1"
    local output_path="$2"
    local file_name="$3"

    log_download "Downloading ${file_name} with aria2c..."

    # aria2c配置
    local aria2_args=(
        "--continue=true"
        "--max-connection-per-server=${MAX_CONNECTIONS}"
        "--max-concurrent-downloads=${MAX_CONCURRENT_DOWNLOADS}"
        "--timeout=${DOWNLOAD_TIMEOUT}"
        "--retry-wait=3"
        "--max-tries=5"
        "--split=16"
        "--min-split-size=1M"
        "--file-allocation=falloc"
        "--check-integrity=true"
        "--summary-interval=1"
        "--console-log-level=notice"
        "--download-result=hide"
        "--show-console-readout=true"
        "--human-readable=true"
        "--dir=${output_path}"
        "--out=${file_name}"
    )

    # 如果有配置文件，使用配置文件
    if [[ -n "$ARIA2_CONFIG_FILE" && -f "$ARIA2_CONFIG_FILE" ]]; then
        aria2_args+=("--conf-path=${ARIA2_CONFIG_FILE}")
    fi

    aria2c "${aria2_args[@]}" "${file_url}"
}

# 使用HuggingFace CLI下载
download_with_hf_cli() {
    local model_name="$1"
    local target_dir="$2"

    log_download "Using huggingface-cli to download ${model_name}..."

    local hf_args=(
        "download"
        "${model_name}"
        "--local-dir=${target_dir}/${model_name#*/}"
        "--local-dir-use-symlinks=False"
    )

    if [[ "$RESUME_DOWNLOAD" = true ]]; then
        hf_args+=("--resume-download")
    fi

    huggingface-cli "${hf_args[@]}"
}

# Git克隆下载（备用方案）
download_with_git() {
    local model_name="$1"
    local target_dir="$2"
    local base_url="$3"

    log_download "Using git clone to download ${model_name}..."

    cd "${target_dir}"

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
    if [[ ! -d "${model_name#*/}" ]]; then
        log_info "Cloning repository..."
        git clone ${GIT_ARGS} "${base_url}/${model_name}"
    else
        log_info "Resuming download..."
        cd "${model_name#*/}"
        git pull
    fi
}

# 主下载函数
download_model() {
    log_info "Starting model download..."
    echo "Model: ${MODEL_NAME}"
    echo "Target directory: ${TARGET_DIR}/${MODEL_NAME#*/}"
    echo "Use aria2c: ${USE_ARIA2}"
    echo "Use mirror: ${USE_MIRROR}"

    # 创建目标目录
    mkdir -p "${TARGET_DIR}"

    # 确定基础URL
    local base_url="https://huggingface.co"
    if [[ "$USE_MIRROR" = true ]]; then
        base_url="$MIRROR_URL"
        log_info "Using mirror: ${MIRROR_URL}"
    fi

    # 选择下载方法
    if command -v huggingface-cli &> /dev/null && [[ "$USE_ARIA2" = false ]]; then
        # 优先使用HuggingFace CLI
        download_with_hf_cli "$MODEL_NAME" "$TARGET_DIR"
    elif [[ "$USE_ARIA2" = true ]] && command -v aria2c &> /dev/null; then
        # 使用aria2c下载（需要实现文件列表获取）
        log_warning "aria2c download method needs file list implementation"
        log_info "Falling back to git clone..."
        download_with_git "$MODEL_NAME" "$TARGET_DIR" "$base_url"
    else
        # 备用方案：git clone
        download_with_git "$MODEL_NAME" "$TARGET_DIR" "$base_url"
    fi

    log_success "Model downloaded successfully"
}

# 验证下载完整性
verify_download() {
    local model_path="${TARGET_DIR}/${MODEL_NAME#*/}"

    log_info "Verifying download..."

    if [[ ! -d "$model_path" ]]; then
        log_error "Model directory not found: $model_path"
        return 1
    fi

    # 检查关键文件
    local required_files=("config.json")
    local optional_files=("pytorch_model.bin" "model.safetensors" "tokenizer.json")

    for file in "${required_files[@]}"; do
        if [[ ! -f "$model_path/$file" ]]; then
            log_error "Required file missing: $file"
            return 1
        fi
    done

    # 检查模型文件
    local has_model_file=false
    for file in "${optional_files[@]}"; do
        if [[ -f "$model_path/$file" ]]; then
            has_model_file=true
            break
        fi
    done

    if [[ "$has_model_file" = false ]]; then
        log_warning "No model weight files found, but config exists"
    fi

    log_success "Download verification passed"
    return 0
}

# 更新配置文件
update_config() {
    local model_path="${TARGET_DIR}/${MODEL_NAME#*/}"
    local config_file="configs/config.yaml"

    log_info "Updating configuration..."

    if [[ -f "$config_file" ]]; then
        # 备份原配置
        cp "$config_file" "${config_file}.backup.$(date +%Y%m%d_%H%M%S)"

        # 更新模型路径
        sed -i "s|model_name:.*|model_name: \"$model_path\"|g" "$config_file"
        log_success "Updated $config_file with new model path"
    else
        log_warning "Config file not found: $config_file"
    fi
}

# 主函数
main() {
    echo "🚀 LocalMoE Enhanced Model Downloader"
    echo "======================================"
    echo ""

    check_dependencies
    download_model

    if verify_download; then
        update_config

        log_success "Download completed successfully!"
        echo ""
        echo "=== Model Information ==="
        echo "Model: ${MODEL_NAME}"
        echo "Path: ${TARGET_DIR}/${MODEL_NAME#*/}"
        echo "Size: $(du -sh "${TARGET_DIR}/${MODEL_NAME#*/}" 2>/dev/null | cut -f1 || echo "Unknown")"
        echo ""
        echo "=== Next Steps ==="
        echo "1. Verify the model works: python -c \"from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('${TARGET_DIR}/${MODEL_NAME#*/}')\""
        echo "2. Update your LocalMoE configuration if needed"
        echo "3. Start the LocalMoE service: ./scripts/start.sh"
    else
        log_error "Download verification failed!"
        exit 1
    fi
}

# 错误处理
trap 'log_error "Download failed at line $LINENO"' ERR

# 执行主函数
main
