#!/bin/bash

# aria2c 安装脚本
# 支持 Ubuntu/Debian 系统

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

# 检测系统
detect_system() {
    if [[ -f /etc/os-release ]]; then
        . /etc/os-release
        OS=$NAME
        VER=$VERSION_ID
    else
        log_error "Cannot detect operating system"
        exit 1
    fi
    
    log_info "Detected system: $OS $VER"
}

# 安装 aria2c
install_aria2() {
    log_info "Installing aria2c..."
    
    case "$OS" in
        *"Ubuntu"*|*"Debian"*)
            sudo apt-get update
            sudo apt-get install -y aria2
            ;;
        *"CentOS"*|*"Red Hat"*|*"Rocky"*)
            sudo yum install -y epel-release
            sudo yum install -y aria2
            ;;
        *"Fedora"*)
            sudo dnf install -y aria2
            ;;
        *)
            log_error "Unsupported operating system: $OS"
            log_info "Please install aria2c manually:"
            log_info "  Ubuntu/Debian: sudo apt-get install aria2"
            log_info "  CentOS/RHEL: sudo yum install aria2"
            log_info "  Fedora: sudo dnf install aria2"
            exit 1
            ;;
    esac
}

# 验证安装
verify_installation() {
    log_info "Verifying aria2c installation..."
    
    if command -v aria2c &> /dev/null; then
        local version=$(aria2c --version | head -n1)
        log_success "aria2c installed successfully: $version"
        return 0
    else
        log_error "aria2c installation failed"
        return 1
    fi
}

# 创建 aria2c 配置文件
create_config() {
    local config_dir="$HOME/.aria2"
    local config_file="$config_dir/aria2.conf"
    
    log_info "Creating aria2c configuration..."
    
    mkdir -p "$config_dir"
    
    cat > "$config_file" << 'EOF'
# aria2c 配置文件 - 针对模型下载优化

# 基本设置
continue=true
max-connection-per-server=16
max-concurrent-downloads=4
split=16
min-split-size=1M
timeout=300
retry-wait=3
max-tries=5

# 文件分配
file-allocation=falloc

# 进度显示
summary-interval=1
console-log-level=notice
show-console-readout=true
human-readable=true

# 性能优化
disk-cache=64M
piece-length=1M
stream-piece-selector=inorder

# 网络设置
user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36
connect-timeout=60
lowest-speed-limit=1K

# 日志
log-level=notice
EOF

    log_success "Configuration created: $config_file"
    echo "You can customize the configuration by editing: $config_file"
}

# 主函数
main() {
    echo "🚀 aria2c Installation Script for LocalMoE"
    echo "=========================================="
    echo ""
    
    detect_system
    
    # 检查是否已安装
    if command -v aria2c &> /dev/null; then
        local version=$(aria2c --version | head -n1)
        log_warning "aria2c is already installed: $version"
        read -p "Do you want to reinstall? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Skipping installation"
            create_config
            exit 0
        fi
    fi
    
    install_aria2
    
    if verify_installation; then
        create_config
        
        echo ""
        log_success "aria2c installation completed!"
        echo ""
        echo "=== Usage Examples ==="
        echo "1. Download model with enhanced script:"
        echo "   ./scripts/download_model.sh -m Qwen/Qwen1.5-MoE-A2.7B-Chat"
        echo ""
        echo "2. Download with Python script:"
        echo "   python scripts/download_model_aria2.py -m Qwen/Qwen1.5-MoE-A2.7B-Chat"
        echo ""
        echo "3. Use mirror for faster download:"
        echo "   ./scripts/download_model.sh --use-mirror -m mistralai/Mixtral-8x7B-Instruct-v0.1"
        echo ""
        echo "=== Configuration ==="
        echo "Config file: $HOME/.aria2/aria2.conf"
        echo "You can customize download settings by editing this file."
    else
        log_error "Installation failed!"
        exit 1
    fi
}

# 错误处理
trap 'log_error "Installation failed at line $LINENO"' ERR

# 执行主函数
main "$@"
