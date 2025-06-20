#!/bin/bash

# LocalMoE 部署脚本
# 支持开发和生产环境部署

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
ENVIRONMENT="development"
BUILD_IMAGE=true
START_SERVICES=true
SKIP_TESTS=false
CONFIG_FILE=""

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --no-build)
            BUILD_IMAGE=false
            shift
            ;;
        --no-start)
            START_SERVICES=false
            shift
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  -e, --environment ENV    Set environment (development/production)"
            echo "  --no-build              Skip Docker image build"
            echo "  --no-start              Skip starting services"
            echo "  --skip-tests            Skip running tests"
            echo "  -c, --config FILE       Use specific config file"
            echo "  -h, --help              Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

log_info "Starting LocalMoE deployment..."
log_info "Environment: $ENVIRONMENT"

# 检查依赖
check_dependencies() {
    log_info "Checking dependencies..."
    
    # 检查Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    # 检查Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi
    
    # 检查NVIDIA Docker
    if ! docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi &> /dev/null; then
        log_warning "NVIDIA Docker runtime not available or no GPUs detected"
    fi
    
    log_success "Dependencies check passed"
}

# 创建必要的目录
create_directories() {
    log_info "Creating necessary directories..."
    
    mkdir -p logs
    mkdir -p models
    mkdir -p configs
    mkdir -p cache
    mkdir -p docker/ssl
    
    log_success "Directories created"
}

# 生成配置文件
generate_config() {
    log_info "Generating configuration files..."
    
    # 生成主配置文件
    if [[ -z "$CONFIG_FILE" ]]; then
        CONFIG_FILE="configs/config.yaml"
    fi
    
    if [[ ! -f "$CONFIG_FILE" ]]; then
        cat > "$CONFIG_FILE" << EOF
# LocalMoE Configuration
environment: $ENVIRONMENT
debug: $([ "$ENVIRONMENT" = "development" ] && echo "true" || echo "false")

gpu:
  device_count: 4
  memory_limit_gb: 40.0
  utilization_threshold: 0.85
  temperature_threshold: 85.0
  enable_monitoring: true
  monitoring_interval: 1.0

model:
  num_experts: 8
  top_k_experts: 2
  hidden_size: 768
  intermediate_size: 3072
  max_sequence_length: 2048
  quantization_type: "fp16"

inference:
  preferred_engine: "auto"
  enable_fallback: true
  enable_load_balancing: true
  max_concurrent_requests: 100
  request_timeout: 30.0
  enable_batching: true
  max_batch_size: 32

api:
  host: "0.0.0.0"
  port: 8000
  workers: 1
  log_level: "info"
  enable_cors: true
  rate_limit_calls: 100
  rate_limit_period: 60

monitoring:
  enable_metrics: true
  metrics_interval: 10.0
  enable_prometheus: true
  prometheus_port: 9090
  enable_logging: true
  log_file: "logs/localmoe.log"

security:
  enable_auth: $([ "$ENVIRONMENT" = "production" ] && echo "true" || echo "false")
  enable_rate_limiting: true
  max_request_size_mb: 10
EOF
        log_success "Generated config file: $CONFIG_FILE"
    else
        log_info "Using existing config file: $CONFIG_FILE"
    fi
    
    # 生成Prometheus配置
    cat > docker/prometheus.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'localmoe'
    static_configs:
      - targets: ['localmoe:9090']
    scrape_interval: 5s
    metrics_path: /v1/metrics

  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
EOF
    
    log_success "Configuration files generated"
}

# 构建Docker镜像
build_image() {
    if [[ "$BUILD_IMAGE" = true ]]; then
        log_info "Building Docker image..."
        
        cd docker
        docker-compose build --no-cache localmoe
        cd ..
        
        log_success "Docker image built successfully"
    else
        log_info "Skipping Docker image build"
    fi
}

# 运行测试
run_tests() {
    if [[ "$SKIP_TESTS" = false ]]; then
        log_info "Running tests..."
        
        # 运行单元测试
        if [[ -f "tests/test_basic.py" ]]; then
            python -m pytest tests/ -v
            log_success "Tests passed"
        else
            log_warning "No tests found, skipping"
        fi
    else
        log_info "Skipping tests"
    fi
}

# 启动服务
start_services() {
    if [[ "$START_SERVICES" = true ]]; then
        log_info "Starting services..."
        
        cd docker
        
        # 停止现有服务
        docker-compose down
        
        # 启动服务
        docker-compose up -d
        
        cd ..
        
        # 等待服务启动
        log_info "Waiting for services to start..."
        sleep 30
        
        # 检查服务状态
        check_services
        
        log_success "Services started successfully"
    else
        log_info "Skipping service startup"
    fi
}

# 检查服务状态
check_services() {
    log_info "Checking service status..."
    
    # 检查主服务
    if curl -f http://localhost:8000/health &> /dev/null; then
        log_success "LocalMoE service is healthy"
    else
        log_error "LocalMoE service is not responding"
        return 1
    fi
    
    # 检查Prometheus
    if curl -f http://localhost:9091 &> /dev/null; then
        log_success "Prometheus is running"
    else
        log_warning "Prometheus is not responding"
    fi
    
    # 显示服务信息
    echo ""
    log_info "Service URLs:"
    echo "  LocalMoE API: http://localhost:8000"
    echo "  API Documentation: http://localhost:8000/docs"
    echo "  Health Check: http://localhost:8000/health"
    echo "  Metrics: http://localhost:8000/v1/metrics"
    echo "  Prometheus: http://localhost:9091"
    echo ""
}

# 显示部署信息
show_deployment_info() {
    log_success "Deployment completed successfully!"
    echo ""
    echo "=== LocalMoE Deployment Information ==="
    echo "Environment: $ENVIRONMENT"
    echo "Config File: $CONFIG_FILE"
    echo ""
    echo "=== Quick Start ==="
    echo "1. Check service status:"
    echo "   curl http://localhost:8000/health"
    echo ""
    echo "2. Test inference:"
    echo "   curl -X POST http://localhost:8000/v1/inference \\"
    echo "     -H 'Content-Type: application/json' \\"
    echo "     -d '{\"text\": \"Hello\", \"code\": \"print('world')\", \"mode\": \"multimodal\"}'"
    echo ""
    echo "3. View logs:"
    echo "   docker-compose -f docker/docker-compose.yml logs -f localmoe"
    echo ""
    echo "4. Stop services:"
    echo "   docker-compose -f docker/docker-compose.yml down"
    echo ""
}

# 主函数
main() {
    check_dependencies
    create_directories
    generate_config
    build_image
    run_tests
    start_services
    show_deployment_info
}

# 错误处理
trap 'log_error "Deployment failed at line $LINENO"' ERR

# 执行主函数
main
