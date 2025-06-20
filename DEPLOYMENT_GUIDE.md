# LocalMoE 部署指南 (仅vLLM引擎)

> 注意：DeepSpeed已被注释掉，本指南仅适用于vLLM引擎部署

## 🎯 快速部署

### 方案1: 自动化脚本部署 (推荐)

```bash
# 1. 克隆项目到服务器
git clone <your-repo-url> LocalMoE
cd LocalMoE

# 2. 运行自动化部署脚本
chmod +x scripts/deploy_vllm_only.sh
./scripts/deploy_vllm_only.sh

# 3. 启动服务
./scripts/deploy_vllm_only.sh start
```

### 方案2: Docker部署

```bash
# 1. 构建并启动服务
docker-compose -f docker-compose.vllm.yml up -d

# 2. 查看日志
docker-compose -f docker-compose.vllm.yml logs -f localmoe-vllm

# 3. 停止服务
docker-compose -f docker-compose.vllm.yml down
```

### 方案3: 手动部署

```bash
# 1. 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 2. 安装依赖
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install vllm>=0.2.0

# 3. 设置环境变量
export PYTHONPATH=$(pwd)
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 4. 启动服务
python -m src.api.main
```

## 🔧 系统要求

### 硬件要求
- **GPU**: 4x NVIDIA L40S (48GB VRAM each)
- **内存**: 376GB RAM
- **CPU**: 128核心
- **存储**: 2TB+ NVMe SSD

### 软件要求
- **操作系统**: Ubuntu 20.04+ / CentOS 8+
- **Python**: 3.8+
- **CUDA**: 12.1+
- **Docker**: 20.10+ (可选)
- **Docker Compose**: 2.0+ (可选)

## 📋 部署前检查

```bash
# 检查GPU
nvidia-smi

# 检查内存
free -h

# 检查CUDA版本
nvcc --version

# 检查Python版本
python3 --version
```

## 🚀 服务访问

部署完成后，您可以通过以下地址访问服务：

- **API服务**: http://localhost:8000
- **API文档**: http://localhost:8000/docs
- **健康检查**: http://localhost:8000/health
- **监控面板**: http://localhost:3000 (如果启用了Grafana)

## 🧪 测试部署

```bash
# 健康检查
curl http://localhost:8000/health

# 测试推理接口
curl -X POST "http://localhost:8000/v1/inference" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, world!",
    "model_config": {
      "max_tokens": 100,
      "temperature": 0.7
    }
  }'
```

## 🔍 故障排除

### 常见问题

1. **GPU内存不足**
   ```bash
   # 检查GPU使用情况
   nvidia-smi
   
   # 调整配置文件中的gpu_memory_utilization
   # configs/config.yaml -> vllm.gpu_memory_utilization: 0.8
   ```

2. **端口被占用**
   ```bash
   # 检查端口占用
   netstat -tlnp | grep 8000
   
   # 修改端口配置
   # configs/config.yaml -> api.port: 8001
   ```

3. **依赖安装失败**
   ```bash
   # 更新pip
   pip install --upgrade pip
   
   # 清理缓存
   pip cache purge
   
   # 重新安装
   pip install -r requirements.txt --force-reinstall
   ```

### 日志查看

```bash
# 查看应用日志
tail -f logs/localmoe.log

# 查看错误日志
tail -f logs/error.log

# Docker日志
docker-compose -f docker-compose.vllm.yml logs -f
```

## ⚙️ 配置调优

### 性能优化

1. **GPU内存优化**
   ```yaml
   # configs/config.yaml
   vllm:
     gpu_memory_utilization: 0.9  # 调整GPU内存使用率
     max_num_batched_tokens: 8192  # 调整批处理大小
   ```

2. **并发优化**
   ```yaml
   # configs/config.yaml
   inference:
     max_concurrent_requests: 100  # 调整最大并发数
     enable_batching: true
     max_batch_size: 32
   ```

3. **系统优化**
   ```bash
   # 设置GPU性能模式
   sudo nvidia-smi -pm 1
   
   # 设置最大时钟频率
   sudo nvidia-smi -ac 6001,1410
   ```

## 🔄 服务管理

### 启动/停止服务

```bash
# 启动服务
./scripts/deploy_vllm_only.sh start

# 停止服务 (Ctrl+C)

# 后台运行
nohup ./scripts/deploy_vllm_only.sh start > logs/service.log 2>&1 &
```

### 服务监控

```bash
# 查看GPU使用情况
watch -n 1 nvidia-smi

# 查看系统资源
htop

# 查看网络连接
netstat -tlnp | grep 8000
```

## 📞 技术支持

如果遇到部署问题，请检查：

1. 系统要求是否满足
2. 依赖是否正确安装
3. 配置文件是否正确
4. 日志文件中的错误信息

## 🔄 重新启用DeepSpeed

如需重新启用DeepSpeed支持：

1. 取消`src/core/inference/deepspeed_engine.py`中的代码注释
2. 取消相关配置文件中的DeepSpeed配置注释
3. 安装DeepSpeed: `pip install deepspeed>=0.12.0`
4. 重新部署服务
