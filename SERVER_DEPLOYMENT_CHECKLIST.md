# LocalMoE Linux服务器部署检查清单

## 🎯 部署前准备

### 硬件要求确认
- [ ] **GPU**: 4x NVIDIA L40S (48GB VRAM each)
- [ ] **内存**: 376GB RAM
- [ ] **CPU**: 128核心
- [ ] **存储**: 2TB+ NVMe SSD
- [ ] **网络**: 千兆以上内网连接

### 软件环境确认
- [ ] **操作系统**: Ubuntu 20.04+ 或 CentOS 8+
- [ ] **NVIDIA驱动**: 已安装且版本 >= 525.xx
- [ ] **CUDA**: 12.1+ (可选，PyTorch会自带)
- [ ] **Python**: 3.8+ (建议3.10)
- [ ] **Git**: 已安装
- [ ] **网络**: 可访问PyPI和HuggingFace

## 🚀 快速部署步骤

### 1. 上传代码到服务器

```bash
# 方法1: Git克隆 (推荐)
git clone <your-repo-url> LocalMoE
cd LocalMoE

# 方法2: scp上传
scp -r LocalMoE/ user@server-ip:~/

# 方法3: rsync同步
rsync -avz --exclude='venv' --exclude='__pycache__' LocalMoE/ user@server-ip:~/LocalMoE/
```

### 2. 执行一键部署

```bash
cd LocalMoE
chmod +x deploy_linux_server.sh
./deploy_linux_server.sh
```

### 3. 启动服务

```bash
# 方法1: 前台运行 (测试用)
./deploy_linux_server.sh start

# 方法2: 后台服务 (生产用)
sudo systemctl start localmoe
sudo systemctl status localmoe
```

## 📋 详细部署步骤

### 步骤1: 环境检查

```bash
# 检查GPU
nvidia-smi

# 检查内存
free -h

# 检查磁盘
df -h

# 检查CPU
nproc
lscpu
```

### 步骤2: 系统依赖安装

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y build-essential python3-dev python3-pip python3-venv git wget curl htop nvtop tmux vim

# CentOS/RHEL
sudo yum update -y
sudo yum groupinstall -y "Development Tools"
sudo yum install -y python3-devel python3-pip git wget curl htop tmux vim
```

### 步骤3: Python环境设置

```bash
# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 升级pip
pip install --upgrade pip

# 安装PyTorch (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 步骤4: 项目依赖安装

```bash
# 安装基础依赖
pip install -r requirements.txt

# 安装vLLM (跳过DeepSpeed)
pip install vllm>=0.2.0

# 安装优化库
pip install flash-attn xformers triton pynvml zstandard transformers accelerate
```

### 步骤5: 配置环境

```bash
# 创建目录
mkdir -p logs models checkpoints data

# 设置环境变量
export PYTHONPATH=$(pwd)
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

### 步骤6: 测试安装

```bash
# 测试CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# 测试vLLM
python -c "import vllm; print('vLLM OK')"

# 测试项目模块
python -c "from src.config.settings import load_settings; print('Config OK')"
```

## 🔧 配置优化

### L40S GPU优化配置

编辑 `configs/config.yaml`:

```yaml
# L40S优化配置
vllm:
  model_name: "meta-llama/Llama-2-13b-chat-hf"  # 或其他模型
  quantization: "awq"                            # L40S推荐量化
  tensor_parallel_size: 4                        # 4张GPU并行
  gpu_memory_utilization: 0.85                   # L40S保守内存使用
  max_model_len: 4096                           # 适中序列长度
  max_num_batched_tokens: 16384                 # 大批处理
  enable_prefix_caching: true                   # 利用大显存
  use_v2_block_manager: true                    # v2内存管理器
```

### 系统优化

```bash
# GPU性能模式
sudo nvidia-smi -pm 1

# 设置最大时钟频率
sudo nvidia-smi -ac 6001,1410

# 优化系统参数
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
echo 'net.core.rmem_max=134217728' | sudo tee -a /etc/sysctl.conf
echo 'net.core.wmem_max=134217728' | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

## 🌐 服务访问

### API端点

- **主页**: http://server-ip:8000/
- **API文档**: http://server-ip:8000/docs
- **健康检查**: http://server-ip:8000/health
- **模型列表**: http://server-ip:8000/models
- **推理接口**: http://server-ip:8000/v1/inference

### 测试API

```bash
# 健康检查
curl http://localhost:8000/health

# 推理测试
curl -X POST "http://localhost:8000/v1/inference" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, how are you?",
    "model_config": {
      "max_tokens": 100,
      "temperature": 0.7
    }
  }'
```

## 📊 监控和管理

### 服务管理

```bash
# systemd服务管理
sudo systemctl start localmoe      # 启动
sudo systemctl stop localmoe       # 停止
sudo systemctl restart localmoe    # 重启
sudo systemctl status localmoe     # 状态
sudo systemctl enable localmoe     # 开机自启

# 查看日志
sudo journalctl -u localmoe -f     # 实时日志
sudo journalctl -u localmoe -n 100 # 最近100行
tail -f logs/localmoe.log          # 应用日志
```

### 性能监控

```bash
# GPU监控
watch -n 1 nvidia-smi
nvtop

# 系统监控
htop
iotop

# 网络监控
netstat -tlnp | grep 8000
ss -tlnp | grep 8000
```

### 日志管理

```bash
# 日志轮转配置
sudo tee /etc/logrotate.d/localmoe << EOF
/path/to/LocalMoE/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    copytruncate
}
EOF
```

## 🚨 故障排除

### 常见问题

1. **GPU内存不足**
   ```bash
   # 检查GPU使用情况
   nvidia-smi
   
   # 调整配置
   # configs/config.yaml -> vllm.gpu_memory_utilization: 0.8
   ```

2. **端口被占用**
   ```bash
   # 检查端口
   netstat -tlnp | grep 8000
   
   # 修改端口
   # .env -> LOCALMOE_PORT=8001
   ```

3. **依赖冲突**
   ```bash
   # 重新创建环境
   rm -rf venv
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

4. **模型下载失败**
   ```bash
   # 手动下载模型
   huggingface-cli download microsoft/DialoGPT-medium --local-dir ./models/DialoGPT-medium
   
   # 配置本地路径
   # configs/config.yaml -> vllm.model_name: "./models/DialoGPT-medium"
   ```

### 性能调优

1. **内存优化**
   - 调整 `gpu_memory_utilization`
   - 启用 `enable_prefix_caching`
   - 使用 `use_v2_block_manager`

2. **并发优化**
   - 调整 `max_num_batched_tokens`
   - 设置合适的 `max_num_seqs`
   - 优化 `tensor_parallel_size`

3. **量化优化**
   - L40S推荐使用 `awq` 或 `fp8`
   - 大模型必须使用量化
   - 小模型可以不量化

## ✅ 部署完成检查

- [ ] 服务正常启动
- [ ] API接口响应正常
- [ ] GPU正常识别和使用
- [ ] 内存使用合理
- [ ] 日志正常输出
- [ ] systemd服务配置正确
- [ ] 防火墙端口开放
- [ ] 性能监控正常

## 📞 技术支持

如果遇到问题，请检查：

1. **系统日志**: `sudo journalctl -u localmoe -f`
2. **应用日志**: `tail -f logs/localmoe.log`
3. **GPU状态**: `nvidia-smi`
4. **系统资源**: `htop`
5. **网络连接**: `netstat -tlnp | grep 8000`

部署成功后，您将拥有一个高性能的LocalMoE推理服务，充分利用L40S GPU的硬件优势！
