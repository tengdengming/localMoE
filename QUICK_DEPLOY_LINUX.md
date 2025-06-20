# LocalMoE Linux服务器快速部署指南

## 🚀 一键部署 (推荐)

### 前提条件
- Linux服务器 (Ubuntu 20.04+ / CentOS 8+)
- 4x NVIDIA L40S GPU + 376GB RAM + 128 CPU
- NVIDIA驱动已安装
- 网络连接正常

### 部署命令

```bash
# 1. 上传代码到服务器
git clone <your-repo-url> LocalMoE
cd LocalMoE

# 2. 执行一键部署
chmod +x deploy_linux_server.sh
./deploy_linux_server.sh

# 3. 启动服务
./deploy_linux_server.sh start
```

就这么简单！🎉

## 📋 详细步骤

### 步骤1: 准备服务器

```bash
# SSH连接到服务器
ssh user@your-server-ip

# 检查GPU状态
nvidia-smi

# 检查系统资源
free -h
nproc
df -h
```

### 步骤2: 上传代码

```bash
# 方法1: Git克隆 (推荐)
git clone <your-repo-url> LocalMoE
cd LocalMoE

# 方法2: 从本地上传
# 在本地执行:
# rsync -avz --exclude='venv' --exclude='__pycache__' LocalMoE/ user@server-ip:~/LocalMoE/
```

### 步骤3: 执行部署

```bash
# 给脚本执行权限
chmod +x deploy_linux_server.sh

# 运行部署脚本
./deploy_linux_server.sh
```

部署脚本会自动完成：
- ✅ 系统依赖安装
- ✅ Python环境设置
- ✅ PyTorch + CUDA安装
- ✅ vLLM引擎安装
- ✅ 项目依赖安装
- ✅ 环境配置
- ✅ systemd服务创建
- ✅ 安装测试

### 步骤4: 启动服务

```bash
# 方法1: 前台运行 (测试用)
./deploy_linux_server.sh start

# 方法2: 后台服务 (生产用)
sudo systemctl start localmoe
sudo systemctl status localmoe
```

## 🔧 配置选择

### 默认配置 (13B模型)
```yaml
# configs/config.yaml
vllm:
  model_name: "meta-llama/Llama-2-13b-chat-hf"
  quantization: "awq"
  tensor_parallel_size: 4
  gpu_memory_utilization: 0.85
```

### 高性能配置 (7B模型)
```bash
# 使用7B模型配置
cp configs/l40s_quantization_configs.yaml configs/config.yaml
# 编辑选择: small_model_fp16
```

### 大模型配置 (70B模型)
```bash
# 使用70B模型配置
cp configs/l40s_quantization_configs.yaml configs/config.yaml
# 编辑选择: large_model_awq
```

## 🌐 验证部署

### 检查服务状态

```bash
# 检查服务
sudo systemctl status localmoe

# 检查端口
netstat -tlnp | grep 8000

# 检查GPU使用
nvidia-smi
```

### 测试API

```bash
# 健康检查
curl http://localhost:8000/health

# 推理测试
curl -X POST "http://localhost:8000/v1/inference" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "你好，请介绍一下LocalMoE系统",
    "model_config": {
      "max_tokens": 100,
      "temperature": 0.7
    }
  }'
```

### 访问Web界面

- **API文档**: http://server-ip:8000/docs
- **健康检查**: http://server-ip:8000/health
- **模型信息**: http://server-ip:8000/models

## 📊 监控和管理

### 服务管理

```bash
# 启动服务
sudo systemctl start localmoe

# 停止服务
sudo systemctl stop localmoe

# 重启服务
sudo systemctl restart localmoe

# 查看状态
sudo systemctl status localmoe

# 开机自启
sudo systemctl enable localmoe
```

### 日志查看

```bash
# 系统日志
sudo journalctl -u localmoe -f

# 应用日志
tail -f logs/localmoe.log

# 错误日志
tail -f logs/error.log
```

### 性能监控

```bash
# GPU监控
watch -n 1 nvidia-smi

# 系统监控
htop

# 网络监控
netstat -tlnp | grep 8000
```

## 🚨 常见问题

### 1. GPU内存不足

```bash
# 检查GPU使用
nvidia-smi

# 解决方案: 降低内存使用率
# 编辑 configs/config.yaml
# vllm.gpu_memory_utilization: 0.8
```

### 2. 端口被占用

```bash
# 检查端口占用
netstat -tlnp | grep 8000

# 解决方案: 修改端口
# 编辑 .env
# LOCALMOE_PORT=8001
```

### 3. 模型下载失败

```bash
# 手动下载模型
huggingface-cli download meta-llama/Llama-2-13b-chat-hf --local-dir ./models/llama2-13b

# 修改配置使用本地路径
# configs/config.yaml
# vllm.model_name: "./models/llama2-13b"
```

### 4. 依赖安装失败

```bash
# 重新创建环境
rm -rf venv
./deploy_linux_server.sh
```

## 🎯 性能优化

### L40S GPU优化

```yaml
# 推荐配置
vllm:
  quantization: "awq"              # L40S最佳量化
  gpu_memory_utilization: 0.85     # 保守内存使用
  enable_prefix_caching: true      # 利用大显存
  use_v2_block_manager: true       # v2内存管理
  max_num_batched_tokens: 16384    # 大批处理
```

### 系统优化

```bash
# GPU性能模式
sudo nvidia-smi -pm 1

# 系统参数优化
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

## 📞 技术支持

### 获取帮助

```bash
# 查看部署脚本帮助
./deploy_linux_server.sh --help

# 运行测试
./deploy_linux_server.sh test

# 查看配置
curl http://localhost:8000/v1/config
```

### 收集诊断信息

```bash
# 系统信息
uname -a
nvidia-smi
free -h
df -h

# 服务状态
sudo systemctl status localmoe
sudo journalctl -u localmoe --no-pager -n 50

# GPU状态
nvidia-smi --query-gpu=name,memory.used,memory.total,temperature.gpu --format=csv
```

## 🎉 部署完成

恭喜！您已经成功部署了LocalMoE服务。

### 下一步

1. **配置模型**: 根据需求选择合适的模型和量化方案
2. **性能调优**: 根据实际负载调整配置参数
3. **监控设置**: 配置Prometheus和Grafana监控
4. **安全配置**: 设置防火墙和访问控制
5. **备份策略**: 配置模型和配置文件备份

### 重要提醒

- ✅ DeepSpeed已被注释掉，只使用vLLM引擎
- ✅ 配置已针对L40S GPU优化
- ✅ 支持AWQ/FP8量化，充分利用硬件特性
- ✅ 生产环境配置，稳定可靠

享受您的高性能LocalMoE推理服务！🚀
