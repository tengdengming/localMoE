# 远程部署命令集合

## 🚀 一键部署命令

您可以将以下命令复制到服务器终端执行：

### 方案1: 完整自动化部署

```bash
# 创建项目目录并进入
mkdir -p ~/LocalMoE && cd ~/LocalMoE

# 如果您的代码在Git仓库中，克隆代码
# git clone <your-repo-url> .

# 如果您需要手动上传代码，可以使用scp或rsync
# scp -r /path/to/LocalMoE/* user@server:~/LocalMoE/

# 运行自动化部署脚本
chmod +x scripts/deploy_vllm_only.sh
./scripts/deploy_vllm_only.sh

# 启动服务
./scripts/deploy_vllm_only.sh start
```

### 方案2: 分步部署

```bash
# 1. 系统环境检查
nvidia-smi
free -h
python3 --version

# 2. 创建Python虚拟环境
python3 -m venv venv
source venv/bin/activate

# 3. 安装PyTorch (CUDA 12.1)
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. 安装项目依赖
pip install -r requirements.txt

# 5. 安装vLLM引擎
pip install vllm>=0.2.0

# 6. 安装优化库
pip install flash-attn xformers triton pynvml zstandard

# 7. 创建必要目录
mkdir -p logs models checkpoints data

# 8. 设置环境变量
export PYTHONPATH=$(pwd)
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 9. 测试安装
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
python3 -c "import vllm; print('vLLM OK')"

# 10. 启动服务
python -m src.api.main
```

### 方案3: Docker部署

```bash
# 1. 安装Docker和Docker Compose (如果未安装)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# 安装Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# 2. 安装NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# 3. 构建并启动服务
docker-compose -f docker-compose.vllm.yml up -d

# 4. 查看日志
docker-compose -f docker-compose.vllm.yml logs -f
```

## 📋 快速检查命令

```bash
# 检查服务状态
curl http://localhost:8000/health

# 检查GPU使用情况
nvidia-smi

# 查看服务日志
tail -f logs/localmoe.log

# 测试API
curl -X POST "http://localhost:8000/v1/inference" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello", "model_config": {"max_tokens": 50}}'
```

## 🔧 常用管理命令

```bash
# 启动服务 (后台运行)
nohup python -m src.api.main > logs/service.log 2>&1 &

# 查看进程
ps aux | grep python

# 停止服务
pkill -f "python -m src.api.main"

# 重启服务
pkill -f "python -m src.api.main" && sleep 2 && nohup python -m src.api.main > logs/service.log 2>&1 &
```

## 🚨 故障排除命令

```bash
# 检查端口占用
netstat -tlnp | grep 8000

# 检查磁盘空间
df -h

# 检查内存使用
free -h

# 检查GPU内存
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# 清理GPU内存
python3 -c "import torch; torch.cuda.empty_cache()"

# 重新安装依赖
pip install -r requirements.txt --force-reinstall
```

## 📁 文件传输命令

如果您需要将本地代码上传到服务器：

```bash
# 使用scp上传整个项目
scp -r LocalMoE/ user@server-ip:~/

# 使用rsync同步代码 (推荐)
rsync -avz --exclude='venv' --exclude='__pycache__' --exclude='*.pyc' LocalMoE/ user@server-ip:~/LocalMoE/

# 压缩后上传
tar -czf localmoe.tar.gz LocalMoE/
scp localmoe.tar.gz user@server-ip:~/
# 在服务器上解压
ssh user@server-ip "cd ~ && tar -xzf localmoe.tar.gz"
```

## 🔐 SSH连接示例

```bash
# 连接到服务器
ssh user@your-server-ip

# 使用密钥连接
ssh -i /path/to/private-key user@your-server-ip

# 端口转发 (将服务器的8000端口转发到本地)
ssh -L 8000:localhost:8000 user@your-server-ip

# 后台运行并保持连接
ssh user@your-server-ip "cd ~/LocalMoE && nohup ./scripts/deploy_vllm_only.sh start > deployment.log 2>&1 &"
```

## 📊 监控命令

```bash
# 实时监控GPU
watch -n 1 nvidia-smi

# 监控系统资源
htop

# 监控网络连接
watch -n 1 "netstat -tlnp | grep 8000"

# 监控日志
tail -f logs/localmoe.log

# 监控磁盘IO
iostat -x 1
```
