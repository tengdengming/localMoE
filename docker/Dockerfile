# LocalMoE - 多模态MoE推理服务 Docker镜像
# 基于NVIDIA PyTorch容器，支持L40S GPU

FROM nvcr.io/nvidia/pytorch:23.10-py3

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0,1,2,3
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    htop \
    nvtop \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 升级pip
RUN pip install --upgrade pip

# 复制requirements文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 安装vLLM (DeepSpeed已注释掉)
# RUN pip install deepspeed>=0.12.0  # DeepSpeed已注释掉
RUN pip install vllm>=0.2.0

# 安装其他推理优化库
RUN pip install \
    flash-attn \
    xformers \
    triton \
    pynvml \
    zstandard

# 复制源代码
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

# 创建必要的目录
RUN mkdir -p /app/logs /app/models /app/cache

# 设置权限
RUN chmod +x scripts/*.sh

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 暴露端口
EXPOSE 8000 9090

# 启动命令
CMD ["python", "-m", "src.api.main"]
