#!/bin/bash

# HuggingFace连接修复脚本

echo "🌐 修复HuggingFace连接"
echo "===================="

# 检查DNS解析
echo "检查DNS解析..."
nslookup huggingface.co
echo ""

# 测试不同的连接方式
echo "测试连接方式..."
echo "1. 直接连接:"
curl -I https://huggingface.co --connect-timeout 10 --max-time 30

echo ""
echo "2. 使用IPv4:"
curl -4 -I https://huggingface.co --connect-timeout 10 --max-time 30

echo ""
echo "3. 测试模型下载端点:"
curl -I https://huggingface.co/microsoft/DialoGPT-medium/resolve/main/config.json --connect-timeout 10 --max-time 30

# 提供解决方案
echo ""
echo "🛠️ 解决方案:"
echo ""
echo "方案1: 配置镜像源"
echo "export HF_ENDPOINT=https://hf-mirror.com"
echo "# 或者"
echo "export HF_ENDPOINT=https://huggingface.co"
echo ""
echo "方案2: 配置代理 (如果需要)"
echo "export HTTP_PROXY=http://your-proxy:port"
echo "export HTTPS_PROXY=http://your-proxy:port"
echo ""
echo "方案3: 离线模式"
echo "export HF_HUB_OFFLINE=1"
echo "# 需要预先下载模型到本地"
echo ""
echo "方案4: 使用本地模型缓存"
echo "export HF_HOME=/data/huggingface_cache"
echo "mkdir -p /data/huggingface_cache"
echo ""
echo "测试HuggingFace连接:"
echo "python -c \"from transformers import AutoTokenizer; print('连接正常')\" 2>/dev/null || echo '连接异常'"
