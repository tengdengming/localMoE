# vLLM MoE模型部署环境扫描工具

这个工具集用于全面检查服务器环境是否满足vLLM单机多卡MoE模型部署的要求。

## 文件说明

- `environment_scan.py` - 详细的Python扫描脚本（推荐）
- `environment_scan.sh` - 快速的Bash扫描脚本
- `requirements_scan.txt` - Python脚本依赖包
- `README_scan.md` - 本说明文档

## 使用方法

### 方法1：Python详细扫描（推荐）

```bash
# 1. 安装依赖
pip install -r scripts/requirements_scan.txt

# 2. 运行扫描（建议以root用户运行）
sudo python3 scripts/environment_scan.py

# 或者在WSL中
wsl -u root
python3 scripts/environment_scan.py
```

### 方法2：Bash快速扫描

```bash
# 给脚本执行权限
chmod +x scripts/environment_scan.sh

# 运行扫描
sudo ./scripts/environment_scan.sh

# 或者在WSL中
wsl -u root
./scripts/environment_scan.sh
```

## 检查项目

### 系统信息
- 操作系统版本和架构
- 内核版本
- 主机名和用户信息
- Python版本

### 硬件配置
- CPU型号、核心数和支持的指令集
- 内存容量和使用情况
- 交换分区配置

### GPU环境
- NVIDIA驱动版本
- CUDA版本和安装状态
- GPU型号、数量和显存
- GPU温度和功耗状态

### Python环境
- Python版本和路径
- 虚拟环境状态
- pip版本
- site-packages路径

### 关键依赖包
- torch (PyTorch)
- vllm
- transformers
- fastapi
- deepspeed
- flash-attn
- xformers
- triton
- 其他相关包

### 网络配置
- 网络接口信息
- DNS配置
- 互联网连接状态
- HuggingFace连接状态

### 存储空间
- 磁盘分区和使用情况
- 可用空间检查
- 临时目录空间

## 输出结果

### Python脚本输出
- 控制台彩色格式化报告
- JSON格式详细结果文件 (`environment_scan_results.json`)
- 针对性的建议和警告

### Bash脚本输出
- 控制台彩色格式化快速报告
- 关键问题提示

## 建议阈值

### 最低要求
- 内存：32GB+
- GPU：至少1张支持CUDA的GPU
- 显存：总计24GB+
- 存储：100GB+可用空间
- CUDA：11.8+

### 推荐配置
- 内存：64GB+
- GPU：2张以上L40S或类似级别
- 显存：总计48GB+
- 存储：500GB+可用空间
- 网络：稳定的互联网连接

## 常见问题

### Q: 为什么建议以root用户运行？
A: root权限可以获取更完整的系统信息，包括硬件详情、系统配置等。

### Q: 扫描脚本是否会修改系统？
A: 不会。扫描脚本只读取系统信息，不会进行任何修改操作。

### Q: 如何解决依赖包缺失问题？
A: 根据扫描结果中的建议，使用pip或conda安装缺失的包：
```bash
pip install torch vllm transformers fastapi deepspeed
```

### Q: GPU检测失败怎么办？
A: 检查NVIDIA驱动是否正确安装：
```bash
nvidia-smi
nvcc --version
```

## 针对L40S GPU的特殊说明

如果您使用的是L40S GPU（如您的服务器配置），扫描脚本会特别关注：
- L40S的48GB显存配置
- 多卡NVLink连接状态
- 适合的CUDA版本兼容性
- 量化优化建议

## 下一步

扫描完成后，根据报告中的建议：
1. 安装缺失的依赖包
2. 配置GPU环境
3. 优化系统设置
4. 准备模型部署环境

扫描通过后，即可开始vLLM MoE模型的部署工作。
