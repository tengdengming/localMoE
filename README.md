# LocalMoE - 高性能多模态推理服务

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![CUDA 12.1+](https://img.shields.io/badge/CUDA-12.1+-green.svg)](https://developer.nvidia.com/cuda-downloads)

LocalMoE是一个基于vLLM引擎的高性能多模态推理服务，专门针对NVIDIA L40S GPU优化，支持大规模语言模型的高效推理。

## 系统架构

```
LocalMoE/
├── src/                          # 源代码目录
│   ├── core/                     # 核心组件
│   │   ├── moe/                  # MoE相关组件
│   │   ├── multimodal/           # 多模态处理
│   │   ├── inference/            # 推理引擎
│   │   └── routing/              # 路由管理
│   ├── api/                      # API服务
│   ├── config/                   # 配置管理
│   ├── utils/                    # 工具函数
│   └── monitoring/               # 监控组件
├── models/                       # 模型文件
├── configs/                      # 配置文件
├── tests/                        # 测试用例
├── scripts/                      # 部署脚本
├── docker/                       # Docker配置
└── docs/                         # 文档
```

## 硬件配置

- **内存**: 376GB
- **CPU**: 128核
- **GPU**: 4x NVIDIA L40S (48GB VRAM each)
- **部署**: 内网服务器

## 技术栈

- **推理框架**: vLLM (DeepSpeed已注释掉)
- **深度学习**: PyTorch
- **API框架**: FastAPI
- **多模态**: 文本 + 编码
- **架构**: Mixture of Experts (MoE)

## 核心特性

1. **智能专家路由**: 基于输入特征的动态专家选择
2. **多模态融合**: 文本和编码模态的统一处理
3. **分布式推理**: 4GPU并行推理优化
4. **高性能API**: 基于FastAPI的REST服务
5. **资源管理**: GPU资源调度和负载均衡
6. **实时监控**: 性能指标和健康状态监控

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 启动服务
python -m src.api.main

# 健康检查
curl http://localhost:8000/health
```

## API接口

### 推理接口
```bash
POST /v1/inference
{
    "text": "输入文本",
    "code": "输入代码",
    "model_config": {
        "max_tokens": 512,
        "temperature": 0.7
    }
}
```

### 监控接口
```bash
GET /v1/metrics      # 性能指标
GET /v1/health       # 健康状态
GET /v1/experts      # 专家状态
```

## 开发指南

详细的开发文档请参考 [docs/](docs/) 目录。

## 许可证

MIT License
