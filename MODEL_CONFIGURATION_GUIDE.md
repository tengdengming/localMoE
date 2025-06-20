# LocalMoE 模型配置指南

## 🎯 概述

LocalMoE项目中的"专家模型"配置分为两个层面：

1. **MoE架构层面**: 8个专家网络的分配和路由 (已注释掉DeepSpeed部分)
2. **vLLM推理层面**: 实际的预训练模型加载和推理

## 🏗️ 当前架构状态

### DeepSpeed MoE (已注释掉)
- ❌ **专家分片**: 8个专家分布到4张GPU
- ❌ **动态路由**: Top-K专家选择
- ❌ **内存管理**: LRU专家交换

### vLLM推理引擎 (当前使用)
- ✅ **模型并行**: 4张GPU张量并行
- ✅ **批处理**: 动态批处理优化
- ✅ **内存优化**: PagedAttention

## 📍 模型配置位置

### 1. 主配置文件: `configs/config.yaml`

```yaml
# 模型配置
model:
  num_experts: 8              # MoE专家数量 (已注释掉)
  top_k_experts: 2           # 激活专家数量 (已注释掉)
  hidden_size: 768           # 隐藏层大小
  intermediate_size: 3072    # 中间层大小
  max_sequence_length: 2048  # 最大序列长度
  quantization_type: "fp16"  # 量化类型
  enable_compilation: true   # 启用编译优化

# vLLM配置 (实际使用的模型)
vllm:
  model_name: "microsoft/DialoGPT-medium"  # 🔥 实际模型路径
  tensor_parallel_size: 4                   # 4张GPU并行
  gpu_memory_utilization: 0.9              # GPU内存使用率
  max_model_len: 2048                      # 最大模型长度
  block_size: 16                           # 内存块大小
  swap_space: 4                            # 交换空间(GB)
  max_num_batched_tokens: 8192             # 最大批处理token数
  enable_prefix_caching: true              # 启用前缀缓存
  quantization: null                       # 量化方式
```

### 2. 代码中的配置: `src/core/inference/vllm_engine.py`

```python
@dataclass
class VLLMConfig:
    # 🔥 关键配置项
    model_name: str = "microsoft/DialoGPT-medium"  # 模型路径
    tensor_parallel_size: int = 4                   # GPU并行数
    gpu_memory_utilization: float = 0.9            # 内存使用率
    max_model_len: int = 2048                      # 最大长度
```

## 🔧 硬件匹配分析

### 当前硬件配置
- **GPU**: 4x NVIDIA L40S (48GB VRAM each)
- **内存**: 376GB RAM
- **CPU**: 128核心

### 配置匹配度

#### ✅ 匹配良好的配置
```yaml
vllm:
  tensor_parallel_size: 4        # 完美匹配4张GPU
  gpu_memory_utilization: 0.9    # 合理利用48GB显存
  max_num_batched_tokens: 8192   # 适合L40S的计算能力
  swap_space: 4                  # 合理的交换空间
```

#### ⚠️ 需要调整的配置
```yaml
vllm:
  model_name: "microsoft/DialoGPT-medium"  # 较小模型，未充分利用硬件
  max_model_len: 2048                      # 可以增加到4096或8192
```

## 🚀 推荐的硬件优化配置

### 1. 大模型配置 (充分利用L40S)

```yaml
# configs/config.yaml
vllm:
  model_name: "meta-llama/Llama-2-70b-chat-hf"  # 70B大模型
  tensor_parallel_size: 4                        # 4张GPU并行
  gpu_memory_utilization: 0.85                  # 稍微保守的内存使用
  max_model_len: 4096                           # 增加序列长度
  max_num_batched_tokens: 16384                 # 增加批处理能力
  block_size: 32                                # 增加块大小
  swap_space: 8                                 # 增加交换空间
```

### 2. 中等模型配置 (平衡性能和资源)

```yaml
vllm:
  model_name: "mistralai/Mistral-7B-Instruct-v0.2"  # 7B模型
  tensor_parallel_size: 2                            # 2张GPU并行
  gpu_memory_utilization: 0.9                       # 高内存使用率
  max_model_len: 8192                               # 长序列支持
  max_num_batched_tokens: 32768                     # 大批处理
```

### 3. 多模型并行配置

```yaml
# 可以同时运行多个模型实例
vllm_instances:
  - model_name: "codellama/CodeLlama-13b-Instruct-hf"
    tensor_parallel_size: 2
    gpu_devices: [0, 1]
  - model_name: "mistralai/Mistral-7B-Instruct-v0.2"
    tensor_parallel_size: 2
    gpu_devices: [2, 3]
```

## 📁 模型文件存储

### 1. 本地模型路径

```bash
# 模型存储目录
./models/
├── microsoft/
│   └── DialoGPT-medium/
├── meta-llama/
│   └── Llama-2-70b-chat-hf/
├── mistralai/
│   └── Mistral-7B-Instruct-v0.2/
└── codellama/
    └── CodeLlama-13b-Instruct-hf/
```

### 2. 配置本地模型

```yaml
# 使用本地模型路径
vllm:
  model_name: "./models/meta-llama/Llama-2-70b-chat-hf"
  trust_remote_code: false
  load_format: "auto"
```

## 🔄 模型切换和管理

### 1. 运行时模型切换

```python
# 通过API切换模型
curl -X POST "http://localhost:8000/v1/models/switch" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "meta-llama/Llama-2-70b-chat-hf",
    "tensor_parallel_size": 4
  }'
```

### 2. 多模型服务

```python
# 启动多个模型实例
python -m src.api.main --config configs/multi_model_config.yaml
```

## 🎛️ 性能调优建议

### 1. GPU内存优化

```yaml
vllm:
  gpu_memory_utilization: 0.85  # L40S建议值
  enable_prefix_caching: true   # 启用缓存
  use_v2_block_manager: true    # 使用v2内存管理器
```

### 2. 批处理优化

```yaml
vllm:
  max_num_batched_tokens: 16384  # 根据GPU数量调整
  max_num_seqs: 256             # 最大序列数
  enable_chunked_prefill: true   # 启用分块预填充
```

### 3. 量化配置

```yaml
vllm:
  quantization: "awq"           # AWQ量化 (推荐)
  # quantization: "gptq"        # GPTQ量化
  # quantization: "squeezellm"  # SqueezeLLM量化
```

## 🔍 监控和诊断

### 1. 检查当前配置

```bash
# 查看当前模型信息
curl http://localhost:8000/v1/models

# 查看GPU使用情况
nvidia-smi

# 查看内存统计
curl http://localhost:8000/v1/stats/memory
```

### 2. 性能监控

```bash
# 查看推理性能
curl http://localhost:8000/v1/stats/performance

# 实时GPU监控
watch -n 1 nvidia-smi
```

## 🚨 常见问题和解决方案

### 1. 模型加载失败

```bash
# 检查模型路径
ls -la ./models/microsoft/DialoGPT-medium/

# 检查权限
chmod -R 755 ./models/

# 重新下载模型
python -c "from transformers import AutoModel; AutoModel.from_pretrained('microsoft/DialoGPT-medium')"
```

### 2. GPU内存不足

```yaml
# 降低内存使用率
vllm:
  gpu_memory_utilization: 0.7
  tensor_parallel_size: 4
  swap_space: 8
```

### 3. 性能不佳

```yaml
# 优化配置
vllm:
  enable_prefix_caching: true
  use_v2_block_manager: true
  enable_chunked_prefill: true
  max_chunked_prefill_tokens: 1024
```

## 📝 总结

当前项目配置：
- ✅ **vLLM引擎**: 使用`microsoft/DialoGPT-medium`模型
- ❌ **MoE专家**: DeepSpeed部分已注释掉
- 🔧 **硬件匹配**: 4张L40S GPU，配置合理但可优化
- 📈 **优化空间**: 可使用更大模型充分利用硬件资源
