"""
MoE层实现 - 整合专家路由、内存管理和分布式计算
基于DesignPlan.md的完整MoE架构
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Dict, List, Optional, Any, Tuple
import asyncio
import logging
from dataclasses import dataclass

from .expert_router import ExpertGate, HierarchicalRouter, RoutingResult, ExpertCapacity
from .expert_module import ExpertPool, ExpertConfig, QuantizationType
from .memory_manager import ExpertMemoryManager

logger = logging.getLogger(__name__)


@dataclass
class MoEConfig:
    """MoE层配置"""
    hidden_size: int
    num_experts: int
    top_k: int = 2
    capacity_factor: float = 1.25
    gate_type: str = "linear"
    expert_hidden_size: Optional[int] = None
    expert_intermediate_size: Optional[int] = None
    activation: str = "swiglu"
    dropout: float = 0.1
    quantization: Optional[QuantizationType] = None
    load_balance_weight: float = 0.01
    enable_expert_parallelism: bool = True
    max_active_experts: int = 4


class MoELayer(nn.Module):
    """
    Mixture of Experts层
    实现完整的MoE计算流程
    """
    
    def __init__(
        self,
        config: MoEConfig,
        device_mapping: Optional[Dict[int, int]] = None,
        pcie_topology: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.load_balance_weight = config.load_balance_weight
        
        # 设备映射 (expert_id -> device_id)
        if device_mapping is None:
            # 默认均匀分布到4个GPU
            device_mapping = {i: i % 4 for i in range(config.num_experts)}
        self.device_mapping = device_mapping
        
        # 专家门控网络
        self.gate = ExpertGate(
            hidden_size=config.hidden_size,
            num_experts=config.num_experts,
            top_k=config.top_k,
            gate_type=config.gate_type
        )
        
        # 创建专家配置
        expert_configs = []
        for expert_id in range(config.num_experts):
            device_id = device_mapping[expert_id]
            expert_config = ExpertConfig(
                expert_id=expert_id,
                hidden_size=config.hidden_size,
                intermediate_size=config.expert_intermediate_size or config.hidden_size * 4,
                activation=config.activation,
                dropout=config.dropout,
                quantization=config.quantization,
                device=torch.device(f"cuda:{device_id}")
            )
            expert_configs.append(expert_config)
        
        # 专家池
        self.expert_pool = ExpertPool(expert_configs)
        
        # 内存管理器
        self.memory_manager = ExpertMemoryManager(
            max_active_experts=config.max_active_experts
        )
        
        # 分层路由器
        if pcie_topology is None:
            pcie_topology = {
                "pcie_groups": [[0, 1], [2, 3]],
                "bandwidth_matrix": [[0, 14, 8, 8], [14, 0, 8, 8], [8, 8, 0, 14], [8, 8, 14, 0]]
            }
        
        self.hierarchical_router = HierarchicalRouter(
            pcie_topology=pcie_topology,
            num_experts=config.num_experts,
            device_mapping=device_mapping
        )
        
        # 专家容量管理
        self.expert_capacities = self._init_expert_capacities()
        
        logger.info(f"MoELayer initialized with {config.num_experts} experts, top_k={config.top_k}")
    
    def _init_expert_capacities(self) -> Dict[int, ExpertCapacity]:
        """初始化专家容量"""
        capacities = {}
        base_capacity = 1024  # 基础容量
        
        for expert_id in range(self.num_experts):
            device_id = self.device_mapping[expert_id]
            capacities[expert_id] = ExpertCapacity(
                max_tokens=base_capacity,
                current_load=0,
                device_id=device_id,
                bandwidth=14.0,  # GB/s，L40S的PCIe带宽
                latency=0.1      # ms
            )
        
        return capacities
    
    def _reset_expert_loads(self):
        """重置专家负载"""
        for capacity in self.expert_capacities.values():
            capacity.current_load = 0
    
    def _prepare_expert_inputs(
        self,
        hidden_states: torch.Tensor,
        routing_result: RoutingResult
    ) -> Dict[int, torch.Tensor]:
        """
        准备专家输入数据
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            routing_result: 路由结果
            
        Returns:
            Dict[int, torch.Tensor]: 每个专家的输入数据
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        flat_hidden_states = hidden_states.view(-1, hidden_size)
        
        expert_inputs = {}
        
        for expert_id, token_indices in routing_result.expert_assignments.items():
            if token_indices:
                # 提取分配给该专家的tokens
                indices_tensor = torch.tensor(
                    [idx[0] * seq_len + idx[1] for idx in token_indices],
                    device=hidden_states.device,
                    dtype=torch.long
                )
                expert_input = flat_hidden_states[indices_tensor]
                expert_inputs[expert_id] = expert_input
        
        return expert_inputs
    
    def _aggregate_expert_outputs(
        self,
        expert_outputs: Dict[int, torch.Tensor],
        routing_result: RoutingResult,
        original_shape: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        聚合专家输出
        
        Args:
            expert_outputs: 专家输出
            routing_result: 路由结果
            original_shape: 原始形状
            
        Returns:
            torch.Tensor: 聚合后的输出
        """
        batch_size, seq_len, hidden_size = original_shape
        device = next(iter(expert_outputs.values())).device
        
        # 初始化输出张量
        aggregated_output = torch.zeros(
            batch_size * seq_len, hidden_size,
            device=device,
            dtype=next(iter(expert_outputs.values())).dtype
        )
        
        # 聚合专家输出
        for expert_id, expert_output in expert_outputs.items():
            if expert_id in routing_result.expert_assignments:
                token_indices = routing_result.expert_assignments[expert_id]
                if token_indices and expert_output.size(0) > 0:
                    # 获取该专家对应的权重
                    expert_weights = self._get_expert_weights(
                        expert_id, token_indices, routing_result
                    )
                    
                    # 加权聚合
                    weighted_output = expert_output * expert_weights.unsqueeze(-1)
                    
                    # 累加到对应位置
                    flat_indices = torch.tensor(
                        [idx[0] * seq_len + idx[1] for idx in token_indices],
                        device=device,
                        dtype=torch.long
                    )
                    aggregated_output.index_add_(0, flat_indices, weighted_output)
        
        return aggregated_output.view(batch_size, seq_len, hidden_size)
    
    def _get_expert_weights(
        self,
        expert_id: int,
        token_indices: List[List[int]],
        routing_result: RoutingResult
    ) -> torch.Tensor:
        """获取专家权重"""
        weights = []
        for batch_idx, seq_idx in token_indices:
            # 从路由结果中获取权重
            expert_weights = routing_result.expert_weights[batch_idx, seq_idx]
            top_k_indices = (routing_result.expert_weights[batch_idx, seq_idx] > 0).nonzero(as_tuple=False)
            
            # 找到当前专家的权重
            expert_weight = 0.0
            for k_idx in top_k_indices:
                if k_idx.item() < expert_weights.size(0):
                    expert_weight = expert_weights[k_idx.item()].item()
                    break
            
            weights.append(expert_weight)
        
        return torch.tensor(weights, device=routing_result.expert_weights.device)
    
    async def _forward_expert_async(
        self,
        expert_id: int,
        expert_input: torch.Tensor
    ) -> Tuple[int, torch.Tensor]:
        """异步专家前向传播"""
        # 确保专家在内存中
        device = torch.device(f"cuda:{self.device_mapping[expert_id]}")
        if not self.memory_manager.allocate(str(expert_id), device):
            logger.error(f"Failed to allocate expert {expert_id}")
            return expert_id, torch.zeros_like(expert_input)
        
        # 执行前向传播
        expert_output = self.expert_pool.forward_expert(
            expert_id, expert_input, use_compilation=True
        )
        
        if expert_output is None:
            expert_output = torch.zeros_like(expert_input)
        
        return expert_id, expert_output
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        MoE层前向传播
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: 注意力掩码
            
        Returns:
            Tuple[torch.Tensor, Dict]: (输出, 统计信息)
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        original_device = hidden_states.device
        
        # 重置专家负载
        self._reset_expert_loads()
        
        # 1. 专家路由
        routing_result = self.gate(hidden_states, self.expert_capacities)
        
        # 2. 优化专家分配
        optimized_assignments = self.hierarchical_router.get_optimal_expert_assignment(
            routing_result, self.expert_capacities
        )
        routing_result.expert_assignments = optimized_assignments
        
        # 3. 准备专家输入
        expert_inputs = self._prepare_expert_inputs(hidden_states, routing_result)
        
        # 4. 分布式专家计算
        if self.config.enable_expert_parallelism and len(expert_inputs) > 1:
            # 异步并行计算
            expert_outputs = self._forward_experts_parallel(expert_inputs)
        else:
            # 串行计算
            expert_outputs = self._forward_experts_sequential(expert_inputs)
        
        # 5. 聚合输出
        output = self._aggregate_expert_outputs(
            expert_outputs, routing_result, (batch_size, seq_len, hidden_size)
        )
        
        # 确保输出在原始设备上
        if output.device != original_device:
            output = output.to(original_device)
        
        # 6. 计算总损失
        total_loss = routing_result.load_balance_loss * self.load_balance_weight
        
        # 7. 收集统计信息
        stats = {
            "routing_loss": routing_result.load_balance_loss.item(),
            "total_loss": total_loss.item(),
            "active_experts": len(expert_outputs),
            "expert_utilization": {
                expert_id: len(tokens) for expert_id, tokens in routing_result.expert_assignments.items()
            },
            "memory_stats": self.memory_manager.get_stats(),
            "expert_stats": self.expert_pool.get_all_stats()
        }
        
        return output, stats
    
    def _forward_experts_parallel(self, expert_inputs: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        """并行专家计算"""
        # 简化实现，实际应该使用异步计算
        expert_outputs = {}
        for expert_id, expert_input in expert_inputs.items():
            expert_output = self.expert_pool.forward_expert(expert_id, expert_input)
            if expert_output is not None:
                expert_outputs[expert_id] = expert_output
        return expert_outputs
    
    def _forward_experts_sequential(self, expert_inputs: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        """串行专家计算"""
        expert_outputs = {}
        for expert_id, expert_input in expert_inputs.items():
            expert_output = self.expert_pool.forward_expert(expert_id, expert_input)
            if expert_output is not None:
                expert_outputs[expert_id] = expert_output
        return expert_outputs
    
    def get_expert_load_stats(self) -> Dict[str, Any]:
        """获取专家负载统计"""
        return {
            "expert_capacities": {
                expert_id: {
                    "max_tokens": cap.max_tokens,
                    "current_load": cap.current_load,
                    "utilization": cap.current_load / cap.max_tokens,
                    "device_id": cap.device_id
                }
                for expert_id, cap in self.expert_capacities.items()
            },
            "memory_manager_stats": self.memory_manager.get_stats(),
            "expert_pool_stats": self.expert_pool.get_all_stats()
        }
