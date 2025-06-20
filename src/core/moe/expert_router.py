"""
专家路由器 - 实现智能专家选择和负载均衡
基于DesignPlan.md的分层聚合通信协议
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class RoutingStrategy(Enum):
    """路由策略枚举"""
    TOP_K = "top_k"
    THRESHOLD = "threshold"
    LOAD_BALANCED = "load_balanced"
    ADAPTIVE = "adaptive"


@dataclass
class ExpertCapacity:
    """专家容量配置"""
    max_tokens: int
    current_load: int
    device_id: int
    bandwidth: float
    latency: float


@dataclass
class RoutingResult:
    """路由结果"""
    expert_ids: List[int]
    expert_weights: torch.Tensor
    routing_probs: torch.Tensor
    load_balance_loss: torch.Tensor
    expert_assignments: Dict[int, List[int]]  # expert_id -> token_indices


class ExpertGate(nn.Module):
    """
    专家门控网络
    实现可学习的专家选择机制
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int = 2,
        gate_type: str = "linear",
        noise_epsilon: float = 1e-2,
        capacity_factor: float = 1.25
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate_type = gate_type
        self.noise_epsilon = noise_epsilon
        self.capacity_factor = capacity_factor
        
        # 门控网络
        if gate_type == "linear":
            self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        elif gate_type == "mlp":
            self.gate = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, num_experts)
            )
        else:
            raise ValueError(f"Unsupported gate type: {gate_type}")
        
        # 初始化权重
        self._init_weights()
        
        # 负载均衡相关
        self.register_buffer("expert_counts", torch.zeros(num_experts))
        self.register_buffer("expert_weights_sum", torch.zeros(num_experts))
    
    def _init_weights(self):
        """初始化门控网络权重"""
        if self.gate_type == "linear":
            nn.init.normal_(self.gate.weight, mean=0.0, std=0.02)
        else:
            for module in self.gate.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
    
    def _add_noise(self, logits: torch.Tensor) -> torch.Tensor:
        """添加噪声以提高探索性"""
        if self.training and self.noise_epsilon > 0:
            noise = torch.randn_like(logits) * self.noise_epsilon
            return logits + noise
        return logits
    
    def _compute_load_balance_loss(
        self, 
        routing_weights: torch.Tensor, 
        expert_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        计算负载均衡损失
        
        Args:
            routing_weights: [batch_size, seq_len, top_k]
            expert_indices: [batch_size, seq_len, top_k]
        """
        batch_size, seq_len, top_k = routing_weights.shape
        
        # 计算每个专家的权重总和
        expert_weights = torch.zeros(
            self.num_experts, 
            device=routing_weights.device,
            dtype=routing_weights.dtype
        )
        
        for i in range(top_k):
            expert_weights.scatter_add_(
                0, 
                expert_indices[:, :, i].flatten(),
                routing_weights[:, :, i].flatten()
            )
        
        # 计算每个专家被选择的次数
        expert_counts = torch.zeros_like(expert_weights)
        for i in range(top_k):
            expert_counts.scatter_add_(
                0,
                expert_indices[:, :, i].flatten(),
                torch.ones_like(expert_indices[:, :, i].flatten(), dtype=expert_weights.dtype)
            )
        
        # 负载均衡损失：鼓励均匀分布
        total_tokens = batch_size * seq_len * top_k
        expected_load = total_tokens / self.num_experts
        
        load_balance_loss = torch.sum(expert_weights * expert_counts) / (expected_load ** 2)
        
        # 更新统计信息
        if self.training:
            self.expert_counts += expert_counts.detach()
            self.expert_weights_sum += expert_weights.detach()
        
        return load_balance_loss
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        expert_capacities: Optional[Dict[int, ExpertCapacity]] = None
    ) -> RoutingResult:
        """
        前向传播
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            expert_capacities: 专家容量信息
            
        Returns:
            RoutingResult: 路由结果
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # 计算门控logits
        gate_logits = self.gate(hidden_states)  # [batch_size, seq_len, num_experts]
        gate_logits = self._add_noise(gate_logits)
        
        # 应用softmax获取概率
        routing_probs = torch.softmax(gate_logits, dim=-1)
        
        # Top-K选择
        top_k_weights, top_k_indices = torch.topk(
            routing_probs, 
            k=min(self.top_k, self.num_experts), 
            dim=-1
        )
        
        # 重新归一化权重
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        # 计算负载均衡损失
        load_balance_loss = self._compute_load_balance_loss(top_k_weights, top_k_indices)
        
        # 生成专家分配
        expert_assignments = {}
        for expert_id in range(self.num_experts):
            # 找到分配给该专家的token位置
            mask = (top_k_indices == expert_id).any(dim=-1)  # [batch_size, seq_len]
            token_indices = torch.nonzero(mask, as_tuple=False).tolist()
            expert_assignments[expert_id] = token_indices
        
        return RoutingResult(
            expert_ids=top_k_indices.flatten().unique().tolist(),
            expert_weights=top_k_weights,
            routing_probs=routing_probs,
            load_balance_loss=load_balance_loss,
            expert_assignments=expert_assignments
        )


class HierarchicalRouter:
    """
    分层路由器
    实现基于PCIe拓扑的分层聚合通信
    """
    
    def __init__(
        self,
        pcie_topology: Dict[str, Any],
        num_experts: int,
        device_mapping: Dict[int, int]  # expert_id -> device_id
    ):
        self.pcie_topology = pcie_topology
        self.num_experts = num_experts
        self.device_mapping = device_mapping
        
        # 构建通信组
        self._build_communication_groups()
        
        logger.info(f"HierarchicalRouter initialized with {num_experts} experts")
    
    def _build_communication_groups(self):
        """构建分层通信组"""
        # 节点内PCIe组 (如 [[0,1], [2,3]])
        self.pcie_groups = self.pcie_topology.get("pcie_groups", [[0, 1], [2, 3]])
        
        # 节点间组 (节点领导者)
        self.node_leaders = [group[0] for group in self.pcie_groups]
        
        # 创建分布式进程组
        if dist.is_initialized():
            self.intra_node_groups = []
            for group in self.pcie_groups:
                if len(group) > 1:
                    pg = dist.new_group(ranks=group)
                    self.intra_node_groups.append(pg)
            
            if len(self.node_leaders) > 1:
                self.inter_node_group = dist.new_group(ranks=self.node_leaders)
            else:
                self.inter_node_group = None
    
    def hierarchical_all_to_all(
        self, 
        inputs: torch.Tensor,
        expert_assignments: Dict[int, List[int]]
    ) -> Dict[int, torch.Tensor]:
        """
        分层All-to-All通信
        
        Args:
            inputs: 输入张量
            expert_assignments: 专家分配
            
        Returns:
            Dict[int, torch.Tensor]: 每个专家的输入数据
        """
        if not dist.is_initialized():
            # 单机模式，直接返回本地分配
            return self._local_assignment(inputs, expert_assignments)
        
        rank = dist.get_rank()
        results = {}
        
        # 第一层：节点内PCIe通信
        intra_node_results = {}
        for i, group in enumerate(self.pcie_groups):
            if rank in group:
                # 在组内进行all-to-all
                group_inputs = self._prepare_group_inputs(inputs, expert_assignments, group)
                if len(group) > 1 and i < len(self.intra_node_groups):
                    group_outputs = self._all_to_all_single_group(
                        group_inputs, self.intra_node_groups[i]
                    )
                else:
                    group_outputs = group_inputs
                intra_node_results.update(group_outputs)
        
        # 第二层：节点间InfiniBand通信
        if self.inter_node_group is not None and rank in self.node_leaders:
            inter_node_inputs = self._prepare_inter_node_inputs(intra_node_results)
            inter_node_outputs = self._all_to_all_single_group(
                inter_node_inputs, self.inter_node_group
            )
            results.update(inter_node_outputs)
        else:
            results.update(intra_node_results)
        
        return results
    
    def _local_assignment(
        self, 
        inputs: torch.Tensor, 
        expert_assignments: Dict[int, List[int]]
    ) -> Dict[int, torch.Tensor]:
        """本地专家分配"""
        results = {}
        for expert_id, token_indices in expert_assignments.items():
            if token_indices:
                # 提取分配给该专家的tokens
                indices_tensor = torch.tensor(token_indices, device=inputs.device)
                expert_inputs = inputs.view(-1, inputs.size(-1))[indices_tensor]
                results[expert_id] = expert_inputs
        return results
    
    def _prepare_group_inputs(
        self, 
        inputs: torch.Tensor, 
        expert_assignments: Dict[int, List[int]], 
        group: List[int]
    ) -> Dict[int, torch.Tensor]:
        """准备组内输入数据"""
        group_inputs = {}
        for expert_id, token_indices in expert_assignments.items():
            if self.device_mapping.get(expert_id) in group and token_indices:
                indices_tensor = torch.tensor(token_indices, device=inputs.device)
                expert_inputs = inputs.view(-1, inputs.size(-1))[indices_tensor]
                group_inputs[expert_id] = expert_inputs
        return group_inputs
    
    def _prepare_inter_node_inputs(
        self, 
        intra_node_results: Dict[int, torch.Tensor]
    ) -> Dict[int, torch.Tensor]:
        """准备节点间输入数据"""
        # 这里可以实现更复杂的节点间数据准备逻辑
        return intra_node_results
    
    def _all_to_all_single_group(
        self, 
        inputs: Dict[int, torch.Tensor], 
        process_group
    ) -> Dict[int, torch.Tensor]:
        """单个组的all-to-all通信"""
        # 简化实现，实际应该使用torch.distributed.all_to_all
        # 这里返回输入作为占位符
        return inputs
    
    def get_optimal_expert_assignment(
        self, 
        routing_result: RoutingResult,
        expert_capacities: Dict[int, ExpertCapacity]
    ) -> Dict[int, List[int]]:
        """
        获取最优专家分配
        考虑负载均衡和通信开销
        """
        optimized_assignments = {}
        
        for expert_id, token_indices in routing_result.expert_assignments.items():
            if expert_id in expert_capacities:
                capacity = expert_capacities[expert_id]
                
                # 检查容量限制
                if capacity.current_load + len(token_indices) <= capacity.max_tokens:
                    optimized_assignments[expert_id] = token_indices
                    capacity.current_load += len(token_indices)
                else:
                    # 容量不足，需要重新分配
                    available_tokens = capacity.max_tokens - capacity.current_load
                    if available_tokens > 0:
                        optimized_assignments[expert_id] = token_indices[:available_tokens]
                        capacity.current_load = capacity.max_tokens
                        
                        # 剩余tokens需要分配给其他专家
                        remaining_tokens = token_indices[available_tokens:]
                        self._reassign_tokens(remaining_tokens, expert_capacities, optimized_assignments)
            else:
                # 专家不可用，重新分配
                self._reassign_tokens(token_indices, expert_capacities, optimized_assignments)
        
        return optimized_assignments
    
    def _reassign_tokens(
        self, 
        tokens: List[int], 
        expert_capacities: Dict[int, ExpertCapacity],
        current_assignments: Dict[int, List[int]]
    ):
        """重新分配tokens到可用专家"""
        for token in tokens:
            # 找到负载最轻的可用专家
            best_expert = None
            min_load_ratio = float('inf')
            
            for expert_id, capacity in expert_capacities.items():
                load_ratio = capacity.current_load / capacity.max_tokens
                if load_ratio < min_load_ratio and capacity.current_load < capacity.max_tokens:
                    best_expert = expert_id
                    min_load_ratio = load_ratio
            
            if best_expert is not None:
                if best_expert not in current_assignments:
                    current_assignments[best_expert] = []
                current_assignments[best_expert].append(token)
                expert_capacities[best_expert].current_load += 1
