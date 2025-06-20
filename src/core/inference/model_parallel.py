"""
模型并行管理器 - 基于DesignPlan.md的硬件感知部署方案
支持专家-设备拓扑映射和分层聚合通信
"""

import torch
import torch.distributed as dist
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DeviceTopology:
    """设备拓扑信息"""
    device_id: int
    pcie_bandwidth: float  # GB/s
    memory_gb: float
    compute_capability: Tuple[int, int]
    connected_devices: List[int]
    bandwidth_matrix: List[List[float]]


@dataclass
class ExpertMapping:
    """专家映射信息"""
    expert_id: int
    primary_device: int
    secondary_device: Optional[int] = None
    bandwidth: float = 0.0
    memory_requirement: float = 0.0


class PCIeTopologyDetector:
    """PCIe拓扑检测器"""
    
    def __init__(self):
        self.device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.topology_cache = None
    
    def detect_topology(self) -> Dict[str, Any]:
        """检测PCIe拓扑结构"""
        if self.topology_cache is not None:
            return self.topology_cache
        
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, using mock topology")
            return self._create_mock_topology()
        
        try:
            topology = self._detect_real_topology()
            self.topology_cache = topology
            return topology
        except Exception as e:
            logger.error(f"Failed to detect topology: {e}")
            return self._create_mock_topology()
    
    def _detect_real_topology(self) -> Dict[str, Any]:
        """检测真实的PCIe拓扑"""
        devices = []
        bandwidth_matrix = []
        
        for i in range(self.device_count):
            device_info = self._get_device_info(i)
            devices.append(device_info)
        
        # 构建带宽矩阵
        for i in range(self.device_count):
            row = []
            for j in range(self.device_count):
                if i == j:
                    row.append(0.0)  # 自身
                else:
                    bandwidth = self._measure_bandwidth(i, j)
                    row.append(bandwidth)
            bandwidth_matrix.append(row)
        
        # 检测PCIe组
        pcie_groups = self._detect_pcie_groups(bandwidth_matrix)
        
        return {
            "devices": devices,
            "bandwidth_matrix": bandwidth_matrix,
            "pcie_groups": pcie_groups,
            "best_pairs": self._find_best_pairs(bandwidth_matrix)
        }
    
    def _get_device_info(self, device_id: int) -> DeviceTopology:
        """获取设备信息"""
        props = torch.cuda.get_device_properties(device_id)
        
        # 获取PCIe带宽（简化估算）
        pcie_bandwidth = self._estimate_pcie_bandwidth(device_id)
        
        return DeviceTopology(
            device_id=device_id,
            pcie_bandwidth=pcie_bandwidth,
            memory_gb=props.total_memory / 1024**3,
            compute_capability=(props.major, props.minor),
            connected_devices=[],
            bandwidth_matrix=[]
        )
    
    def _estimate_pcie_bandwidth(self, device_id: int) -> float:
        """估算PCIe带宽"""
        try:
            # 使用nvidia-ml-py获取更详细信息
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            
            # 获取PCIe信息
            pcie_info = pynvml.nvmlDeviceGetPciInfo(handle)
            
            # 根据PCIe版本和宽度估算带宽
            # 这是简化实现，实际应该更精确
            return 14.0  # L40S典型PCIe 4.0 x16带宽
            
        except Exception:
            # Fallback估算
            return 14.0
    
    def _measure_bandwidth(self, device_i: int, device_j: int) -> float:
        """测量设备间带宽"""
        try:
            # 简化的带宽测量
            # 实际应该进行P2P内存传输测试
            
            # 检查P2P访问能力
            can_access = torch.cuda.can_device_access_peer(device_i, device_j)
            
            if can_access:
                # 假设同一PCIe switch下的设备有更高带宽
                if abs(device_i - device_j) == 1:
                    return 14.0  # 高带宽连接
                else:
                    return 8.0   # 跨switch连接
            else:
                return 4.0  # 通过CPU的连接
                
        except Exception:
            return 8.0  # 默认带宽
    
    def _detect_pcie_groups(self, bandwidth_matrix: List[List[float]]) -> List[List[int]]:
        """检测PCIe组"""
        groups = []
        used_devices = set()
        
        for i in range(len(bandwidth_matrix)):
            if i in used_devices:
                continue
            
            group = [i]
            used_devices.add(i)
            
            # 找到与当前设备高带宽连接的设备
            for j in range(i + 1, len(bandwidth_matrix)):
                if j not in used_devices and bandwidth_matrix[i][j] >= 12.0:
                    group.append(j)
                    used_devices.add(j)
            
            groups.append(group)
        
        return groups
    
    def _find_best_pairs(self, bandwidth_matrix: List[List[float]]) -> List[Tuple[Tuple[int, int], float]]:
        """找到最佳设备对"""
        pairs = []
        
        for i in range(len(bandwidth_matrix)):
            for j in range(i + 1, len(bandwidth_matrix)):
                bandwidth = bandwidth_matrix[i][j]
                pairs.append(((i, j), bandwidth))
        
        # 按带宽排序
        pairs.sort(key=lambda x: x[1], reverse=True)
        
        return pairs
    
    def _create_mock_topology(self) -> Dict[str, Any]:
        """创建模拟拓扑（用于测试）"""
        device_count = max(4, self.device_count)
        
        # 模拟4个L40S设备的拓扑
        devices = []
        for i in range(device_count):
            devices.append(DeviceTopology(
                device_id=i,
                pcie_bandwidth=14.0,
                memory_gb=48.0,  # L40S显存
                compute_capability=(8, 9),  # Ada Lovelace
                connected_devices=[],
                bandwidth_matrix=[]
            ))
        
        # 模拟带宽矩阵
        bandwidth_matrix = [
            [0, 14, 8, 8],   # GPU 0
            [14, 0, 8, 8],   # GPU 1
            [8, 8, 0, 14],   # GPU 2
            [8, 8, 14, 0]    # GPU 3
        ]
        
        # 扩展到实际设备数量
        if device_count > 4:
            for i in range(4, device_count):
                for row in bandwidth_matrix:
                    row.append(8.0)
                new_row = [8.0] * device_count
                new_row[i] = 0.0
                bandwidth_matrix.append(new_row)
        
        return {
            "devices": devices,
            "bandwidth_matrix": bandwidth_matrix,
            "pcie_groups": [[0, 1], [2, 3]],
            "best_pairs": [((0, 1), 14.0), ((2, 3), 14.0), ((0, 2), 8.0), ((1, 3), 8.0)]
        }


class ModelParallelManager:
    """
    模型并行管理器
    实现基于DesignPlan.md的专家-设备拓扑映射
    """
    
    def __init__(self, num_experts: int = 8):
        self.num_experts = num_experts
        self.topology_detector = PCIeTopologyDetector()
        self.topology = None
        self.expert_mappings = {}
        
        # 初始化拓扑
        self._initialize_topology()
        
        # 创建专家映射
        self._create_expert_mappings()
        
        logger.info(f"ModelParallelManager initialized with {num_experts} experts")
    
    def _initialize_topology(self):
        """初始化拓扑信息"""
        self.topology = self.topology_detector.detect_topology()
        
        logger.info(f"Detected topology with {len(self.topology['devices'])} devices")
        logger.info(f"PCIe groups: {self.topology['pcie_groups']}")
    
    def _create_expert_mappings(self):
        """创建专家映射"""
        if not self.topology:
            logger.error("Topology not initialized")
            return
        
        devices = self.topology["devices"]
        best_pairs = self.topology["best_pairs"]
        
        # 基于PCIe拓扑的专家分配
        expert_mappings = {}
        
        # 优先分配高带宽设备对
        pair_index = 0
        for expert_id in range(self.num_experts):
            if pair_index < len(best_pairs):
                (primary, secondary), bandwidth = best_pairs[pair_index]
                
                expert_mappings[expert_id] = ExpertMapping(
                    expert_id=expert_id,
                    primary_device=primary,
                    secondary_device=secondary,
                    bandwidth=bandwidth,
                    memory_requirement=4.0  # 4GB per expert (32B / 8 experts)
                )
                
                # 循环使用设备对
                pair_index = (pair_index + 1) % len(best_pairs)
            else:
                # 单设备分配
                device_id = expert_id % len(devices)
                expert_mappings[expert_id] = ExpertMapping(
                    expert_id=expert_id,
                    primary_device=device_id,
                    memory_requirement=4.0
                )
        
        self.expert_mappings = expert_mappings
        
        logger.info(f"Created expert mappings for {len(expert_mappings)} experts")
    
    def get_expert_device(self, expert_id: int) -> int:
        """获取专家的主设备"""
        if expert_id in self.expert_mappings:
            return self.expert_mappings[expert_id].primary_device
        else:
            # Fallback到轮询分配
            device_count = len(self.topology["devices"]) if self.topology else 4
            return expert_id % device_count
    
    def get_expert_mapping(self, expert_id: int) -> Optional[ExpertMapping]:
        """获取专家映射信息"""
        return self.expert_mappings.get(expert_id)
    
    def get_device_experts(self, device_id: int) -> List[int]:
        """获取设备上的专家列表"""
        experts = []
        for expert_id, mapping in self.expert_mappings.items():
            if mapping.primary_device == device_id:
                experts.append(expert_id)
        return experts
    
    def get_communication_plan(self, expert_ids: List[int]) -> Dict[str, Any]:
        """
        获取通信计划
        基于专家分布优化通信路径
        """
        if not expert_ids:
            return {"error": "No experts specified"}
        
        # 按设备分组专家
        device_groups = {}
        for expert_id in expert_ids:
            device = self.get_expert_device(expert_id)
            if device not in device_groups:
                device_groups[device] = []
            device_groups[device].append(expert_id)
        
        # 分析通信模式
        involved_devices = list(device_groups.keys())
        pcie_groups = self.topology["pcie_groups"] if self.topology else [[0, 1], [2, 3]]
        
        # 确定通信层次
        intra_group_comm = []
        inter_group_comm = []
        
        for group in pcie_groups:
            group_devices = [d for d in involved_devices if d in group]
            if len(group_devices) > 1:
                intra_group_comm.append(group_devices)
        
        # 跨组通信
        group_leaders = []
        for group in pcie_groups:
            leaders = [d for d in involved_devices if d in group]
            if leaders:
                group_leaders.append(min(leaders))  # 选择组内最小设备ID作为leader
        
        if len(group_leaders) > 1:
            inter_group_comm = group_leaders
        
        return {
            "device_groups": device_groups,
            "intra_group_communication": intra_group_comm,
            "inter_group_communication": inter_group_comm,
            "communication_cost": self._estimate_communication_cost(device_groups)
        }
    
    def _estimate_communication_cost(self, device_groups: Dict[int, List[int]]) -> float:
        """估算通信开销"""
        if not self.topology:
            return 0.0
        
        total_cost = 0.0
        bandwidth_matrix = self.topology["bandwidth_matrix"]
        
        devices = list(device_groups.keys())
        for i in range(len(devices)):
            for j in range(i + 1, len(devices)):
                device_i, device_j = devices[i], devices[j]
                
                # 数据量估算（基于专家数量）
                data_size = len(device_groups[device_i]) * len(device_groups[device_j]) * 0.1  # GB
                
                # 带宽
                if (device_i < len(bandwidth_matrix) and 
                    device_j < len(bandwidth_matrix[device_i])):
                    bandwidth = bandwidth_matrix[device_i][device_j]
                    if bandwidth > 0:
                        cost = data_size / bandwidth  # 传输时间
                        total_cost += cost
        
        return total_cost
    
    def optimize_expert_placement(self, workload_pattern: Dict[int, float]) -> Dict[int, int]:
        """
        优化专家放置
        基于工作负载模式重新分配专家
        
        Args:
            workload_pattern: {expert_id: usage_frequency}
            
        Returns:
            Dict[int, int]: {expert_id: device_id}
        """
        if not self.topology:
            return {expert_id: expert_id % 4 for expert_id in workload_pattern.keys()}
        
        # 按使用频率排序专家
        sorted_experts = sorted(
            workload_pattern.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # 获取设备信息
        devices = self.topology["devices"]
        device_loads = {i: 0.0 for i in range(len(devices))}
        
        # 重新分配专家
        new_placement = {}
        
        for expert_id, frequency in sorted_experts:
            # 找到负载最轻的设备
            best_device = min(device_loads.keys(), key=lambda d: device_loads[d])
            
            new_placement[expert_id] = best_device
            device_loads[best_device] += frequency
        
        logger.info(f"Optimized placement for {len(new_placement)} experts")
        return new_placement
    
    def get_topology_info(self) -> Dict[str, Any]:
        """获取拓扑信息"""
        if not self.topology:
            return {"error": "Topology not available"}
        
        return {
            "device_count": len(self.topology["devices"]),
            "pcie_groups": self.topology["pcie_groups"],
            "bandwidth_matrix": self.topology["bandwidth_matrix"],
            "best_pairs": self.topology["best_pairs"],
            "expert_mappings": {
                expert_id: {
                    "primary_device": mapping.primary_device,
                    "secondary_device": mapping.secondary_device,
                    "bandwidth": mapping.bandwidth,
                    "memory_requirement": mapping.memory_requirement
                }
                for expert_id, mapping in self.expert_mappings.items()
            }
        }
    
    def validate_placement(self) -> Dict[str, Any]:
        """验证专家放置的有效性"""
        if not self.topology:
            return {"error": "Topology not available"}
        
        devices = self.topology["devices"]
        device_memory_usage = {i: 0.0 for i in range(len(devices))}
        device_expert_counts = {i: 0 for i in range(len(devices))}
        
        # 计算每个设备的内存使用和专家数量
        for expert_id, mapping in self.expert_mappings.items():
            device = mapping.primary_device
            device_memory_usage[device] += mapping.memory_requirement
            device_expert_counts[device] += 1
        
        # 检查内存限制
        memory_violations = []
        for device_id, usage in device_memory_usage.items():
            if device_id < len(devices):
                available_memory = devices[device_id].memory_gb * 0.8  # 80%利用率
                if usage > available_memory:
                    memory_violations.append({
                        "device": device_id,
                        "usage": usage,
                        "available": available_memory
                    })
        
        # 负载均衡检查
        expert_counts = list(device_expert_counts.values())
        load_balance_score = min(expert_counts) / max(expert_counts) if max(expert_counts) > 0 else 1.0
        
        return {
            "memory_violations": memory_violations,
            "device_memory_usage": device_memory_usage,
            "device_expert_counts": device_expert_counts,
            "load_balance_score": load_balance_score,
            "is_valid": len(memory_violations) == 0
        }
