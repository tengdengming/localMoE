"""
专家内存管理器 - 基于DesignPlan.md的动态显存管理实现
支持LRU缓存策略、ZSTD压缩和分层卸载
"""

import torch
import zstandard as zstd
import threading
import time
from typing import Dict, Optional, Any, List
from collections import OrderedDict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExpertState:
    """专家状态信息"""
    expert_id: str
    state_dict: Dict[str, torch.Tensor]
    last_access: float
    device: torch.device
    compressed_size: Optional[int] = None
    is_compressed: bool = False


class ExpertMemoryManager:
    """
    专家内存管理器
    实现动态显存管理、LRU缓存策略和分层卸载
    """
    
    def __init__(
        self,
        max_active_experts: int = 4,
        compression_level: int = 3,
        enable_compression: bool = True,
        memory_threshold: float = 0.85
    ):
        self.max_active_experts = max_active_experts
        self.compression_level = compression_level
        self.enable_compression = enable_compression
        self.memory_threshold = memory_threshold
        
        # 活跃专家池 (GPU)
        self.active_experts: OrderedDict[str, ExpertState] = OrderedDict()
        
        # CPU缓存池
        self.cpu_pool: Dict[str, bytes] = {}
        
        # 压缩器
        self.compressor = zstd.ZstdCompressor(level=compression_level)
        self.decompressor = zstd.ZstdDecompressor()
        
        # 线程锁
        self.lock = threading.RLock()
        
        # 统计信息
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'swaps_out': 0,
            'swaps_in': 0,
            'compression_ratio': 0.0
        }
        
        logger.info(f"ExpertMemoryManager initialized: max_active={max_active_experts}")
    
    def get_memory_info(self) -> Dict[str, float]:
        """获取显存使用信息"""
        if not torch.cuda.is_available():
            return {'allocated': 0, 'reserved': 0, 'free': 0}
            
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        free = total - allocated
        
        return {
            'allocated': allocated,
            'reserved': reserved,
            'free': free,
            'total': total,
            'utilization': allocated / total
        }
    
    def _compress_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> bytes:
        """压缩专家状态字典"""
        if not self.enable_compression:
            return torch.save(state_dict, None)
        
        # 将状态字典序列化为字节
        import io
        buffer = io.BytesIO()
        torch.save(state_dict, buffer)
        raw_data = buffer.getvalue()
        
        # 使用ZSTD压缩
        compressed_data = self.compressor.compress(raw_data)
        
        # 更新压缩比统计
        compression_ratio = len(compressed_data) / len(raw_data)
        self.stats['compression_ratio'] = (
            self.stats['compression_ratio'] * 0.9 + compression_ratio * 0.1
        )
        
        logger.debug(f"Compressed state dict: {len(raw_data)} -> {len(compressed_data)} bytes "
                    f"(ratio: {compression_ratio:.3f})")
        
        return compressed_data
    
    def _decompress_state_dict(self, compressed_data: bytes) -> Dict[str, torch.Tensor]:
        """解压缩专家状态字典"""
        if not self.enable_compression:
            return torch.load(compressed_data)
        
        # 解压缩
        raw_data = self.decompressor.decompress(compressed_data)
        
        # 反序列化
        import io
        buffer = io.BytesIO(raw_data)
        state_dict = torch.load(buffer, map_location='cpu')
        
        return state_dict
    
    def _get_lru_expert(self) -> Optional[str]:
        """获取最近最少使用的专家ID"""
        if not self.active_experts:
            return None
        
        # OrderedDict保持插入顺序，最早的就是LRU
        return next(iter(self.active_experts))
    
    def _check_memory_pressure(self) -> bool:
        """检查内存压力"""
        memory_info = self.get_memory_info()
        return memory_info['utilization'] > self.memory_threshold
    
    def swap_out(self, expert_id: str) -> bool:
        """
        将专家从GPU交换到CPU
        
        Args:
            expert_id: 专家ID
            
        Returns:
            bool: 是否成功交换
        """
        with self.lock:
            if expert_id not in self.active_experts:
                logger.warning(f"Expert {expert_id} not in active pool")
                return False
            
            expert_state = self.active_experts[expert_id]
            
            # 将状态字典移到CPU并压缩
            cpu_state_dict = {
                k: v.cpu() for k, v in expert_state.state_dict.items()
            }
            
            compressed_data = self._compress_state_dict(cpu_state_dict)
            self.cpu_pool[expert_id] = compressed_data
            
            # 从活跃池中移除
            del self.active_experts[expert_id]
            
            # 清理GPU内存
            torch.cuda.empty_cache()
            
            self.stats['swaps_out'] += 1
            
            logger.info(f"Swapped out expert {expert_id} to CPU "
                       f"(compressed size: {len(compressed_data)} bytes)")
            
            return True
    
    def swap_in(self, expert_id: str, device: torch.device) -> bool:
        """
        将专家从CPU交换到GPU
        
        Args:
            expert_id: 专家ID
            device: 目标设备
            
        Returns:
            bool: 是否成功交换
        """
        with self.lock:
            if expert_id not in self.cpu_pool:
                logger.error(f"Expert {expert_id} not found in CPU pool")
                return False
            
            # 从CPU池加载并解压缩
            compressed_data = self.cpu_pool[expert_id]
            state_dict = self._decompress_state_dict(compressed_data)
            
            # 移动到GPU
            gpu_state_dict = {
                k: v.to(device) for k, v in state_dict.items()
            }
            
            # 创建专家状态
            expert_state = ExpertState(
                expert_id=expert_id,
                state_dict=gpu_state_dict,
                last_access=time.time(),
                device=device
            )
            
            # 添加到活跃池
            self.active_experts[expert_id] = expert_state
            
            # 从CPU池移除
            del self.cpu_pool[expert_id]
            
            self.stats['swaps_in'] += 1
            
            logger.info(f"Swapped in expert {expert_id} to {device}")
            
            return True
    
    def allocate(self, expert_id: str, device: Optional[torch.device] = None) -> bool:
        """
        分配专家到GPU
        
        Args:
            expert_id: 专家ID
            device: 目标设备，默认为cuda:0
            
        Returns:
            bool: 是否成功分配
        """
        if device is None:
            device = torch.device('cuda:0')
        
        with self.lock:
            # 如果已经在活跃池中，更新访问时间
            if expert_id in self.active_experts:
                expert_state = self.active_experts[expert_id]
                expert_state.last_access = time.time()
                # 移到末尾（最近使用）
                self.active_experts.move_to_end(expert_id)
                self.stats['cache_hits'] += 1
                return True
            
            self.stats['cache_misses'] += 1
            
            # 检查是否需要释放空间
            while (len(self.active_experts) >= self.max_active_experts or 
                   self._check_memory_pressure()):
                lru_expert = self._get_lru_expert()
                if lru_expert is None:
                    break
                if not self.swap_out(lru_expert):
                    logger.error(f"Failed to swap out LRU expert {lru_expert}")
                    break
            
            # 从CPU池加载专家
            if expert_id in self.cpu_pool:
                return self.swap_in(expert_id, device)
            else:
                logger.error(f"Expert {expert_id} not found in any pool")
                return False
    
    def get_expert_state(self, expert_id: str) -> Optional[ExpertState]:
        """获取专家状态"""
        with self.lock:
            return self.active_experts.get(expert_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        memory_info = self.get_memory_info()
        
        return {
            **self.stats,
            'active_experts': len(self.active_experts),
            'cpu_pool_size': len(self.cpu_pool),
            'memory_info': memory_info,
            'cache_hit_rate': (
                self.stats['cache_hits'] / 
                max(1, self.stats['cache_hits'] + self.stats['cache_misses'])
            )
        }
    
    def clear_cache(self):
        """清空所有缓存"""
        with self.lock:
            self.active_experts.clear()
            self.cpu_pool.clear()
            torch.cuda.empty_cache()
            logger.info("Cleared all expert caches")
