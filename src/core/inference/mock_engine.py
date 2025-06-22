"""
模拟推理引擎 - 用于快速验证功能
"""

import asyncio
import time
import random
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime

from ...api.models import (
    InferenceRequest, InferenceResponse, InferenceResult,
    ExpertInfo, InferenceMode, SystemMetrics, ExpertStatus
)


class MockInferenceEngine:
    """模拟推理引擎"""
    
    def __init__(self):
        self.model_name = "LocalMoE-Mock-Engine"
        self.version = "1.0.0-mock"
        self.start_time = time.time()
        self.request_count = 0
        self.total_tokens = 0
        
        # 模拟专家信息
        self.experts = [
            ExpertInfo(expert_id=i, device_id=i % 4, load=0.0, memory_usage=0.0, active=True)
            for i in range(8)
        ]
        
        # 模拟响应模板
        self.response_templates = {
            InferenceMode.TEXT_ONLY: [
                "基于您的文本输入 '{input}', 我理解您想要了解相关信息。这是LocalMoE系统的文本处理响应。",
                "您提到了 '{input}', 这是一个有趣的话题。LocalMoE的文本专家为您提供以下见解...",
                "关于 '{input}' 的问题，文本处理专家认为这涉及到多个方面的考虑。",
            ],
            InferenceMode.CODE_ONLY: [
                "# 基于您的代码输入，LocalMoE代码专家为您提供以下分析：\n# {input}\n# 这段代码的功能是...",
                "```python\n# 代码分析结果\n# 输入: {input}\n# LocalMoE代码专家建议...\n```",
                "// 代码审查结果\n// 原始代码: {input}\n// 建议优化...",
            ],
            InferenceMode.MULTIMODAL: [
                "多模态分析结果：\n文本部分: {text}\n代码部分: {code}\nLocalMoE融合了文本和代码专家的见解...",
                "综合分析：\n- 文本理解: {text}\n- 代码分析: {code}\n- 融合建议: ...",
            ],
            InferenceMode.AUTO: [
                "自动模式分析：LocalMoE智能路由选择了最适合的专家组合来处理您的请求 '{input}'。",
                "智能路由结果：根据输入内容特征，系统选择了合适的处理策略...",
            ]
        }
    
    async def initialize(self) -> bool:
        """初始化引擎"""
        print("🚀 初始化模拟推理引擎...")
        await asyncio.sleep(1)  # 模拟初始化时间
        print("✅ 模拟推理引擎初始化完成")
        return True
    
    async def inference(self, request: InferenceRequest) -> InferenceResponse:
        """执行推理"""
        start_time = time.time()
        self.request_count += 1
        
        # 模拟推理延迟
        await asyncio.sleep(random.uniform(0.5, 2.0))
        
        # 生成响应
        generated_text = self._generate_response(request)
        
        # 模拟token计数
        input_tokens = len((request.text or "") + (request.code or "")) // 4
        output_tokens = len(generated_text) // 4
        self.total_tokens += input_tokens + output_tokens
        
        # 模拟专家使用
        experts_used = self._select_experts(request.mode)
        
        # 更新专家负载
        for expert_id in experts_used:
            if expert_id < len(self.experts):
                self.experts[expert_id].load = random.uniform(0.3, 0.8)
        
        inference_time = time.time() - start_time
        
        # 创建推理结果
        result = InferenceResult(
            generated_text=generated_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            inference_time_ms=inference_time * 1000,
            experts_used=experts_used,
            model_info={
                "model_name": self.model_name,
                "version": self.version,
                "mode": request.mode.value,
                "engine": "mock"
            }
        )
        
        return InferenceResponse(
            success=True,
            request_id=request.request_id or str(uuid.uuid4()),
            result=result,
            error=None,
            timestamp=time.time()
        )
    
    def _generate_response(self, request: InferenceRequest) -> str:
        """生成模拟响应"""
        mode = request.mode
        templates = self.response_templates.get(mode, self.response_templates[InferenceMode.AUTO])
        template = random.choice(templates)
        
        if mode == InferenceMode.MULTIMODAL:
            return template.format(
                text=request.text or "无文本输入",
                code=request.code or "无代码输入"
            )
        else:
            input_text = request.text or request.code or "无输入"
            return template.format(input=input_text[:100])
    
    def _select_experts(self, mode: InferenceMode) -> List[int]:
        """选择专家"""
        if mode == InferenceMode.TEXT_ONLY:
            return [0, 1]  # 文本专家
        elif mode == InferenceMode.CODE_ONLY:
            return [2, 3]  # 代码专家
        elif mode == InferenceMode.MULTIMODAL:
            return [0, 1, 2, 3]  # 多模态专家
        else:  # AUTO
            return random.sample(range(8), k=random.randint(2, 4))
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        uptime = time.time() - self.start_time
        
        return {
            "engine": {
                "name": self.model_name,
                "version": self.version,
                "type": "mock",
                "uptime_seconds": uptime,
                "status": "running"
            },
            "requests": {
                "total": self.request_count,
                "rate_per_minute": self.request_count / (uptime / 60) if uptime > 0 else 0
            },
            "tokens": {
                "total": self.total_tokens,
                "rate_per_second": self.total_tokens / uptime if uptime > 0 else 0
            },
            "experts": {
                "total": len(self.experts),
                "active": sum(1 for e in self.experts if e.active),
                "average_load": sum(e.load for e in self.experts) / len(self.experts)
            },
            "performance": {
                "average_latency_ms": random.uniform(800, 1500),
                "throughput_tokens_per_second": random.uniform(50, 150)
            }
        }
    
    def get_models(self) -> List[Dict[str, Any]]:
        """获取模型列表"""
        return [
            {
                "name": self.model_name,
                "version": self.version,
                "type": "mock",
                "status": "loaded",
                "parameters": {
                    "experts": 8,
                    "max_tokens": 2048,
                    "context_length": 4096
                },
                "capabilities": ["text", "code", "multimodal"],
                "loaded_at": datetime.fromtimestamp(self.start_time).isoformat()
            }
        ]
    
    def get_expert_status(self) -> List[ExpertStatus]:
        """获取专家状态"""
        return [
            ExpertStatus(
                expert_id=expert.expert_id,
                device_id=expert.device_id,
                status="active" if expert.active else "inactive",
                load=expert.load,
                memory_usage_gb=random.uniform(2.0, 8.0),
                request_count=random.randint(10, 100),
                avg_latency_ms=random.uniform(500, 1200)
            )
            for expert in self.experts
        ]
    
    async def shutdown(self):
        """关闭引擎"""
        print("🛑 关闭模拟推理引擎...")
        await asyncio.sleep(0.5)
        print("✅ 模拟推理引擎已关闭")


# 全局实例
mock_engine = MockInferenceEngine()
