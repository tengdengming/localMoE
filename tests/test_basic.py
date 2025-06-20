"""
基础测试 - 测试LocalMoE核心功能
"""

import pytest
import torch
import requests
import time
import json
from typing import Dict, Any

# 测试配置
API_BASE_URL = "http://localhost:8000"
TEST_TIMEOUT = 30


class TestBasicFunctionality:
    """基础功能测试"""
    
    def test_health_check(self):
        """测试健康检查"""
        response = requests.get(f"{API_BASE_URL}/health", timeout=TEST_TIMEOUT)
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] in ["healthy", "unhealthy"]
        assert "uptime_seconds" in data
    
    def test_root_endpoint(self):
        """测试根端点"""
        response = requests.get(f"{API_BASE_URL}/", timeout=TEST_TIMEOUT)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "LocalMoE" in data["message"]
    
    def test_readiness_check(self):
        """测试就绪检查"""
        response = requests.get(f"{API_BASE_URL}/ready", timeout=TEST_TIMEOUT)
        assert response.status_code in [200, 503]  # 可能还未就绪


class TestInferenceAPI:
    """推理API测试"""
    
    def test_text_only_inference(self):
        """测试纯文本推理"""
        payload = {
            "text": "Hello, how are you?",
            "mode": "text_only",
            "sampling_params": {
                "temperature": 0.8,
                "max_tokens": 50
            }
        }
        
        response = requests.post(
            f"{API_BASE_URL}/v1/inference",
            json=payload,
            timeout=TEST_TIMEOUT
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "result" in data
        assert "generated_text" in data["result"]
    
    def test_code_only_inference(self):
        """测试纯代码推理"""
        payload = {
            "code": "def hello():\n    print('Hello, World!')",
            "mode": "code_only",
            "sampling_params": {
                "temperature": 0.5,
                "max_tokens": 100
            }
        }
        
        response = requests.post(
            f"{API_BASE_URL}/v1/inference",
            json=payload,
            timeout=TEST_TIMEOUT
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "result" in data
    
    def test_multimodal_inference(self):
        """测试多模态推理"""
        payload = {
            "text": "Explain this code:",
            "code": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
            "mode": "multimodal",
            "sampling_params": {
                "temperature": 0.7,
                "max_tokens": 200
            }
        }
        
        response = requests.post(
            f"{API_BASE_URL}/v1/inference",
            json=payload,
            timeout=TEST_TIMEOUT
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "result" in data
        assert data["result"]["inference_time_ms"] > 0
    
    def test_batch_inference(self):
        """测试批量推理"""
        requests_data = [
            {
                "text": "Hello",
                "code": "print('world')",
                "mode": "multimodal"
            },
            {
                "text": "Goodbye",
                "code": "print('farewell')",
                "mode": "multimodal"
            }
        ]
        
        payload = {
            "requests": requests_data,
            "batch_size": 2
        }
        
        response = requests.post(
            f"{API_BASE_URL}/v1/inference/batch",
            json=payload,
            timeout=TEST_TIMEOUT * 2
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert len(data["results"]) == 2
        assert data["total_time_ms"] > 0
    
    def test_invalid_request(self):
        """测试无效请求"""
        payload = {
            "mode": "invalid_mode"
        }
        
        response = requests.post(
            f"{API_BASE_URL}/v1/inference",
            json=payload,
            timeout=TEST_TIMEOUT
        )
        
        assert response.status_code == 422  # Validation error


class TestMonitoringAPI:
    """监控API测试"""
    
    def test_system_metrics(self):
        """测试系统指标"""
        response = requests.get(f"{API_BASE_URL}/v1/metrics", timeout=TEST_TIMEOUT)
        assert response.status_code == 200
        
        data = response.json()
        assert "gpu_metrics" in data
        assert "memory_metrics" in data
        assert "inference_metrics" in data
    
    def test_expert_status(self):
        """测试专家状态"""
        response = requests.get(f"{API_BASE_URL}/v1/experts", timeout=TEST_TIMEOUT)
        assert response.status_code == 200
        
        data = response.json()
        assert "experts" in data
        assert "total_experts" in data
        assert "active_experts" in data
    
    def test_performance_stats(self):
        """测试性能统计"""
        response = requests.get(f"{API_BASE_URL}/v1/performance", timeout=TEST_TIMEOUT)
        assert response.status_code == 200
        
        data = response.json()
        assert "total_requests" in data or "error" in data


class TestManagementAPI:
    """管理API测试"""
    
    def test_get_config(self):
        """测试获取配置"""
        response = requests.get(f"{API_BASE_URL}/v1/config", timeout=TEST_TIMEOUT)
        assert response.status_code == 200
        
        data = response.json()
        assert "inference" in data
        assert "model" in data
        assert "system" in data
    
    def test_system_info(self):
        """测试系统信息"""
        response = requests.get(f"{API_BASE_URL}/v1/system/info", timeout=TEST_TIMEOUT)
        assert response.status_code == 200
        
        data = response.json()
        assert "platform" in data
        assert "pytorch" in data
        assert "service" in data


class TestCoreComponents:
    """核心组件测试"""
    
    def test_torch_cuda_availability(self):
        """测试CUDA可用性"""
        assert torch.cuda.is_available(), "CUDA should be available"
        assert torch.cuda.device_count() > 0, "At least one GPU should be available"
    
    def test_model_loading(self):
        """测试模型加载"""
        try:
            from src.core.multimodal import FeatureExtractor, FeatureExtractorConfig
            
            config = FeatureExtractorConfig()
            extractor = FeatureExtractor(config)
            
            assert extractor is not None
            assert hasattr(extractor, 'text_processor')
            assert hasattr(extractor, 'code_processor')
            
        except ImportError:
            pytest.skip("Core components not available")
    
    def test_gpu_manager(self):
        """测试GPU管理器"""
        try:
            from src.core.routing import GPUManager, GPUConfig
            
            configs = [GPUConfig(device_id=i) for i in range(min(4, torch.cuda.device_count()))]
            manager = GPUManager(configs)
            
            assert manager is not None
            assert len(manager.gpu_configs) > 0
            
            # 测试专家分配
            device_id = manager.allocate_expert(0, memory_requirement=1.0)
            assert device_id is not None
            
            # 测试释放
            success = manager.deallocate_expert(0)
            assert success is True
            
            manager.cleanup()
            
        except ImportError:
            pytest.skip("GPU manager not available")


class TestErrorHandling:
    """错误处理测试"""
    
    def test_404_endpoint(self):
        """测试404错误"""
        response = requests.get(f"{API_BASE_URL}/nonexistent", timeout=TEST_TIMEOUT)
        assert response.status_code == 404
    
    def test_malformed_json(self):
        """测试格式错误的JSON"""
        response = requests.post(
            f"{API_BASE_URL}/v1/inference",
            data="invalid json",
            headers={"Content-Type": "application/json"},
            timeout=TEST_TIMEOUT
        )
        assert response.status_code == 422
    
    def test_large_request(self):
        """测试过大的请求"""
        large_text = "x" * (10 * 1024 * 1024)  # 10MB
        payload = {
            "text": large_text,
            "mode": "text_only"
        }
        
        response = requests.post(
            f"{API_BASE_URL}/v1/inference",
            json=payload,
            timeout=TEST_TIMEOUT
        )
        
        # 应该被限制或处理
        assert response.status_code in [413, 422, 500]


class TestPerformance:
    """性能测试"""
    
    def test_concurrent_requests(self):
        """测试并发请求"""
        import concurrent.futures
        import threading
        
        def make_request():
            payload = {
                "text": "Test concurrent request",
                "mode": "text_only",
                "sampling_params": {"max_tokens": 10}
            }
            
            response = requests.post(
                f"{API_BASE_URL}/v1/inference",
                json=payload,
                timeout=TEST_TIMEOUT
            )
            return response.status_code == 200
        
        # 发送5个并发请求
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(5)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # 至少一半的请求应该成功
        success_rate = sum(results) / len(results)
        assert success_rate >= 0.5, f"Success rate too low: {success_rate}"
    
    def test_response_time(self):
        """测试响应时间"""
        payload = {
            "text": "Quick test",
            "mode": "text_only",
            "sampling_params": {"max_tokens": 5}
        }
        
        start_time = time.time()
        response = requests.post(
            f"{API_BASE_URL}/v1/inference",
            json=payload,
            timeout=TEST_TIMEOUT
        )
        end_time = time.time()
        
        response_time = end_time - start_time
        
        assert response.status_code == 200
        assert response_time < 10.0, f"Response time too slow: {response_time}s"


if __name__ == "__main__":
    # 运行基础测试
    pytest.main([__file__, "-v"])
