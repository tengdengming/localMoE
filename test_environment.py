#!/usr/bin/env python3
"""
LocalMoE环境测试脚本
测试项目结构、导入和基础功能
"""

import sys
import os
import importlib.util
from pathlib import Path

def test_python_version():
    """测试Python版本"""
    print("=== Python环境测试 ===")
    print(f"Python版本: {sys.version}")
    print(f"Python路径: {sys.executable}")
    
    # 检查Python版本是否满足要求
    if sys.version_info >= (3, 8):
        print("✅ Python版本满足要求 (>= 3.8)")
        return True
    else:
        print("❌ Python版本过低，需要 >= 3.8")
        return False

def test_project_structure():
    """测试项目结构"""
    print("\n=== 项目结构测试 ===")
    
    required_dirs = [
        "src",
        "src/core", 
        "src/api",
        "src/config",
        "src/monitoring",
        "tests",
        "docker",
        "scripts",
        "configs"
    ]
    
    required_files = [
        "requirements.txt",
        "src/__init__.py",
        "src/core/__init__.py",
        "src/api/main.py",
        "configs/config.yaml"
    ]
    
    all_passed = True
    
    # 检查目录
    for dir_path in required_dirs:
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            print(f"✅ 目录存在: {dir_path}")
        else:
            print(f"❌ 目录缺失: {dir_path}")
            all_passed = False
    
    # 检查文件
    for file_path in required_files:
        if os.path.exists(file_path) and os.path.isfile(file_path):
            print(f"✅ 文件存在: {file_path}")
        else:
            print(f"❌ 文件缺失: {file_path}")
            all_passed = False
    
    return all_passed

def test_basic_imports():
    """测试基础导入"""
    print("\n=== 基础导入测试 ===")
    
    # 添加项目路径到sys.path
    project_root = os.path.abspath(".")
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    import_tests = [
        ("json", "JSON处理"),
        ("yaml", "YAML处理"),
        ("pathlib", "路径处理"),
        ("dataclasses", "数据类"),
        ("typing", "类型注解"),
        ("enum", "枚举类型"),
        ("threading", "线程支持"),
        ("asyncio", "异步支持"),
        ("logging", "日志系统"),
        ("time", "时间处理")
    ]
    
    all_passed = True
    
    for module_name, description in import_tests:
        try:
            __import__(module_name)
            print(f"✅ {description}: {module_name}")
        except ImportError as e:
            print(f"❌ {description}: {module_name} - {e}")
            all_passed = False
    
    return all_passed

def test_project_imports():
    """测试项目模块导入"""
    print("\n=== 项目模块导入测试 ===")
    
    # 添加项目路径
    project_root = os.path.abspath(".")
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    project_modules = [
        ("src.config.settings", "配置设置"),
        ("src.api.models", "API模型"),
        ("src.core.moe.expert", "MoE专家模块"),
        ("src.core.multimodal.feature_extractor", "特征提取器"),
        ("src.monitoring.metrics_collector", "指标收集器")
    ]
    
    all_passed = True
    
    for module_name, description in project_modules:
        try:
            spec = importlib.util.find_spec(module_name)
            if spec is not None:
                print(f"✅ {description}: {module_name}")
            else:
                print(f"❌ {description}: {module_name} - 模块未找到")
                all_passed = False
        except Exception as e:
            print(f"❌ {description}: {module_name} - {e}")
            all_passed = False
    
    return all_passed

def test_config_loading():
    """测试配置加载"""
    print("\n=== 配置加载测试 ===")
    
    try:
        # 测试YAML配置文件
        import yaml
        config_file = "configs/config.yaml"
        
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            required_sections = ['gpu', 'model', 'inference', 'api']
            all_sections_present = all(section in config for section in required_sections)
            
            if all_sections_present:
                print(f"✅ 配置文件加载成功: {config_file}")
                print(f"   - GPU设备数: {config.get('gpu', {}).get('device_count', 'N/A')}")
                print(f"   - 专家数量: {config.get('model', {}).get('num_experts', 'N/A')}")
                print(f"   - API端口: {config.get('api', {}).get('port', 'N/A')}")
                return True
            else:
                print(f"❌ 配置文件缺少必要部分: {config_file}")
                return False
        else:
            print(f"❌ 配置文件不存在: {config_file}")
            return False
            
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        return False

def test_api_models():
    """测试API模型定义"""
    print("\n=== API模型测试 ===")
    
    try:
        # 添加项目路径
        project_root = os.path.abspath(".")
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        # 尝试导入API模型
        from src.api.models import InferenceRequest, InferenceResponse, SamplingParams
        
        # 测试创建模型实例
        sampling_params = SamplingParams(temperature=0.8, max_tokens=100)
        request = InferenceRequest(
            text="Hello world",
            code="print('hello')",
            mode="multimodal",
            sampling_params=sampling_params
        )
        
        print("✅ API模型导入和实例化成功")
        print(f"   - 请求模式: {request.mode}")
        print(f"   - 采样温度: {request.sampling_params.temperature}")
        return True
        
    except Exception as e:
        print(f"❌ API模型测试失败: {e}")
        return False

def test_core_components():
    """测试核心组件"""
    print("\n=== 核心组件测试 ===")
    
    try:
        project_root = os.path.abspath(".")
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        # 测试MoE组件
        from src.core.moe.expert import Expert, ExpertConfig
        from src.core.moe.router import Router, RouterConfig
        
        # 创建测试实例
        expert_config = ExpertConfig(input_dim=768, hidden_dim=3072, output_dim=768)
        expert = Expert(expert_config)
        
        router_config = RouterConfig(input_dim=768, num_experts=8, top_k=2)
        router = Router(router_config)
        
        print("✅ MoE核心组件导入成功")
        print(f"   - 专家输入维度: {expert_config.input_dim}")
        print(f"   - 路由器专家数: {router_config.num_experts}")
        
        return True
        
    except Exception as e:
        print(f"❌ 核心组件测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 LocalMoE环境测试开始")
    print("=" * 50)
    
    tests = [
        ("Python版本", test_python_version),
        ("项目结构", test_project_structure), 
        ("基础导入", test_basic_imports),
        ("项目导入", test_project_imports),
        ("配置加载", test_config_loading),
        ("API模型", test_api_models),
        ("核心组件", test_core_components)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name}测试异常: {e}")
            results.append((test_name, False))
    
    # 汇总结果
    print("\n" + "=" * 50)
    print("📊 测试结果汇总")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{total} 项测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！环境准备就绪。")
        return True
    else:
        print("⚠️  部分测试失败，请检查环境配置。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
