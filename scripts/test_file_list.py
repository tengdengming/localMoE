#!/usr/bin/env python3
"""
测试模型文件列表获取
用于调试下载脚本
"""

import requests
import json
import sys

def test_file_list(model_name, use_mirror=False):
    """测试获取模型文件列表"""
    
    print(f"🔍 Testing file list for: {model_name}")
    print(f"📡 Use mirror: {use_mirror}")
    print("=" * 50)
    
    if use_mirror:
        base_url = "https://hf-mirror.com"
    else:
        base_url = "https://huggingface.co"
    
    print(f"🌐 Base URL: {base_url}")
    print()
    
    # 方法1: 尝试获取 index.json
    print("📋 Method 1: Checking model.safetensors.index.json")
    index_url = f"{base_url}/{model_name}/resolve/main/model.safetensors.index.json"
    print(f"URL: {index_url}")
    
    try:
        response = requests.get(index_url, timeout=30)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            index_data = response.json()
            weight_map = index_data.get('weight_map', {})
            weight_files = set(weight_map.values())
            
            print(f"✅ Found index file!")
            print(f"📁 Weight files in index: {len(weight_files)}")
            
            for i, weight_file in enumerate(sorted(weight_files)[:10]):  # 显示前10个
                print(f"   {i+1}. {weight_file}")
            
            if len(weight_files) > 10:
                print(f"   ... and {len(weight_files) - 10} more files")
                
            # 测试第一个文件是否可访问
            if weight_files:
                first_file = list(weight_files)[0]
                test_url = f"{base_url}/{model_name}/resolve/main/{first_file}"
                test_resp = requests.head(test_url, timeout=10)
                print(f"🔗 First file test: {test_resp.status_code}")
                if test_resp.status_code == 200:
                    size = int(test_resp.headers.get('content-length', 0))
                    print(f"📏 File size: {size / (1024**3):.2f} GB")
                    
        else:
            print(f"❌ Index file not found (status: {response.status_code})")
            
    except Exception as e:
        print(f"❌ Error getting index: {e}")
    
    print()
    
    # 方法2: 尝试 HuggingFace API (仅非镜像)
    if not use_mirror:
        print("📋 Method 2: Using HuggingFace API")
        api_url = f"https://huggingface.co/api/models/{model_name}/tree/main"
        print(f"URL: {api_url}")
        
        try:
            response = requests.get(api_url, timeout=30)
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                files = response.json()
                model_files = []
                
                for item in files:
                    if item.get('type') == 'file':
                        file_path = item.get('path', '')
                        if any(file_path.endswith(ext) for ext in 
                              ['.bin', '.safetensors', '.json', '.txt', '.py', '.md']) or \
                           'model-' in file_path:
                            model_files.append({
                                'path': file_path,
                                'size': item.get('size', 0)
                            })
                
                print(f"✅ Found {len(model_files)} files via API")
                
                # 显示权重文件
                weight_files = [f for f in model_files if 'safetensors' in f['path'] or 'bin' in f['path']]
                print(f"🏋️ Weight files: {len(weight_files)}")
                
                total_size = sum(f['size'] for f in model_files)
                print(f"📏 Total size: {total_size / (1024**3):.2f} GB")
                
                # 显示前几个文件
                for i, file_info in enumerate(sorted(model_files, key=lambda x: x['path'])[:10]):
                    size_mb = file_info['size'] / (1024**2)
                    print(f"   {i+1}. {file_info['path']} ({size_mb:.1f} MB)")
                    
            else:
                print(f"❌ API request failed (status: {response.status_code})")
                
        except Exception as e:
            print(f"❌ Error with API: {e}")
    
    print()
    
    # 方法3: 测试常见文件
    print("📋 Method 3: Testing common files")
    common_files = [
        'config.json',
        'tokenizer_config.json', 
        'model.safetensors',
        'model-00001-of-00010.safetensors',
        'model-00001-of-00015.safetensors',
        'model-00001-of-00020.safetensors'
    ]
    
    found_files = []
    for filename in common_files:
        url = f"{base_url}/{model_name}/resolve/main/{filename}"
        try:
            response = requests.head(url, timeout=10)
            if response.status_code == 200:
                size = int(response.headers.get('content-length', 0))
                found_files.append((filename, size))
                print(f"✅ {filename} - {size / (1024**2):.1f} MB")
            else:
                print(f"❌ {filename} - Not found")
        except Exception as e:
            print(f"❌ {filename} - Error: {e}")
    
    print()
    print("📊 Summary:")
    print(f"   Found {len(found_files)} common files")
    if found_files:
        total_common_size = sum(size for _, size in found_files)
        print(f"   Total size of found files: {total_common_size / (1024**3):.2f} GB")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_file_list.py <model_name> [--use-mirror]")
        print("Example: python test_file_list.py Qwen/Qwen3-30B-A3B --use-mirror")
        sys.exit(1)
    
    model_name = sys.argv[1]
    use_mirror = "--use-mirror" in sys.argv
    
    test_file_list(model_name, use_mirror)
