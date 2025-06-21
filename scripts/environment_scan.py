#!/usr/bin/env python3
"""
vLLM MoE模型部署环境扫描脚本
用于检查服务器的计算环境和组件依赖是否满足单机多卡MoE模型部署要求
"""

import os
import sys
import subprocess
import platform
import json
import psutil
import socket
from pathlib import Path
from typing import Dict, List, Any, Optional

class EnvironmentScanner:
    def __init__(self):
        self.results = {
            "system_info": {},
            "hardware": {},
            "gpu": {},
            "python_env": {},
            "dependencies": {},
            "network": {},
            "storage": {},
            "recommendations": []
        }
    
    def run_command(self, cmd: str, shell: bool = True) -> tuple[str, str, int]:
        """执行命令并返回输出"""
        try:
            result = subprocess.run(
                cmd, shell=shell, capture_output=True, text=True, timeout=30
            )
            return result.stdout.strip(), result.stderr.strip(), result.returncode
        except subprocess.TimeoutExpired:
            return "", "Command timeout", 1
        except Exception as e:
            return "", str(e), 1
    
    def check_system_info(self):
        """检查系统基本信息"""
        print("🔍 检查系统基本信息...")
        
        self.results["system_info"] = {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "hostname": socket.gethostname(),
            "python_version": platform.python_version(),
            "architecture": platform.architecture()[0]
        }
        
        # 检查内核版本
        if platform.system() == "Linux":
            kernel_version, _, _ = self.run_command("uname -r")
            self.results["system_info"]["kernel_version"] = kernel_version
    
    def check_hardware(self):
        """检查硬件信息"""
        print("🔍 检查硬件信息...")
        
        # CPU信息
        cpu_info = {
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "max_frequency": psutil.cpu_freq().max if psutil.cpu_freq() else "Unknown",
            "current_frequency": psutil.cpu_freq().current if psutil.cpu_freq() else "Unknown"
        }
        
        # 内存信息
        memory = psutil.virtual_memory()
        memory_info = {
            "total_gb": round(memory.total / (1024**3), 2),
            "available_gb": round(memory.available / (1024**3), 2),
            "used_gb": round(memory.used / (1024**3), 2),
            "percentage": memory.percent
        }
        
        # 交换分区信息
        swap = psutil.swap_memory()
        swap_info = {
            "total_gb": round(swap.total / (1024**3), 2),
            "used_gb": round(swap.used / (1024**3), 2),
            "percentage": swap.percent
        }
        
        self.results["hardware"] = {
            "cpu": cpu_info,
            "memory": memory_info,
            "swap": swap_info
        }
        
        # 检查CPU特性
        if platform.system() == "Linux":
            cpu_flags, _, _ = self.run_command("cat /proc/cpuinfo | grep flags | head -1")
            if cpu_flags:
                flags = cpu_flags.split(":")[1].strip().split()
                important_flags = ["avx", "avx2", "avx512f", "fma", "sse4_1", "sse4_2"]
                self.results["hardware"]["cpu"]["supported_features"] = [
                    flag for flag in important_flags if flag in flags
                ]
    
    def check_gpu(self):
        """检查GPU信息"""
        print("🔍 检查GPU信息...")
        
        gpu_info = {
            "nvidia_driver": None,
            "cuda_version": None,
            "gpus": [],
            "total_vram_gb": 0
        }
        
        # 检查NVIDIA驱动
        nvidia_smi, _, ret_code = self.run_command("nvidia-smi --version")
        if ret_code == 0:
            for line in nvidia_smi.split('\n'):
                if "Driver Version" in line:
                    gpu_info["nvidia_driver"] = line.split("Driver Version:")[1].split()[0]
                    break
        
        # 检查CUDA版本
        nvcc_version, _, ret_code = self.run_command("nvcc --version")
        if ret_code == 0:
            for line in nvcc_version.split('\n'):
                if "release" in line:
                    gpu_info["cuda_version"] = line.split("release")[1].split(",")[0].strip()
                    break
        
        # 检查GPU详细信息
        gpu_query, _, ret_code = self.run_command(
            "nvidia-smi --query-gpu=index,name,memory.total,memory.free,memory.used,temperature.gpu,power.draw,power.limit --format=csv,noheader,nounits"
        )
        
        if ret_code == 0:
            for line in gpu_query.split('\n'):
                if line.strip():
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 8:
                        gpu_data = {
                            "index": int(parts[0]),
                            "name": parts[1],
                            "memory_total_mb": int(parts[2]),
                            "memory_free_mb": int(parts[3]),
                            "memory_used_mb": int(parts[4]),
                            "temperature_c": parts[5],
                            "power_draw_w": parts[6],
                            "power_limit_w": parts[7]
                        }
                        gpu_info["gpus"].append(gpu_data)
                        gpu_info["total_vram_gb"] += gpu_data["memory_total_mb"] / 1024
        
        gpu_info["total_vram_gb"] = round(gpu_info["total_vram_gb"], 2)
        self.results["gpu"] = gpu_info
    
    def check_python_environment(self):
        """检查Python环境"""
        print("🔍 检查Python环境...")
        
        python_info = {
            "version": sys.version,
            "executable": sys.executable,
            "virtual_env": os.environ.get("VIRTUAL_ENV"),
            "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
            "pip_version": None,
            "site_packages": []
        }
        
        # 检查pip版本
        pip_version, _, ret_code = self.run_command(f"{sys.executable} -m pip --version")
        if ret_code == 0:
            python_info["pip_version"] = pip_version
        
        # 检查site-packages路径
        import site
        python_info["site_packages"] = site.getsitepackages()
        
        self.results["python_env"] = python_info
    
    def check_dependencies(self):
        """检查关键依赖包"""
        print("🔍 检查关键依赖包...")
        
        key_packages = [
            "torch", "torchvision", "transformers", "vllm", 
            "fastapi", "uvicorn", "numpy", "pandas", "accelerate",
            "deepspeed", "flash-attn", "xformers", "triton"
        ]
        
        installed_packages = {}
        
        for package in key_packages:
            try:
                result = subprocess.run(
                    [sys.executable, "-c", f"import {package}; print({package}.__version__)"],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    installed_packages[package] = result.stdout.strip()
                else:
                    installed_packages[package] = "Not installed"
            except:
                installed_packages[package] = "Not installed"
        
        self.results["dependencies"] = installed_packages

    def check_network(self):
        """检查网络配置"""
        print("🔍 检查网络配置...")

        network_info = {
            "interfaces": [],
            "dns_servers": [],
            "internet_connectivity": False,
            "huggingface_connectivity": False
        }

        # 检查网络接口
        for interface, addrs in psutil.net_if_addrs().items():
            for addr in addrs:
                if addr.family == socket.AF_INET:
                    network_info["interfaces"].append({
                        "interface": interface,
                        "ip": addr.address,
                        "netmask": addr.netmask
                    })

        # 检查DNS
        try:
            with open('/etc/resolv.conf', 'r') as f:
                for line in f:
                    if line.startswith('nameserver'):
                        network_info["dns_servers"].append(line.split()[1])
        except:
            pass

        # 检查互联网连接
        try:
            result = subprocess.run(
                ["ping", "-c", "1", "8.8.8.8"],
                capture_output=True, timeout=5
            )
            network_info["internet_connectivity"] = result.returncode == 0
        except:
            pass

        # 检查HuggingFace连接
        try:
            result = subprocess.run(
                ["ping", "-c", "1", "huggingface.co"],
                capture_output=True, timeout=5
            )
            network_info["huggingface_connectivity"] = result.returncode == 0
        except:
            pass

        self.results["network"] = network_info

    def check_storage(self):
        """检查存储信息"""
        print("🔍 检查存储信息...")

        storage_info = {
            "disks": [],
            "total_space_gb": 0,
            "free_space_gb": 0,
            "tmp_space_gb": 0
        }

        # 检查磁盘使用情况
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                disk_info = {
                    "device": partition.device,
                    "mountpoint": partition.mountpoint,
                    "fstype": partition.fstype,
                    "total_gb": round(usage.total / (1024**3), 2),
                    "used_gb": round(usage.used / (1024**3), 2),
                    "free_gb": round(usage.free / (1024**3), 2),
                    "percentage": round((usage.used / usage.total) * 100, 2)
                }
                storage_info["disks"].append(disk_info)

                if partition.mountpoint == "/":
                    storage_info["total_space_gb"] = disk_info["total_gb"]
                    storage_info["free_space_gb"] = disk_info["free_gb"]
            except PermissionError:
                continue

        # 检查/tmp空间
        try:
            tmp_usage = psutil.disk_usage("/tmp")
            storage_info["tmp_space_gb"] = round(tmp_usage.free / (1024**3), 2)
        except:
            pass

        self.results["storage"] = storage_info

    def generate_recommendations(self):
        """生成建议和警告"""
        print("🔍 生成环境建议...")

        recommendations = []

        # 内存检查
        memory_gb = self.results["hardware"]["memory"]["total_gb"]
        if memory_gb < 32:
            recommendations.append("⚠️  内存不足：建议至少32GB内存用于MoE模型部署")
        elif memory_gb < 64:
            recommendations.append("⚠️  内存偏低：推荐64GB+内存以获得更好性能")

        # GPU检查
        gpu_count = len(self.results["gpu"]["gpus"])
        total_vram = self.results["gpu"]["total_vram_gb"]

        if gpu_count == 0:
            recommendations.append("❌ 未检测到GPU：MoE模型部署需要GPU支持")
        elif gpu_count < 2:
            recommendations.append("⚠️  单GPU配置：多卡部署建议至少2张GPU")

        if total_vram < 48:
            recommendations.append("⚠️  显存不足：大型MoE模型建议总显存48GB+")

        # CUDA检查
        if not self.results["gpu"]["cuda_version"]:
            recommendations.append("❌ 未安装CUDA：请安装CUDA 11.8+")

        # 依赖检查
        critical_deps = ["torch", "vllm", "transformers"]
        for dep in critical_deps:
            if self.results["dependencies"].get(dep) == "Not installed":
                recommendations.append(f"❌ 缺少关键依赖：{dep}")

        # 存储检查
        free_space = self.results["storage"]["free_space_gb"]
        if free_space < 100:
            recommendations.append("⚠️  存储空间不足：建议至少100GB可用空间")

        # 网络检查
        if not self.results["network"]["internet_connectivity"]:
            recommendations.append("⚠️  网络连接问题：无法访问互联网")

        if not self.results["network"]["huggingface_connectivity"]:
            recommendations.append("⚠️  HuggingFace连接问题：可能影响模型下载")

        self.results["recommendations"] = recommendations

    def print_results(self):
        """打印扫描结果"""
        print("\n" + "="*80)
        print("🚀 vLLM MoE模型部署环境扫描报告")
        print("="*80)

        # 系统信息
        print("\n📋 系统信息:")
        sys_info = self.results["system_info"]
        print(f"  操作系统: {sys_info['platform']}")
        print(f"  主机名: {sys_info['hostname']}")
        print(f"  架构: {sys_info['architecture']}")
        print(f"  Python版本: {sys_info['python_version']}")
        if 'kernel_version' in sys_info:
            print(f"  内核版本: {sys_info['kernel_version']}")

        # 硬件信息
        print("\n💻 硬件信息:")
        hw = self.results["hardware"]
        print(f"  CPU核心: {hw['cpu']['physical_cores']}物理/{hw['cpu']['logical_cores']}逻辑")
        print(f"  内存: {hw['memory']['total_gb']}GB (可用: {hw['memory']['available_gb']}GB)")
        print(f"  交换分区: {hw['swap']['total_gb']}GB")
        if 'supported_features' in hw['cpu']:
            print(f"  CPU特性: {', '.join(hw['cpu']['supported_features'])}")

        # GPU信息
        print("\n🎮 GPU信息:")
        gpu = self.results["gpu"]
        if gpu['nvidia_driver']:
            print(f"  NVIDIA驱动: {gpu['nvidia_driver']}")
        if gpu['cuda_version']:
            print(f"  CUDA版本: {gpu['cuda_version']}")

        print(f"  GPU数量: {len(gpu['gpus'])}")
        print(f"  总显存: {gpu['total_vram_gb']}GB")

        for i, gpu_info in enumerate(gpu['gpus']):
            print(f"    GPU {i}: {gpu_info['name']}")
            print(f"      显存: {gpu_info['memory_total_mb']/1024:.1f}GB "
                  f"(已用: {gpu_info['memory_used_mb']/1024:.1f}GB)")
            print(f"      温度: {gpu_info['temperature_c']}°C")
            print(f"      功耗: {gpu_info['power_draw_w']}W/{gpu_info['power_limit_w']}W")

        # Python环境
        print("\n🐍 Python环境:")
        py_env = self.results["python_env"]
        print(f"  Python路径: {py_env['executable']}")
        if py_env['virtual_env']:
            print(f"  虚拟环境: {py_env['virtual_env']}")
        if py_env['conda_env']:
            print(f"  Conda环境: {py_env['conda_env']}")
        if py_env['pip_version']:
            print(f"  Pip版本: {py_env['pip_version']}")

        # 依赖包
        print("\n📦 关键依赖包:")
        deps = self.results["dependencies"]
        for package, version in deps.items():
            status = "✅" if version != "Not installed" else "❌"
            print(f"  {status} {package}: {version}")

        # 网络
        print("\n🌐 网络配置:")
        net = self.results["network"]
        print(f"  网络接口数: {len(net['interfaces'])}")
        print(f"  互联网连接: {'✅' if net['internet_connectivity'] else '❌'}")
        print(f"  HuggingFace连接: {'✅' if net['huggingface_connectivity'] else '❌'}")

        # 存储
        print("\n💾 存储信息:")
        storage = self.results["storage"]
        print(f"  总空间: {storage['total_space_gb']}GB")
        print(f"  可用空间: {storage['free_space_gb']}GB")
        print(f"  临时空间: {storage['tmp_space_gb']}GB")

        for disk in storage['disks']:
            print(f"    {disk['device']} ({disk['mountpoint']}): "
                  f"{disk['free_gb']}/{disk['total_gb']}GB 可用")

        # 建议
        print("\n💡 建议和警告:")
        if self.results["recommendations"]:
            for rec in self.results["recommendations"]:
                print(f"  {rec}")
        else:
            print("  ✅ 环境配置良好，满足MoE模型部署要求")

        print("\n" + "="*80)

    def save_results(self, filename: str = "environment_scan_results.json"):
        """保存结果到JSON文件"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            print(f"\n📄 详细结果已保存到: {filename}")
        except Exception as e:
            print(f"\n❌ 保存结果失败: {e}")

    def run_scan(self):
        """运行完整扫描"""
        print("🔍 开始vLLM MoE模型部署环境扫描...")

        try:
            self.check_system_info()
            self.check_hardware()
            self.check_gpu()
            self.check_python_environment()
            self.check_dependencies()
            self.check_network()
            self.check_storage()
            self.generate_recommendations()

            self.print_results()
            self.save_results()

        except KeyboardInterrupt:
            print("\n\n⚠️  扫描被用户中断")
        except Exception as e:
            print(f"\n❌ 扫描过程中出现错误: {e}")
            import traceback
            traceback.print_exc()


def main():
    """主函数"""
    print("vLLM MoE模型部署环境扫描工具")
    print("适用于单机多卡部署场景")
    print("-" * 50)

    # 检查是否为root用户（推荐）
    if os.geteuid() != 0:
        print("⚠️  建议以root用户运行以获取完整系统信息")

    scanner = EnvironmentScanner()
    scanner.run_scan()


if __name__ == "__main__":
    main()
