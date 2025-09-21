"""
GPU 和系统环境测试脚本
功能：测试GPU性能、内存使用、CUDA环境等
作者：PyTorch环境测试
日期：2025-06-28
"""

import torch
import time
import multiprocessing as mp
import sys
import os
import platform
import subprocess
import psutil
from typing import Dict, List, Optional, Tuple


def print_separator(title: str, width: int = 80) -> None:
    """打印分隔符和标题"""
    print("=" * width)
    print(f" {title} ".center(width))
    print("=" * width)


def get_system_info() -> Dict:
    """获取系统基本信息"""
    try:
        # 获取CPU信息
        cpu_count = psutil.cpu_count(logical=False)  # 物理核心
        cpu_count_logical = psutil.cpu_count(logical=True)  # 逻辑核心
        
        # 获取内存信息
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        
        # 获取系统信息
        system_info = {
            'platform': platform.platform(),
            'architecture': platform.architecture()[0],
            'processor': platform.processor(),
            'cpu_physical_cores': cpu_count,
            'cpu_logical_cores': cpu_count_logical,
            'memory_total_gb': memory_gb,
            'python_version': sys.version.split()[0],
            'pytorch_version': torch.__version__,
        }
        
        return system_info
    except Exception as e:
        print(f"⚠️  获取系统信息时出错: {e}")
        return {}


def print_system_info() -> None:
    """打印详细的系统信息"""
    print_separator("系统环境信息")
    
    system_info = get_system_info()
    
    print(f"🖥️  操作系统: {system_info.get('platform', '未知')}")
    print(f"🏗️  架构: {system_info.get('architecture', '未知')}")
    print(f"⚙️  处理器: {system_info.get('processor', '未知')}")
    print(f"🔧  CPU核心: {system_info.get('cpu_physical_cores', '未知')} 物理 / {system_info.get('cpu_logical_cores', '未知')} 逻辑")
    print(f"💾  总内存: {system_info.get('memory_total_gb', 0):.1f} GB")
    print(f"🐍  Python版本: {system_info.get('python_version', '未知')}")
    print(f"🔥  PyTorch版本: {system_info.get('pytorch_version', '未知')}")
    
    # CUDA信息
    print(f"\n📊  CUDA环境:")
    print(f"   - CUDA可用: {'✅ 是' if torch.cuda.is_available() else '❌ 否'}")
    
    if torch.cuda.is_available():
        print(f"   - CUDA版本: {torch.version.cuda}")
        print(f"   - cuDNN版本: {torch.backends.cudnn.version()}")
        print(f"   - GPU数量: {torch.cuda.device_count()}")
    else:
        print("   ⚠️  CUDA不可用，请检查PyTorch CUDA支持")


def get_gpu_memory_info(device_id: int) -> Tuple[float, float]:
    """获取GPU内存信息 (已用GB, 总计GB)"""
    try:
        torch.cuda.set_device(device_id)
        allocated = torch.cuda.memory_allocated(device_id) / (1024**3)
        total = torch.cuda.get_device_properties(device_id).total_memory / (1024**3)
        return allocated, total
    except Exception:
        return 0.0, 0.0


def test_gpu_comprehensive(gpu_id: int, matrix_size: int = 8000) -> Optional[Dict]:
    """综合GPU测试"""
    try:
        device = torch.device(f'cuda:{gpu_id}')
        torch.cuda.set_device(device)
        
        # 获取GPU基本信息
        gpu_props = torch.cuda.get_device_properties(gpu_id)
        gpu_name = gpu_props.name
        
        print(f"\n🎯  测试 GPU {gpu_id}: {gpu_name}")
        print(f"   - 计算能力: {gpu_props.major}.{gpu_props.minor}")
        print(f"   - 总内存: {gpu_props.total_memory / (1024**3):.1f} GB")
        print(f"   - 多处理器数量: {gpu_props.multi_processor_count}")
        
        # 内存测试
        mem_before_alloc, mem_total = get_gpu_memory_info(gpu_id)
        print(f"   - 测试前内存使用: {mem_before_alloc:.2f} GB / {mem_total:.1f} GB")
        
        # 创建测试矩阵
        print(f"   - 创建 {matrix_size}x{matrix_size} 矩阵...")
        matrix1 = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float32)
        matrix2 = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float32)
        
        mem_after_alloc, _ = get_gpu_memory_info(gpu_id)
        print(f"   - 矩阵分配后内存: {mem_after_alloc:.2f} GB")
        
        # 预热
        print("   - GPU预热中...")
        for _ in range(3):
            _ = torch.mm(matrix1, matrix2)
        torch.cuda.synchronize()
        
        # 性能测试
        print("   - 执行性能测试...")
        times = []
        for i in range(5):
            start_time = time.perf_counter()
            result = torch.mm(matrix1, matrix2)
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        # 计算FLOPS (浮点运算数)
        flops = 2 * matrix_size**3  # 矩阵乘法的FLOPS
        gflops = flops / (avg_time * 1e9)
        
        # 内存带宽测试
        print("   - 内存带宽测试...")
        data_size = matrix_size * matrix_size * 4 * 3  # 4字节float32 * 3个矩阵
        bandwidth_gb_s = data_size / (avg_time * 1e9)
        
        # 清理内存
        del matrix1, matrix2, result
        torch.cuda.empty_cache()
        
        results = {
            'gpu_id': gpu_id,
            'gpu_name': gpu_name,
            'compute_capability': f"{gpu_props.major}.{gpu_props.minor}",
            'total_memory_gb': gpu_props.total_memory / (1024**3),
            'avg_time': avg_time,
            'min_time': min_time,
            'max_time': max_time,
            'gflops': gflops,
            'bandwidth_gb_s': bandwidth_gb_s,
            'memory_used_gb': mem_after_alloc - mem_before_alloc
        }
        
        print(f"   ✅ 平均执行时间: {avg_time:.4f}s (范围: {min_time:.4f}s - {max_time:.4f}s)")
        print(f"   ⚡ 性能: {gflops:.1f} GFLOPS")
        print(f"   🚀 内存带宽: {bandwidth_gb_s:.1f} GB/s")
        
        return results
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"   ❌ GPU {gpu_id} 内存不足，尝试较小的矩阵")
            if matrix_size > 4000:
                return test_gpu_comprehensive(gpu_id, matrix_size // 2)
        print(f"   ❌ GPU {gpu_id} 测试失败: {e}")
        return None
    except Exception as e:
        print(f"   ❌ GPU {gpu_id} 测试出错: {e}")
        return None


def test_cpu_performance(matrix_size: int = 4000) -> Optional[Dict]:
    """CPU性能测试"""
    try:
        print(f"\n🖥️  CPU性能测试 (矩阵大小: {matrix_size}x{matrix_size})")
        
        # 设置线程数
        original_threads = torch.get_num_threads()
        torch.set_num_threads(psutil.cpu_count(logical=False))
        print(f"   - 使用线程数: {torch.get_num_threads()}")
        
        # 创建矩阵
        matrix1 = torch.randn(matrix_size, matrix_size, dtype=torch.float32)
        matrix2 = torch.randn(matrix_size, matrix_size, dtype=torch.float32)
        
        # 预热
        for _ in range(2):
            _ = torch.mm(matrix1, matrix2)
        
        # 性能测试
        times = []
        for _ in range(3):
            start_time = time.perf_counter()
            result = torch.mm(matrix1, matrix2)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        avg_time = sum(times) / len(times)
        
        # 计算FLOPS
        flops = 2 * matrix_size**3
        gflops = flops / (avg_time * 1e9)
        
        # 恢复原始线程数
        torch.set_num_threads(original_threads)
        
        print(f"   ✅ 平均执行时间: {avg_time:.4f}s")
        print(f"   ⚡ 性能: {gflops:.1f} GFLOPS")
        
        return {
            'avg_time': avg_time,
            'gflops': gflops,
            'threads_used': torch.get_num_threads()
        }
        
    except Exception as e:
        print(f"   ❌ CPU测试出错: {e}")
        return None


def run_comprehensive_tests() -> None:
    """运行综合测试"""
    print_separator("PyTorch GPU & 系统环境综合测试")
    
    # 系统信息
    print_system_info()
    
    # GPU测试
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print_separator(f"GPU 性能测试 (共 {num_gpus} 个GPU)")
        
        gpu_results = []
        for i in range(num_gpus):
            result = test_gpu_comprehensive(i)
            if result:
                gpu_results.append(result)
        
        # GPU测试总结
        if gpu_results:
            print_separator("GPU 测试总结")
            print(f"{'GPU':<6} {'设备名称':<25} {'时间(s)':<10} {'GFLOPS':<10} {'带宽(GB/s)':<12}")
            print("-" * 70)
            for result in gpu_results:
                print(f"GPU {result['gpu_id']:<3} {result['gpu_name']:<25} "
                      f"{result['avg_time']:<10.4f} {result['gflops']:<10.1f} "
                      f"{result['bandwidth_gb_s']:<12.1f}")
    else:
        print_separator("GPU 不可用")
        print("❌ 未检测到可用的GPU，跳过GPU测试")
    
    # CPU测试
    print_separator("CPU 性能测试")
    cpu_result = test_cpu_performance()
    
    # 最终总结
    print_separator("测试完成")
    print("✅ 所有测试已完成！")
    
    if torch.cuda.is_available():
        print(f"🎯 GPU测试: {torch.cuda.device_count()} 个设备")
    print("🖥️  CPU测试: 已完成")
    print("\n💡 提示: 如有问题，请检查CUDA驱动和PyTorch版本兼容性")


if __name__ == "__main__":
    try:
        # 设置多进程启动方法
        if hasattr(mp, 'set_start_method'):
            try:
                mp.set_start_method('spawn', force=True)
            except RuntimeError:
                pass  # 已经设置过了
        
        run_comprehensive_tests()
        
    except KeyboardInterrupt:
        print("\n\n⏹️  测试被用户中断")
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        sys.exit(1)