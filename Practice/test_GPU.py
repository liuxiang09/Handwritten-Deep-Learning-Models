import torch
import time
import multiprocessing as mp
import sys


def print_system_info():
    print("系统信息:")
    print(f"Python 版本: {sys.version}")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 版本: {torch.version.cuda}")
    else:
        print("CUDA 不可用。请检查您的PyTorch安装是否包含CUDA支持。")


def test_gpu(gpu_id, matrix_size=10000):
    try:
        device = torch.device(f'cuda:{gpu_id}')
        torch.cuda.set_device(device)

        matrix1 = torch.randn(matrix_size, matrix_size).to(device)
        matrix2 = torch.randn(matrix_size, matrix_size).to(device)

        torch.mm(matrix1, matrix2)
        torch.cuda.synchronize()

        start_time = time.time()
        result = torch.mm(matrix1, matrix2)
        torch.cuda.synchronize()
        end_time = time.time()

        execution_time = end_time - start_time
        gpu_name = torch.cuda.get_device_name(gpu_id)
        print(f"GPU {gpu_id} ({gpu_name}) 执行时间: {execution_time:.4f} 秒")

        return execution_time
    except Exception as e:
        print(f"测试 GPU {gpu_id} 时出错: {str(e)}")
        return None


def test_cpu(matrix_size=5000):
    matrix1 = torch.randn(matrix_size, matrix_size)
    matrix2 = torch.randn(matrix_size, matrix_size)

    start_time = time.time()
    result = torch.mm(matrix1, matrix2)
    end_time = time.time()

    return end_time - start_time


def run_tests():
    print_system_info()

    num_gpus = torch.cuda.device_count()
    print(f"\n检测到 {num_gpus} 个GPU")

    if num_gpus == 0:
        print("没有检测到可用的GPU。将只进行CPU测试。")
        cpu_time = test_cpu()
        print(f"CPU 执行时间: {cpu_time:.4f} 秒")
        return

    results = []
    for i in range(num_gpus):
        results.append(test_gpu(i))

    print("\n性能总结:")
    for i, time in enumerate(results):
        if time is not None:
            print(f"GPU {i}: {time:.4f} 秒")


if __name__ == "__main__":
    # 设置多进程启动方法为'spawn'
    mp.set_start_method('spawn')
    run_tests()