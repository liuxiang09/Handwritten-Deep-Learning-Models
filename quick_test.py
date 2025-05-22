import sys
import torch
import os
import pkg_resources

def check_environment():
    print("=== Python 环境信息 ===")
    print(f"Python 版本: {sys.version}")
    
    print("\n=== CUDA 环境 ===")
    print(f"CUDA_HOME: {os.environ.get('CUDA_HOME', '未设置')}")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"GPU 设备: {torch.cuda.get_device_name(0)}")
    
    print("\n=== 关键包版本 ===")
    packages = [
        'transformers',
        'accelerate',
        'bitsandbytes',
        'peft',
        'flash-attn',
        'deepspeed',
        'torch',
        'triton',
        'torchvision',
        'torchaudio',
    ]
    
    for package in packages:
        try:
            version = pkg_resources.get_distribution(package).version
            print(f"{package}: {version}")
        except pkg_resources.DistributionNotFound:
            print(f"{package}: 未安装")

if __name__ == "__main__":
    check_environment()