"""
Python环境和依赖包检查脚本
功能：检查Python环境、CUDA环境、关键包版本、兼容性等
作者：PyTorch环境测试
日期：2025-06-28
"""

import sys
import os
import platform
import subprocess
import importlib
import warnings
from typing import Dict, List, Tuple, Optional
from pathlib import Path


def print_separator(title: str, width: int = 80) -> None:
    """打印分隔符和标题"""
    print("=" * width)
    print(f" {title} ".center(width))
    print("=" * width)


def get_package_version(package_name: str) -> Optional[str]:
    """安全获取包版本"""
    try:
        # 优先使用importlib.metadata (Python 3.8+)
        if sys.version_info >= (3, 8):
            import importlib.metadata
            return importlib.metadata.version(package_name)
        else:
            # 回退到pkg_resources
            import pkg_resources
            return pkg_resources.get_distribution(package_name).version
    except (ImportError, Exception):
        try:
            # 尝试直接导入并获取版本
            module = importlib.import_module(package_name)
            if hasattr(module, '__version__'):
                return module.__version__
            elif hasattr(module, 'version'):
                return module.version
            elif hasattr(module, 'VERSION'):
                return str(module.VERSION)
        except ImportError:
            pass
    return None


def check_python_environment() -> Dict:
    """检查Python环境详细信息"""
    print_separator("Python 环境信息")
    
    env_info = {
        'python_version': sys.version.split()[0],
        'python_executable': sys.executable,
        'platform': platform.platform(),
        'architecture': platform.architecture()[0],
        'processor': platform.processor() or platform.machine(),
        'python_path': sys.path[:3],  # 显示前3个路径
        'site_packages': None
    }
    
    # 查找site-packages路径
    for path in sys.path:
        if 'site-packages' in path:
            env_info['site_packages'] = path
            break
    
    print(f"🐍  Python版本: {env_info['python_version']}")
    print(f"📍  Python路径: {env_info['python_executable']}")
    print(f"🖥️  操作系统: {env_info['platform']}")
    print(f"🏗️  架构: {env_info['architecture']}")
    print(f"⚙️  处理器: {env_info['processor']}")
    print(f"📦  Site-packages: {env_info['site_packages'] or '未找到'}")
    
    # 检查虚拟环境
    venv_info = check_virtual_environment()
    if venv_info:
        print(f"🌍  虚拟环境: {venv_info}")
    
    return env_info


def check_virtual_environment() -> Optional[str]:
    """检查虚拟环境信息"""
    # 检查常见的虚拟环境标志
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        venv_type = "virtualenv/venv"
        
        # 检查conda环境
        if 'conda' in sys.executable or 'CONDA_DEFAULT_ENV' in os.environ:
            venv_type = "conda"
            conda_env = os.environ.get('CONDA_DEFAULT_ENV', '未知')
            return f"{venv_type} ({conda_env})"
        
        return venv_type
    
    return None


def check_cuda_environment() -> Dict:
    """检查CUDA环境"""
    print_separator("CUDA 环境检查")
    
    cuda_info = {
        'cuda_available': False,
        'cuda_version': None,
        'cudnn_version': None,
        'nvidia_driver': None,
        'cuda_home': os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH'),
        'gpu_count': 0
    }
    
    # 检查PyTorch CUDA支持
    try:
        import torch
        cuda_info['cuda_available'] = torch.cuda.is_available()
        cuda_info['pytorch_version'] = torch.__version__
        
        if cuda_info['cuda_available']:
            cuda_info['cuda_version'] = torch.version.cuda
            cuda_info['gpu_count'] = torch.cuda.device_count()
            
            # 检查cuDNN
            if torch.backends.cudnn.is_available():
                cuda_info['cudnn_version'] = torch.backends.cudnn.version()
            
            print(f"✅ CUDA可用: 是")
            print(f"🔥 PyTorch版本: {cuda_info['pytorch_version']}")
            print(f"🎯 CUDA版本: {cuda_info['cuda_version']}")
            print(f"🧠 cuDNN版本: {cuda_info['cudnn_version'] or '未检测到'}")
            print(f"🎮 GPU数量: {cuda_info['gpu_count']}")
            
            # 显示GPU信息
            for i in range(cuda_info['gpu_count']):
                gpu_name = torch.cuda.get_device_name(i)
                memory_gb = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                print(f"   GPU {i}: {gpu_name} ({memory_gb:.1f} GB)")
        else:
            print(f"❌ CUDA可用: 否")
            print(f"🔥 PyTorch版本: {cuda_info['pytorch_version']}")
            
    except ImportError:
        print("❌ PyTorch未安装")
    
    # 检查NVIDIA驱动
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            cuda_info['nvidia_driver'] = result.stdout.strip().split('\n')[0]
            print(f"🚗 NVIDIA驱动: {cuda_info['nvidia_driver']}")
        else:
            print("❌ 无法获取NVIDIA驱动信息")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ nvidia-smi 命令不可用")
    
    # CUDA_HOME信息
    print(f"🏠 CUDA_HOME: {cuda_info['cuda_home'] or '未设置'}")
    
    return cuda_info


def check_essential_packages() -> Dict[str, Optional[str]]:
    """检查核心包版本"""
    print_separator("核心依赖包检查")
    
    # 分类包列表
    package_categories = {
        'PyTorch生态': [
            'torch', 'torchvision', 'torchaudio', 'torchtext'
        ],
        '深度学习框架': [
            'tensorflow', 'jax', 'flax'
        ],
        'Transformers生态': [
            'transformers', 'accelerate', 'peft', 'bitsandbytes'
        ],
        '数据科学': [
            'numpy', 'pandas', 'scikit-learn', 'matplotlib', 'seaborn', 'plotly'
        ],
        '计算机视觉': [
            'opencv-python', 'pillow', 'albumentations', 'timm'
        ],
        'NLP工具': [
            'nltk', 'spacy', 'tokenizers', 'datasets'
        ],
        '优化工具': [
            'triton', 'flash-attn', 'deepspeed', 'xformers'
        ],
        '其他工具': [
            'jupyter', 'ipython', 'tqdm', 'wandb', 'tensorboard'
        ]
    }
    
    all_results = {}
    
    for category, packages in package_categories.items():
        print(f"\n📦 {category}:")
        category_results = {}
        
        for package in packages:
            version = get_package_version(package)
            category_results[package] = version
            
            if version:
                status = "✅"
                version_info = version
            else:
                status = "❌"
                version_info = "未安装"
            
            print(f"   {status} {package:<20} {version_info}")
        
        all_results.update(category_results)
    
    return all_results


def check_package_compatibility() -> List[str]:
    """检查包兼容性问题"""
    print_separator("兼容性检查")
    
    warnings_list = []
    
    try:
        import torch
        
        # 检查PyTorch和Python版本兼容性
        python_version = tuple(map(int, sys.version.split()[0].split('.')))
        pytorch_version = torch.__version__
        
        if python_version >= (3, 12):
            warnings_list.append("⚠️  Python 3.12+ 可能与某些包不兼容")
        
        # 检查CUDA版本兼容性
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            if cuda_version:
                cuda_major = int(cuda_version.split('.')[0])
                if cuda_major < 11:
                    warnings_list.append(f"⚠️  CUDA版本 {cuda_version} 较旧，建议升级到11.0+")
        
        # 检查关键包组合
        packages_to_check = {
            'transformers': 'accelerate',
            'torch': 'torchvision',
            'datasets': 'tokenizers'
        }
        
        for pkg1, pkg2 in packages_to_check.items():
            ver1 = get_package_version(pkg1)
            ver2 = get_package_version(pkg2)
            
            if ver1 and not ver2:
                warnings_list.append(f"⚠️  {pkg1} 已安装但缺少 {pkg2}")
        
    except ImportError:
        warnings_list.append("❌ PyTorch未安装，无法进行深度兼容性检查")
    
    if warnings_list:
        print("发现以下兼容性问题:")
        for warning in warnings_list:
            print(f"  {warning}")
    else:
        print("✅ 未发现明显的兼容性问题")
    
    return warnings_list


def check_development_tools() -> Dict[str, Optional[str]]:
    """检查开发工具"""
    print_separator("开发工具检查")
    
    dev_tools = {
        'Git': 'git --version',
        'pip': 'pip --version',
        'conda': 'conda --version',
        'jupyter': 'jupyter --version',
        'code': 'code --version'
    }
    
    results = {}
    
    for tool_name, command in dev_tools.items():
        try:
            result = subprocess.run(command.split(), capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                version_line = result.stdout.strip().split('\n')[0]
                results[tool_name] = version_line
                print(f"✅ {tool_name:<10} {version_line}")
            else:
                results[tool_name] = None
                print(f"❌ {tool_name:<10} 不可用")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            results[tool_name] = None
            print(f"❌ {tool_name:<10} 未安装或不在PATH中")
    
    return results


def generate_summary_report(env_info: Dict, cuda_info: Dict, packages: Dict, warnings: List[str]) -> None:
    """生成总结报告"""
    print_separator("环境检查总结报告")
    
    # 统计安装的包数量
    installed_packages = sum(1 for v in packages.values() if v is not None)
    total_packages = len(packages)
    
    print(f"📊 基本信息:")
    print(f"   - Python: {env_info.get('python_version', '未知')}")
    print(f"   - 平台: {env_info.get('platform', '未知')}")
    print(f"   - 虚拟环境: {'是' if check_virtual_environment() else '否'}")
    
    print(f"\n🎮 CUDA支持:")
    print(f"   - CUDA可用: {'是' if cuda_info.get('cuda_available') else '否'}")
    if cuda_info.get('cuda_available'):
        print(f"   - GPU数量: {cuda_info.get('gpu_count', 0)}")
        print(f"   - CUDA版本: {cuda_info.get('cuda_version', '未知')}")
    
    print(f"\n📦 包安装情况:")
    print(f"   - 已安装: {installed_packages}/{total_packages} 个包")
    print(f"   - 安装率: {(installed_packages/total_packages)*100:.1f}%")
    
    if warnings:
        print(f"\n⚠️  发现 {len(warnings)} 个潜在问题:")
        for warning in warnings[:3]:  # 显示前3个警告
            print(f"   - {warning}")
        if len(warnings) > 3:
            print(f"   - ... 还有 {len(warnings)-3} 个问题")
    else:
        print(f"\n✅ 环境状态良好，未发现问题")
    
    print(f"\n💡 建议:")
    if not cuda_info.get('cuda_available'):
        print("   - 考虑安装CUDA支持的PyTorch版本以获得更好性能")
    if installed_packages < total_packages * 0.5:
        print("   - 考虑安装更多必要的依赖包")
    print("   - 定期更新包版本以获得最新功能和安全修复")


def run_comprehensive_check() -> None:
    """运行综合环境检查"""
    print_separator("Python 环境与依赖包综合检查")
    print("🔍 开始检查Python环境和依赖包...")
    
    try:
        # 1. Python环境检查
        env_info = check_python_environment()
        
        # 2. CUDA环境检查  
        cuda_info = check_cuda_environment()
        
        # 3. 包版本检查
        packages = check_essential_packages()
        
        # 4. 兼容性检查
        warnings_list = check_package_compatibility()
        
        # 5. 开发工具检查
        dev_tools = check_development_tools()
        
        # 6. 生成总结报告
        generate_summary_report(env_info, cuda_info, packages, warnings_list)
        
        print_separator("检查完成")
        print("✅ 环境检查已完成！")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  检查被用户中断")
    except Exception as e:
        print(f"\n❌ 检查过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_comprehensive_check()