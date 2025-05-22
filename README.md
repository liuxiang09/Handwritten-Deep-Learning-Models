# PyTorch训练

> 这是一个用于学习和实践 PyTorch 深度学习框架的个人项目仓库。本仓库将逐步实现和训练各种深度学习模型，并最终融入 Hugging Face Transformers 的学习与应用。

---

## 🚀 项目目标

* **深入理解** PyTorch 框架的核心概念和API。
* **手写实现并训练** 经典的深度学习模型。
* **掌握** 常见计算机视觉和自然语言处理任务。
* **探索并应用** Hugging Face Transformers 库进行前沿模型实践。
* **构建一个可复用** 的模型训练与评估流程。

---

## 📦 项目结构

```
PyTorch_Training/
├── models/
│   ├── vgg/
│   │   ├── vgg.py            # VGG模型定义
│   │   └── train_vgg.py      # VGG训练脚本
│   ├── transformer/
│   │   ├── transformer.py    # Transformer模型定义 (你手写的源码)
│   │   └── train_transformer.py # Transformer训练脚本
│   └── ...                   # 未来添加更多模型，如ResNet, BERT等
├── datasets/
│   ├── translation/          # 翻译数据集 (例如，WMT, Multi30k)
│   │   ├── raw/              # 原始数据
│   │   └── processed/        # 预处理后的数据 (词汇表, tokenized files等)
│   ├── image_classification/ # 图像分类数据集 (例如，CIFAR-10, ImageNetSubset)
│   └── ...                   # 未来添加更多数据集
├── utils/
│   ├── data_preprocessing.py # 数据加载、分词、填充等通用工具
│   ├── training_utils.py     # 训练循环、评估、学习率调度等工具
│   ├── visualization.py      # 结果可视化工具
│   └── ...
├── huggingface_transformers/   # 专注于Hugging Face Transformers的学习与实践
│   ├── quick_start.py        # Hugging Face入门示例
│   ├── fine_tuning_bert.py   # BERT微调示例
│   └── ...
├── checkpoints/              # 训练好的模型权重保存目录
├── notebooks/                # Jupyter Notebooks (用于实验、数据探索或教学)
├── README.md                 # 项目说明文件
├── requirements.txt          # 项目依赖
└── setup.py                  # (可选) 如果项目结构更复杂，可以用于打包
```

---

## ✨ 已实现模型

### 1. VGG (Visual Geometry Group)

* **实现状态:** ✅ 完成
* **主要功能:** 图像分类
* **数据集:** (待补充，例如：CIFAR-10)
* **文件路径:** `models/vgg/vgg.py`, `models/vgg/train_vgg.py`
* **简要说明:** VGG 模型通过使用多层小卷积核（3x3），而非大卷积核，有效地增加了网络深度，同时保持了感受野，从而提高了模型的性能。
* **训练结果:** (可选，添加训练曲线或准确率截图)

### 2. Transformer

* **实现状态:** ✅ 完成 (核心架构)
* **主要功能:** 机器翻译
* **数据集:** (待补充，例如：WMT English-German Subset)
* **文件路径:** `models/transformer/transformer.py`, `models/transformer/train_transformer.py`
* **简要说明:** Transformer 模型完全基于自注意力（Self-Attention）机制，彻底摆脱了循环神经网络（RNN）和卷积神经网络（CNN）对序列数据的依赖，实现了并行化处理，极大地提升了处理长序列数据的效率。
* **当前进度:** 已通过 `shape` 输出测试，正准备数据和训练。
* **训练结果:** (待更新)

---

## 📚 学习计划 (未来)

* **更多经典模型实现:**
    * **CNNs:** ResNet, Inception, MobileNet
    * **RNNs/LSTMs/GRUs:** 序列建模，文本生成
    * **GANs:** 生成对抗网络
    * **Diffusion Models:** 扩散模型基础
* **高级 PyTorch 特性:**
    * `torch.distributed` (分布式训练)
    * `torch.jit` (JIT 编译)
    * `torch.quantization` (模型量化)
* **Hugging Face Transformers 深度学习:**
    * **模型加载与使用:** 预训练模型加载、Tokenizers
    * **任务微调:** 文本分类、命名实体识别、问答系统
    * **自定义模型与训练:** 结合 PyTorch 和 Hugging Face 生态
* **部署实践:**
    * ONNX 导出
    * TorchScript
    * 部署到边缘设备或云平台

---

## 🛠️ 环境配置

1.  **克隆仓库:**
    ```bash
    git clone [https://github.com/YourUsername/PyTorch_Training.git](https://github.com/YourUsername/PyTorch_Training.git)
    cd PyTorch_Training
    ```
2.  **创建 Conda 环境 (推荐):**
    
    ```bash
    conda create -n pytorch_env python=3.9
    conda activate pytorch_env
    ```
3.  **安装依赖:**
    ```bash
    pip install -r requirements.txt
    ```
    (请确保 `requirements.txt` 文件中包含了所有 PyTorch 和其他库的依赖，例如：`torch`, `torchvision`, `torchaudio`, `transformers`, `numpy`, `pandas`, `matplotlib`, `seaborn` 等。)
4.  **安装 PyTorch (根据你的CUDA版本):**
    请访问 [PyTorch 官网](https://pytorch.org/get-started/locally/) 获取适合你系统和 CUDA 版本的安装命令。
    例如 (CUDA 11.8):
    ```bash
    pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
    ```

---

## 💡 使用指南 (示例)

### 训练 VGG 模型

```bash
python models/vgg/train_vgg.py --epochs 10 --batch_size 64
```

### 训练 Transformer 模型 (数据准备完成后)

```bash
python models/transformer/train_transformer.py --dataset_path datasets/translation/processed --epochs 20 --batch_size 32
```

## 🤝 贡献

欢迎任何形式的贡献，包括但不限于：

提出 Bug
提交 Pull Request (优化代码、新增模型、改进文档等)
分享学习心得和资源

## 📄 许可证

本项目采用 MIT License 许可。

## 📞 联系我

- GitHub Issues: 提交问题或建议
- Email: liuxiang09192021@163.com