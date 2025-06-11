# PyTorch训练

> 这是一个用于学习和实践 PyTorch 深度学习框架的个人项目仓库。本仓库将逐步实现和训练各种深度学习模型，并最终融入 Hugging Face Transformers 的学习与应用。

---

## 📑 目录

- [PyTorch训练](#pytorch训练)
  - [📑 目录](#-目录)
  - [🚀 项目目标](#-项目目标)
  - [📦 项目结构](#-项目结构)
  - [✨ 已实现模型](#-已实现模型)
    - [1. VGG (Visual Geometry Group)](#1-vgg-visual-geometry-group)
    - [2. Transformer](#2-transformer)
    - [3. CLIP (Contrastive Language-Image Pre-training)](#3-clip-contrastive-language-image-pre-training)
    - [4. ViT (Vision Transformer)](#4-vit-vision-transformer)
    - [5. ViLT (Vision-and-Language Transformer)](#5-vilt-vision-and-language-transformer)
    - [6. DETR (DEtection TRansformer)](#6-detr-detection-transformer)
  - [📚 学习计划 (未来)](#-学习计划-未来)
  - [🛠️ 环境配置](#️-环境配置)
  - [💡 使用指南 (示例)](#-使用指南-示例)
    - [模型训练](#模型训练)
    - [模型评估](#模型评估)
    - [模型推理](#模型推理)
  - [🤝 贡献](#-贡献)
  - [📄 许可证](#-许可证)
  - [📞 联系我](#-联系我)



## 🚀 项目目标

* **深入理解** PyTorch 框架的核心概念和API。
* **手写实现并训练** 经典的深度学习模型。
* **掌握** 常见计算机视觉和自然语言处理任务。
* **探索并应用** Hugging Face Transformers 库进行前沿模型实践。
* **构建一个可复用** 的模型训练与评估流程。

---

## 📦 项目结构

```
PyTorch/
├── .gitignore
├── models/
│   ├── CLIP/...
│   ├── Transformer/...
│   ├── VGG/...
│   ├── ViLT/...
│   └── ViT/...
├── TEST/
│   ├── test_AutoTokenizer.py
│   ├── test_dataset.py
│   ├── test_GPU.py
│   └── ...
├── README.md
└── requirements.txt
```

---

## ✨ 已实现模型

### 1. VGG (Visual Geometry Group)

* **实现状态:** ✅ 完成
* **主要功能:** 图像分类
* **数据集:** CIFAR-10
* **简要说明:** VGG 模型通过使用多层小卷积核（3x3），而非大卷积核，有效地增加了网络深度，同时保持了感受野，从而提高了模型的性能。
* **当前进度:** 已完成模型训练和评估。

### 2. Transformer

* **实现状态:** ✅ 完成
* **主要功能:** 机器翻译
* **数据集:** Multi30k
* **简要说明:** Transformer 模型完全基于自注意力（Self-Attention）机制，彻底摆脱了循环神经网络（RNN）和卷积神经网络（CNN）对序列数据的依赖，实现了并行化处理，极大地提升了处理长序列数据的效率。
* **当前进度:** 已完成模型训练和评估。

### 3. CLIP (Contrastive Language-Image Pre-training)

* **实现状态:** ⚙️ 进行中
* **主要功能:** 跨模态图像-文本理解
* **数据集:** COCO
* **简要说明:** CLIP通过在大规模图像-文本对上进行对比学习，使得模型能够理解图像和文本之间的语义关系，从而实现零样本（zero-shot）图像分类、图像检索等任务。
* **当前进度:** 正在进行数据集准备、处理以及模型训练。

### 4. ViT (Vision Transformer)

* **实现状态:** ✅ 完成
* **主要功能:** 图像分类
* **数据集:** CIFAR-10
* **简要说明:** ViT 将 Transformer 架构首次成功应用于计算机视觉任务，将图像视为一系列序列化的图像块（patches），并直接应用于标准的 Transformer 编码器进行分类。
* **实现方式:** 使用 Hugging Face Transformers 库完成训练和评估流程。
* **当前进度:** 已使用transformers库完成模型训练和评估，暂未手写实现具体模块（可在CLIP中参考）。

### 5. ViLT (Vision-and-Language Transformer)

* **实现状态:** 🚧 初步规划
* **主要功能:** 视觉-语言多模态任务
* **数据集:** COCO
* **简要说明:** ViLT 是一种视觉-语言 Transformer 模型，它通过联合处理图像和文本，实现图像-文本匹配、视觉问答等多模态任务。
* **当前进度:** 已有推理脚本框架，正在设计模型架构。

### 6. DETR (DEtection TRansformer)

* **实现状态:** 🚧 初步规划
* **主要功能:** 端到端目标检测
* **数据集:** COCO
* **简要说明:** DETR 是一种基于 Transformer 的端到端目标检测模型，它摒弃了传统目标检测中的手工设计组件（如锚框、非极大值抑制等），直接将目标检测视为集合预测问题，通过双向匹配损失和并行解码实现高效检测。
* **特点:**
  * 端到端训练，无需后处理
  * 基于 Transformer 编码器-解码器架构
  * 使用匈牙利算法进行二分图匹配
  * 全局建模能力强，可捕获目标间关系
* **当前进度:** 正在设计模型架构和训练流程。

---

## 📚 学习计划 (未来)

* **多模态模型探索:**
    * **BLIP/BLIP-2:** 图像-文本多模态理解与生成
    * **LLaVA:** 大型语言-视觉助手模型
    * **Flamingo:** 少样本视觉语言学习
    * **ImageBind:** 多模态嵌入统一
    * **CoCa:** 对比性跨模态预训练

* **Grounding 相关任务:**
    * **GLIP:** 基于短语的目标检测
    * **Grounding DINO:** 开放词汇目标检测
    * **Referring Expression Segmentation:** 指代表达分割
    * **Visual Grounding:** 视觉定位任务
    * **MDETR:** 端到端调制检测器

* **检测与分割模型:**
    * **DETR/Deformable DETR:** 端到端目标检测
    * **Mask R-CNN/Mask2Former:** 实例分割
    * **SAM (Segment Anything Model):** 通用分割模型
    * **YOLO 系列 (v5-v8):** 实时目标检测
    * **OneFormer:** 统一图像分割

* **增量学习方法:**
    * **知识蒸馏 (Knowledge Distillation):** 模型压缩与知识传递
    * **持续学习 (Continual Learning):** 防止灾难性遗忘
    * **少样本学习 (Few-shot Learning):** 从少量样本中学习
    * **对比学习 (Contrastive Learning):** 自监督表示学习
    * **元学习 (Meta-Learning):** 学会如何学习

* **高级 PyTorch 特性:**
    * `torch.distributed` (分布式训练)
    * `torch.jit` (JIT 编译)
    * `torch.quantization` (模型量化)
    * `torch.fx` (图形转换)
    * `torch.compile` (编译优化)

* **部署实践:**
    * ONNX 导出与优化
    * TorchScript 序列化
    * TensorRT 加速
    * 移动端部署 (Android/iOS)
    * 云端服务部署

---

## 🛠️ 环境配置

1.  **克隆仓库:**

    ```bash
    git clone https://github.com/liuxiang09/Pytorch.git
    cd PyTorch_Training
    ```
2.  **创建 Conda 环境 (推荐):**
    
    ```bash
    conda create -n pytorch_env python=3.10
    conda activate pytorch_env
    ```
3.  **安装依赖:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **安装 PyTorch (根据你的CUDA版本):**
    请访问 [PyTorch 官网](https://pytorch.org/get-started/locally/) 获取适合你系统和 CUDA 版本的安装命令。
    例如 (CUDA 11.8):
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```

---

## 💡 使用指南 (示例)

### 模型训练

```bash
python models/<模型名称>/train.py --epochs 20 --batch_size 32 --lr 0.001 --device cuda
```

### 模型评估

```bash
python models/<模型名称>/eval.py --checkpoint checkpoints/<模型名称>_model.pth --batch_size 64 --device cuda
```

### 模型推理

```bash
python models/<模型名称>/inference.py --checkpoint checkpoints/<模型名称>_model.pth --input_data <输入数据路径> --output_dir <输出目录>
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