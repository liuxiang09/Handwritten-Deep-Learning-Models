# DETR (DEtection TRansformer) 训练框架

这是一个基于PyTorch实现的DETR目标检测模型训练框架，使用Pascal VOC数据集进行训练。

## 模型结构

DETR是一个端到端的目标检测模型，主要包含以下组件：

- **Backbone**: ResNet-50 特征提取网络
- **Transformer**: 编码器-解码器架构，处理图像特征和对象查询
- **预测头**: 分类头和回归头，输出类别概率和边界框坐标
- **匈牙利匹配器**: 用于预测结果与真实标签的最优匹配
- **损失函数**: 包含分类损失、L1损失和GIoU损失

## 目录结构

```
models/DETR/
├── model/
│   ├── backbone.py          # 特征提取网络
│   ├── transformer.py       # Transformer编码器和解码器
│   ├── detr.py             # DETR主模型
│   ├── matcher.py          # 匈牙利匹配器
│   ├── criterion.py        # 损失函数
│   └── position_encoding.py # 位置编码
├── utils/
│   ├── dataset.py          # Pascal VOC数据集加载器
│   └── utils.py            # 工具函数
├── checkpoints/            # 模型检查点保存目录
├── logs/                   # 训练日志目录
├── eval_results/           # 评估结果保存目录
├── train.py               # 训练脚本
├── eval.py                # 评估脚本
└── README.md              # 说明文档
```

## 数据准备

### Pascal VOC 数据集

1. 下载Pascal VOC 2012数据集：
   ```bash
   # 从Kaggle下载
   wget https://www.kaggle.com/api/v1/datasets/download/gopalbhattrai/pascal-voc-2012-dataset
   ```

2. 解压到指定目录：
   ```bash
   unzip pascal-voc-2012-dataset.zip -d /home/hpc/Desktop/Pytorch/data/Pascal_VOC/
   ```

3. 确保目录结构如下：
   ```
   data/Pascal_VOC/
   └── VOC2012_train_val/
       └── VOC2012_train_val/
           ├── Annotations/     # XML标注文件
           ├── ImageSets/       # 数据集划分文件
           └── JPEGImages/      # 图像文件
   ```

### 数据集类别

Pascal VOC包含20个目标类别：
- 交通工具: aeroplane, bicycle, boat, bus, car, motorbike, train
- 动物: bird, cat, cow, dog, horse, sheep
- 人: person
- 室内物品: bottle, chair, diningtable, pottedplant, sofa, tvmonitor

## 使用方法

### 1. 测试环境

首先运行测试脚本确保所有组件正常工作：

```bash
cd /home/hpc/Desktop/Pytorch/models/DETR
python test_detr.py
```

### 2. 训练模型

#### 基本训练命令

```bash
python train.py --train --eval \
    --data_dir /home/hpc/Desktop/Pytorch/data/Pascal_VOC \
    --batch_size 4 \
    --num_epochs 50 \
    --learning_rate 1e-4
```

#### 完整参数说明

**路径参数:**
- `--data_dir`: Pascal VOC数据集目录
- `--save_dir`: 模型检查点保存目录
- `--log_dir`: 训练日志保存目录

**训练参数:**
- `--batch_size`: 批次大小 (默认: 4)
- `--num_epochs`: 训练轮数 (默认: 50)
- `--learning_rate`: 学习率 (默认: 1e-4)
- `--weight_decay`: 权重衰减 (默认: 1e-4)
- `--num_workers`: 数据加载进程数 (默认: 4)

**模型参数:**
- `--num_queries`: 对象查询数量 (默认: 100)
- `--hidden_dim`: Transformer隐藏维度 (默认: 256)
- `--nheads`: 注意力头数 (默认: 8)
- `--num_encoder_layers`: 编码器层数 (默认: 6)
- `--num_decoder_layers`: 解码器层数 (默认: 6)

**其他参数:**
- `--train`: 执行训练
- `--eval`: 执行验证
- `--resume`: 从检查点恢复训练
- `--max_size`: 图像最大尺寸 (默认: 800)

### 3. 评估模型

```bash
python eval.py \
    --checkpoint checkpoints/best_model.pth \
    --data_dir /home/hpc/Desktop/Pytorch/data/Pascal_VOC \
    --confidence_threshold 0.5 \
    --num_samples 10
```

评估脚本会：
- 在验证集上运行模型
- 生成检测结果的可视化图像
- 计算平均检测数量等统计信息
- 将结果保存到 `eval_results/` 目录

### 4. 恢复训练

```bash
python train.py --train --eval \
    --resume checkpoints/checkpoint_epoch_10.pth \
    --num_epochs 50
```

## 训练监控

### 日志文件

训练日志保存在 `logs/` 目录中，包含：
- 每个epoch的平均损失
- 详细的损失组成（分类损失、边界框损失、GIoU损失）
- 学习率变化
- 训练时间统计

### 检查点文件

模型检查点保存在 `checkpoints/` 目录中：
- `best_model.pth`: 验证集上最佳模型
- `final_model.pth`: 最终训练模型
- `checkpoint_epoch_X.pth`: 定期保存的检查点

### 模型改进

- 尝试不同的backbone网络（ResNet-101, EfficientNet等）
- 调整Transformer的层数和维度
- 实现不同的损失函数组合

## 参考文献

1. [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)
2. [DETR Official Implementation](https://github.com/facebookresearch/detr)
3. [Pascal VOC Dataset](http://host.robots.ox.ac.uk/pascal/VOC/)
