# DETR (DEtection TRansformer) 训练框架

这是一个基于 PyTorch 实现的 DETR 目标检测模型训练框架，使用 Pascal VOC 数据集进行训练。

## 模型结构

DETR 是一个端到端的目标检测模型，主要包含以下组件：

- **Backbone**: ResNet-50 特征提取网络
- **PositionEmbedding**: 位置嵌入，用于编码图像中对象的位置
- **Transformer**: 编码器-解码器架构，处理图像特征和对象查询
- **Matcher**: 用于预测结果与真实标签的最优匹配
- **SetCriterion**: 包含分类损失、L1 损失和 GIoU 损失
- **Detr**: 整体模型类，集成了上述组件

## 数据准备

### Pascal VOC 数据集

1. 下载 Pascal VOC 2012 数据集：

   ```bash
   # 从Kaggle下载
   wget https://www.kaggle.com/api/v1/datasets/download/gopalbhattrai/pascal-voc-2012-dataset
   ```

2. 解压到指定目录：

   ```bash
    unzip archive.zip -d ./data && mv ./data/archive ./data/Pascal_VOC
   ```

3. 确保目录结构如下：

   ```
   data/Pascal_VOC/
   ├── VOC2012_train_val/
   │   └── VOC2012_train_val/
   │       ├── Annotations/     # XML标注文件
   │       ├── ImageSets/       # 数据集划分文件
   │       ├── JPEGImages/      # 图像文件
   │       ├── SegmentationClass/  # 分割标注文件
   │       └── SegmentationObject/ # 分割标注文件
   │
   │
   │
   └── VOC2012_test/
       └── VOC2012_test/
           ├── Annotations/     # XML标注文件
           ├── ImageSets/       # 数据集划分文件
           └── JPEGImages/      # 图像文件

   ```

## 使用方法

### 1. 测试环境

首先运行测试脚本确保所有组件正常工作：

```bash
python models/DETR/TEST/test_detr.py
```

### 2. 训练模型

#### 训练模型

```bash
python models/DETR/train.py --train --eval --data_dir data/Pascal_VOC/VOC2012_train_val/VOC2012_train_val --save_dir models/DETR/checkpoints --num_epochs 10
```

### 3. 评估模型

```bash
python models/DETR/eval.py --checkpoint checkpoints/best_model.pth --data_dir data/Pascal_VOC/VOC2012_test/VOC2012_test --confidence_threshold 0.5 --num_samples 10
```

评估脚本会：

- 在验证集上运行模型
- 生成检测结果的可视化图像
- 计算平均检测数量等统计信息
- 将结果保存到 `eval_results/` 目录

### 4. 恢复训练

```bash
python models/DETR/train.py --train --resume models/DETR/checkpoints/checkpoint_epoch_10.pth --num_epochs 50
```

## 训练监控

### 日志文件

训练日志保存在 `logs/` 目录中，包含：

- 每个 epoch 的平均损失
- 详细的损失组成（分类损失、边界框损失、GIoU 损失）
- 学习率变化
- 训练时间统计

### 检查点文件

模型检查点保存在 `checkpoints/` 目录中：

- `best_model.pth`: 验证集上最佳模型
- `final_model.pth`: 最终训练模型
- `checkpoint_epoch_X.pth`: 定期保存的检查点

### 模型改进

- 尝试不同的 backbone 网络（ResNet-101, EfficientNet 等）
- 调整 Transformer 的层数和维度
- 实现不同的损失函数组合

## 参考文献

1. [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)
2. [DETR Official Implementation](https://github.com/facebookresearch/detr)
3. [Pascal VOC Dataset](http://host.robots.ox.ac.uk/pascal/VOC/)
