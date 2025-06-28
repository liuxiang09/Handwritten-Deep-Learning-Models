import sys
import os
import torch

# 添加项目根目录到路径，以便导入模块
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_root)

from models.DETR.model.backbone import Backbone
from models.DETR.utils.utils import NestedTensor

# 全局参数配置
# Backbone参数
BACKBONE_NAME = "resnet50"
TRAIN_BACKBONE = True
RETURN_INTERM_LAYERS = False
RETURN_INTERM_LAYERS_TEST = True

# 测试数据参数
BATCH_SIZE = 8
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
NUM_CHANNELS = 3


def test_backbone():
    print("开始测试 Backbone 模型...")
    
    # 创建标准的backbone实例
    backbone = Backbone(name=BACKBONE_NAME, train_backbone=TRAIN_BACKBONE, return_interm_layers=RETURN_INTERM_LAYERS)
    print("创建标准 Backbone 实例成功")
    
    # 创建返回中间层的backbone实例
    backbone_interm = Backbone(name=BACKBONE_NAME, train_backbone=TRAIN_BACKBONE, return_interm_layers=RETURN_INTERM_LAYERS_TEST)
    print("创建返回中间层的 Backbone 实例成功")
    
    # 准备输入数据 - 使用NestedTensor
    tensors = torch.rand(BATCH_SIZE, NUM_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)
    # 创建掩码，假设没有填充区域
    masks = torch.zeros(BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, dtype=torch.bool)
    nested_tensor = NestedTensor(tensors=tensors, mask=masks)
    print(f"输入NestedTensor - tensors形状: {nested_tensor.tensors.shape}, mask形状: {nested_tensor.mask.shape}")
    
    # 测试标准backbone
    features = backbone(nested_tensor)
    for key, value in features.items():
        print(f"标准 Backbone 输出特征: {key} {value.shape}")
    
    # 测试返回中间层的backbone
    features_interm = backbone_interm(nested_tensor)
    print("返回中间层的 Backbone 输出:")
    for key, value in features_interm.items():
        print(f"  层 {key}: 形状 {value.shape}")
    
    # 检查参数冻结状态
    trainable_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    non_trainable_params = sum(p.numel() for p in backbone.parameters() if not p.requires_grad)
    print(f"标准 Backbone 训练参数: {trainable_params:,}")
    print(f"标准 Backbone 冻结参数: {non_trainable_params:,}")
    print(f"标准 Backbone 总参数: {trainable_params + non_trainable_params:,}\n")

    print("Backbone 测试完成！")


if __name__ == "__main__":
    test_backbone()
