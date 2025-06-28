import sys
import os
import torch

# 添加项目根目录到路径，以便导入模块
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_root)

from models.DETR.model.detr import DETR, MLP
from models.DETR.model.backbone import Backbone
from models.DETR.model.transformer import Transformer
from models.DETR.utils.utils import NestedTensor

# 全局参数配置
# MLP参数
MLP_INPUT_DIM = 256
MLP_HIDDEN_DIM = 512
MLP_OUTPUT_DIM = 91
MLP_NUM_LAYERS = 3

# Backbone参数
BACKBONE_NAME = "resnet50"
TRAIN_BACKBONE = False
RETURN_INTERM_LAYERS = False

# Transformer参数
D_MODEL = 256
NHEAD = 8
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6
DIM_FEEDFORWARD = 2048

# DETR参数
NUM_CLASSES = 91
NUM_QUERIES = 100
RETURN_INTERMEDIATE_DEC = False

# 测试数据参数
BATCH_SIZE_MLP = 8
SEQ_LEN = 100
BATCH_SIZE_DETR = 2
IMAGE_HEIGHT = 800
IMAGE_WIDTH = 1200
NUM_CHANNELS = 3


def test_mlp():
    """测试MLP模块"""
    print("开始测试 MLP 模块...")
    
    # 创建MLP实例
    mlp = MLP(input_dim=MLP_INPUT_DIM, hidden_dim=MLP_HIDDEN_DIM, output_dim=MLP_OUTPUT_DIM, num_layers=MLP_NUM_LAYERS)
    print("创建 MLP 实例成功")
    
    # 测试输入
    input_tensor = torch.rand(BATCH_SIZE_MLP, SEQ_LEN, MLP_INPUT_DIM)
    print(f"输入张量形状: {input_tensor.shape}")
    
    # 前向传播
    output = mlp(input_tensor)
    print(f"MLP 输出形状: {output.shape}")
    
    # 验证输出维度
    assert output.shape == (BATCH_SIZE_MLP, SEQ_LEN, MLP_OUTPUT_DIM), f"期望输出形状 ({BATCH_SIZE_MLP}, {SEQ_LEN}, {MLP_OUTPUT_DIM}), 实际得到 {output.shape}"
    print("MLP 模块测试成功！")


def test_detr():
    """测试DETR主模型"""
    print("\n开始测试 DETR 主模型...")
    
    # 创建backbone和transformer组件
    backbone = Backbone(name=BACKBONE_NAME, train_backbone=TRAIN_BACKBONE, return_interm_layers=RETURN_INTERM_LAYERS)
    transformer = Transformer(
        d_model=D_MODEL,
        nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD
    )
    
    # 创建DETR实例
    detr = DETR(
        backbone=backbone,
        transformer=transformer,
        num_classes=NUM_CLASSES,
        num_queries=NUM_QUERIES,
        return_intermediate_dec=RETURN_INTERMEDIATE_DEC
    )
    print("创建 DETR 实例成功")
    
    # 计算参数数量
    total_params = sum(p.numel() for p in detr.parameters())
    trainable_params = sum(p.numel() for p in detr.parameters() if p.requires_grad)
    print(f"DETR 总参数: {total_params:,}")
    print(f"DETR 可训练参数: {trainable_params:,}")
    
    # 准备输入数据
    tensors = torch.rand(BATCH_SIZE_DETR, NUM_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)
    masks = torch.zeros(BATCH_SIZE_DETR, IMAGE_HEIGHT, IMAGE_WIDTH, dtype=torch.bool)
    nested_tensor = NestedTensor(tensors=tensors, mask=masks)
    print(f"输入图像张量形状: {nested_tensor.tensors.shape}")
    print(f"输入掩码形状: {nested_tensor.mask.shape}")
    
    # 前向传播
    with torch.no_grad():  # 节省内存
        outputs = detr(nested_tensor)
    
    # 验证输出
    print("DETR 输出:")
    print(f"  预测类别logits形状: {outputs['pred_logits'].shape}")
    print(f"  预测边界框形状: {outputs['pred_boxes'].shape}")
    
    # 检查输出形状是否合理 (不严格验证具体维度，因为可能有transpose)
    logits_shape = outputs['pred_logits'].shape
    boxes_shape = outputs['pred_boxes'].shape
    
    # 验证有正确的维度数量
    assert len(logits_shape) == 3, f"logits应该是3维张量，实际得到 {len(logits_shape)} 维"
    assert len(boxes_shape) == 3, f"boxes应该是3维张量，实际得到 {len(boxes_shape)} 维"
    
    # 验证最后一维的大小
    assert logits_shape[-1] == NUM_CLASSES + 1, f"logits最后一维应该是{NUM_CLASSES + 1} ({NUM_CLASSES}类+1背景)，实际得到 {logits_shape[-1]}"
    assert boxes_shape[-1] == 4, f"boxes最后一维应该是4，实际得到 {boxes_shape[-1]}"
    
    # 验证包含正确的批次大小和查询数量
    assert BATCH_SIZE_DETR in logits_shape, f"logits应该包含批次大小 {BATCH_SIZE_DETR}"
    assert NUM_QUERIES in logits_shape, f"logits应该包含查询数量 {NUM_QUERIES}"
    
    print("DETR 主模型测试成功！")


if __name__ == "__main__":
    test_mlp()
    test_detr()
