import sys
import os
import torch

# 添加项目根目录到路径，以便导入模块
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_root)

from models.DETR.model.position_encoding import PositionEmbeddingSine, PositionEmbeddingLearned
from models.DETR.utils.utils import NestedTensor

# 全局参数配置
# 位置编码参数
NUM_POS_FEATS = 128
NORMALIZE = True

# 测试数据参数
BATCH_SIZE = 32
HEIGHT = 30
WIDTH = 40
NUM_CHANNELS = 3
MASK_HEIGHT = 5
MASK_WIDTH = 10


def test_position_encoding():
    print("开始测试位置编码模型...")
    
    # 创建正弦位置编码实例
    pos_enc_sine = PositionEmbeddingSine(num_pos_feats=NUM_POS_FEATS, normalize=NORMALIZE)
    print("创建正弦位置编码实例成功")
    
    # 创建可学习位置编码实例
    pos_enc_learned = PositionEmbeddingLearned(num_pos_feats=NUM_POS_FEATS)
    print("创建可学习位置编码实例成功")
    
    # 准备输入数据 - 使用NestedTensor
    tensors = torch.rand(BATCH_SIZE, NUM_CHANNELS, HEIGHT, WIDTH)
    # 创建掩码，假设部分区域为填充区域
    masks = torch.zeros(BATCH_SIZE, HEIGHT, WIDTH, dtype=torch.bool)
    # 设置一些区域为填充区域（True）
    masks[:, :MASK_HEIGHT, :MASK_WIDTH] = True
    nested_tensor = NestedTensor(tensors=tensors, mask=masks)
    print(f"输入NestedTensor - tensors形状: {nested_tensor.tensors.shape}, mask形状: {nested_tensor.mask.shape}")
    
    # 测试正弦位置编码
    pos_sine = pos_enc_sine(nested_tensor)
    print(f"正弦位置编码输出形状: {pos_sine.shape}")
    
    # 测试可学习位置编码
    pos_learned = pos_enc_learned(nested_tensor.mask)
    print(f"可学习位置编码输出形状: {pos_learned.shape}")
    
    # 验证张量数据类型
    print(f"正弦位置编码数据类型: {pos_sine.dtype}")
    print(f"可学习位置编码数据类型: {pos_learned.dtype}")
    
    # 验证张量值范围
    print(f"正弦位置编码值范围: [{pos_sine.min().item():.4f}, {pos_sine.max().item():.4f}]")
    print(f"可学习位置编码值范围: [{pos_learned.min().item():.4f}, {pos_learned.max().item():.4f}]")
    
    print("\n位置编码测试完成")


if __name__ == "__main__":
    test_position_encoding()
