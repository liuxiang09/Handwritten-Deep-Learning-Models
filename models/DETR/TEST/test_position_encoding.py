import torch
from models.DETR.model.position_encoding import PositionEmbeddingSine, PositionEmbeddingLearned
from models.DETR.utils.utils import NestedTensor

# 全局参数配置
# 位置编码参数
D_MODEL = 256
NORMALIZE = True

# 测试数据参数
BATCH_SIZE = 32
HEIGHT = 30
WIDTH = 40
NUM_CHANNELS = 3
MASK_HEIGHT = 10
MASK_WIDTH = 20


def test_position_encoding():
    """
    单流程、精简版的位置编码测试
    """
    print("\n🚀 开始精简版位置编码测试...")
    
    # 1. 准备输入数据 - 使用NestedTensor
    tensors = torch.rand(BATCH_SIZE, NUM_CHANNELS, HEIGHT, WIDTH)
    masks = torch.zeros(BATCH_SIZE, HEIGHT, WIDTH, dtype=torch.bool)
    masks[:, :MASK_HEIGHT, :MASK_WIDTH] = True
    nested_tensor = NestedTensor(tensors=tensors, mask=masks)
    print(f"输入NestedTensor已创建: tensors shape={nested_tensor.tensors.shape}, mask shape={nested_tensor.mask.shape}")

    # --- 测试正弦位置编码 (PositionEmbeddingSine) ---
    print("\n--- 1. 测试正弦位置编码 ---")
    # 注意: DETR中x和y编码维度各为 num_pos_feats/2，然后拼接
    pos_enc_sine = PositionEmbeddingSine(num_pos_feats=D_MODEL // 2, normalize=NORMALIZE)
    pos_sine = pos_enc_sine(nested_tensor)
    
    # 验证
    expected_shape = (BATCH_SIZE, D_MODEL, HEIGHT, WIDTH)
    assert pos_sine.shape == expected_shape, f"[Sine] 形状断言失败! Got {pos_sine.shape}, Expected {expected_shape}"
    assert pos_sine.dtype == torch.float32, f"[Sine] 数据类型应为 float32, Got {pos_sine.dtype}"
    assert not torch.isnan(pos_sine).any() and not torch.isinf(pos_sine).any(), "[Sine] 输出包含 NaN 或 Inf!"
    
    print(f"✅ [Sine] 实例创建成功")
    print(f"✅ [Sine] 输出形状正确: {pos_sine.shape}")
    print(f"✅ [Sine] 数据类型正确: {pos_sine.dtype}")
    print(f"✅ [Sine] 数值有效 (无NaN/Inf)")

    # --- 测试可学习位置编码 (PositionEmbeddingLearned) ---
    print("\n--- 2. 测试可学习位置编码 ---")
    pos_enc_learned = PositionEmbeddingLearned(num_pos_feats=D_MODEL // 2)
    # 注意: 可学习编码的输入也应该是整个 NestedTensor，以便模块内部获取H和W
    pos_learned = pos_enc_learned(nested_tensor)
    
    # 验证
    assert pos_learned.shape == expected_shape, f"[Learned] 形状断言失败! Got {pos_learned.shape}, Expected {expected_shape}"
    assert pos_learned.dtype == torch.float32, f"[Learned] 数据类型应为 float32, Got {pos_learned.dtype}"
    
    # 验证其参数是否“可学习”
    has_learnable_params = any(p.requires_grad for p in pos_enc_learned.parameters())
    assert has_learnable_params, "[Learned] 模型中没有可学习的参数 (requires_grad=False)"

    print(f"✅ [Learned] 实例创建成功")
    print(f"✅ [Learned] 输出形状正确: {pos_learned.shape}")
    print(f"✅ [Learned] 数据类型正确: {pos_learned.dtype}")
    print(f"✅ [Learned] 包含可学习参数 (requires_grad=True)")
    print(f"✅ [Learned] 可学习参数数量: {sum(p.numel() for p in pos_enc_learned.parameters() if p.requires_grad)}")

    print("\n🎉 所有核心测试断言通过！")


if __name__ == "__main__":
    test_position_encoding()
