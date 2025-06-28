import sys
import os
import torch

# 添加项目根目录到路径，以便导入模块
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_root)

from models.DETR.model.transformer import Transformer

# 全局参数配置
# Transformer参数
D_MODEL = 256
NHEAD = 8
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6
DIM_FEEDFORWARD = 2048
RETURN_INTERMEDIATE_DEC = False

# 测试数据参数
BATCH_SIZE = 2
HEIGHT = 30
WIDTH = 40
TGT_LEN = 100


def test_transformer():
    """测试完整Transformer模型"""
    print("开始测试 Transformer 模型...")
    
    # 创建Transformer实例
    transformer = Transformer(
        d_model=D_MODEL,
        nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        return_intermediate_dec=RETURN_INTERMEDIATE_DEC
    )
    print("创建 Transformer 实例成功")
    
    # 计算参数数量
    total_params = sum(p.numel() for p in transformer.parameters())
    print(f"Transformer 总参数: {total_params:,}")
    
    # 准备输入数据
    memory_len = HEIGHT * WIDTH
    
    src = torch.rand(BATCH_SIZE, D_MODEL, HEIGHT, WIDTH)
    mask = torch.zeros(BATCH_SIZE, HEIGHT, WIDTH, dtype=torch.bool)
    query_embed = torch.rand(TGT_LEN, D_MODEL)
    pos_embed = torch.rand(BATCH_SIZE, D_MODEL, HEIGHT, WIDTH)
    
    print(f"源张量形状: {src.shape}")
    print(f"查询嵌入形状: {query_embed.shape}")
    print(f"位置嵌入形状: {pos_embed.shape}")
    print(f"掩码形状: {mask.shape}")
    
    # 前向传播
    with torch.no_grad():  # 节省内存
        hs, memory = transformer(src, mask, query_embed, pos_embed)
    
    print(f"解码器输出形状: {hs.shape}")
    print(f"编码器输出形状: {memory.shape}")
    
    # 验证输出维度合理性
    assert len(hs.shape) == 4, f"解码器输出应该是4维，实际得到 {len(hs.shape)} 维"
    assert len(memory.shape) == 4, f"编码器输出应该是4维，实际得到 {len(memory.shape)} 维"
    
    # 验证包含正确的维度
    assert TGT_LEN in hs.shape, f"解码器输出应该包含查询数量 {TGT_LEN}"
    assert BATCH_SIZE in hs.shape, f"解码器输出应该包含批次大小 {BATCH_SIZE}"
    assert D_MODEL in hs.shape, f"解码器输出应该包含特征维度 {D_MODEL}"
    
    print("Transformer 模型测试成功！")


if __name__ == "__main__":
    test_transformer()
