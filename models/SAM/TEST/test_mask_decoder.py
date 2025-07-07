import torch
import torch.nn as nn

from models.SAM.model.mask_decoder import MaskDecoder
from models.SAM.model.transformer import TwoWayTransformer


def test_mask_decoder():
    # 定义参数
    transformer_dim = 256
    num_multimask_outputs = 3
    
    # 创建Transformer模块
    transformer = TwoWayTransformer(
        depth=2,
        embedding_dim=transformer_dim,
        num_heads=8,
        mlp_dim=2048,
        attention_downsample_rate=2
    )
    
    # 创建MaskDecoder模块
    mask_decoder = MaskDecoder(
        transformer_dim=transformer_dim,
        transformer=transformer,
        num_multimask_outputs=num_multimask_outputs,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
    )
    
    # 创建模拟输入
    batch_size = 2
    embed_size = 64  # 假设嵌入大小为64x64
    
    # 创建图像嵌入和位置编码
    image_embeddings = torch.randn(1, transformer_dim, embed_size, embed_size)
    image_pe = torch.randn(1, transformer_dim, embed_size, embed_size)
    
    # 创建稀疏和密集提示嵌入
    # 假设每个批次有5个点/框
    num_points = 5
    sparse_prompt_embeddings = torch.randn(batch_size, num_points, transformer_dim)
    dense_prompt_embeddings = torch.randn(batch_size, transformer_dim, embed_size, embed_size)
    
    # 测试单个掩码输出
    print("测试单个掩码输出:")
    masks_single, iou_pred_single = mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=image_pe,
        sparse_prompt_embeddings=sparse_prompt_embeddings,
        dense_prompt_embeddings=dense_prompt_embeddings,
        multimask_output=False
    )
    
    print(f"单个掩码输出形状: {masks_single.shape}")
    print(f"单个IoU预测形状: {iou_pred_single.shape}")
    
    # 测试多个掩码输出
    print("\n测试多个掩码输出:")
    masks_multi, iou_pred_multi = mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=image_pe,
        sparse_prompt_embeddings=sparse_prompt_embeddings,
        dense_prompt_embeddings=dense_prompt_embeddings,
        multimask_output=True
    )
    
    print(f"多个掩码输出形状: {masks_multi.shape}")
    print(f"多个IoU预测形状: {iou_pred_multi.shape}")


if __name__ == "__main__":
    test_mask_decoder()
