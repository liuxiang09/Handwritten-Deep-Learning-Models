import torch
import torch.nn as nn

from models.SAM.model.prompt_encoder import PositionEmbeddingRandom, PromptEncoder



# 测试 prompt_encoder.py 中的 _embed_points 函数
def test_embed_points():
    embed_dim = 256
    input_image_size = (1024, 1024)
    prompt_encoder = PromptEncoder(embed_dim, (64, 64), input_image_size, 16)

    # 模拟一些点和标签
    points = torch.tensor([[[100, 200], [300, 400]], [[500, 600], [700, 800]]], dtype=torch.float32)
    labels = torch.tensor([[0, 1], [1, -1]], dtype=torch.int64)

    # 调用 _embed_points 方法
    # 0 代表背景，1 代表前景，-1 代表填充
    point_embeddings = prompt_encoder._embed_points(points, labels, pad=True)

    print(f"Point embeddings: {point_embeddings}")
    print(f"Point embeddings shape: {point_embeddings.shape}")

# 测试 prompt_encoder.py 中的 _embed_boxes 函数
def test_embed_boxes():
    embed_dim = 256
    input_image_size = (1024, 1024)
    prompt_encoder = PromptEncoder(embed_dim, (64, 64), input_image_size, 16)

    # 模拟一些边界框
    boxes = torch.tensor([[[100, 200, 300, 400], [500, 600, 700, 800]], [[120, 220, 320, 420], [0, 0, 0, 0]]], dtype=torch.float32)

    # 调用 _embed_boxes 方法
    box_embeddings = prompt_encoder._embed_boxes(boxes)

    print(f"Box embeddings: {box_embeddings}")
    print(f"Box embeddings shape: {box_embeddings.shape}")

# 测试 prompt_encoder.py 中的 _embed_masks 函数
def test_embed_masks():
    embed_dim = 256
    input_image_size = (1024, 1024)
    prompt_encoder = PromptEncoder(embed_dim, (64, 64), input_image_size, 16)

    # 模拟一些掩码
    masks = torch.randn(2, 1, 4 * 64, 4 * 64)  # 批次大小为2，单通道掩码

    # 调用 _embed_masks 方法
    mask_embeddings = prompt_encoder._embed_masks(masks)

    print(f"Mask embeddings: {mask_embeddings}")
    print(f"Mask embeddings shape: {mask_embeddings.shape}")

# 测试 PromptEncoder 的前向传播
def test_forward():
    embed_dim = 256
    image_embedding_size = (64, 64)  # 图像嵌入网格的大小
    input_image_size = (1024, 1024)
    prompt_encoder = PromptEncoder(embed_dim, image_embedding_size, input_image_size, 16)

    # 模拟一些点和标签
    points = torch.tensor([[[100, 200], [300, 400]], [[500, 600], [700, 800]]], dtype=torch.float32)
    labels = torch.tensor([[0, 1], [1, 1]], dtype=torch.int64)
    points = (points, labels)
    # 模拟一些边界框
    boxes = torch.tensor([[[100, 200, 300, 400]], [[220, 320, 420, 520]]], dtype=torch.float32)

    # 模拟一些掩码
    masks = torch.randn(2, 1, 4 * 64, 4 * 64)  # 批次大小为2

    # 调用前向传播方法
    outputs = prompt_encoder(points=points, boxes=boxes, masks=masks)

    print(f"Outputs: {outputs}")
    print(f"sparse_embeddings shape: {outputs[0].shape}")
    print(f"dense_embeddings shape: {outputs[1].shape}")

if __name__ == "__main__":
    # test_embed_points()
    # test_embed_boxes()
    # test_embed_masks()
    test_forward()