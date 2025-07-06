import torch
import torch.nn as nn

from models.SAM.model.prompt_encoder import PositionEmbeddingRandom, PromptEncoder



# test _embed_points function in prompt_encoder.py
def test_embed_points():
    embed_dim = 256
    input_image_size = (1024, 1024)
    prompt_encoder = PromptEncoder(embed_dim, (64, 64), input_image_size, 16)

    # Simulate some points and labels
    points = torch.tensor([[[100, 200], [300, 400]], [[500, 600], [700, 800]]], dtype=torch.float32)
    labels = torch.tensor([[0, 1], [1, -1]], dtype=torch.int64)

    # Call the _embed_points method
    # 0 represents background, 1 represents foreground, -1 represents padding
    point_embeddings = prompt_encoder._embed_points(points, labels, pad=True)

    print(f"Point embeddings: {point_embeddings}")
    print(f"Point embeddings shape: {point_embeddings.shape}")

# test _embed_boxes function in prompt_encoder.py
def test_embed_boxes():
    embed_dim = 256
    input_image_size = (1024, 1024)
    prompt_encoder = PromptEncoder(embed_dim, (64, 64), input_image_size, 16)

    # Simulate some boxes
    boxes = torch.tensor([[[100, 200, 300, 400], [500, 600, 700, 800]], [[120, 220, 320, 420], [0, 0, 0, 0]]], dtype=torch.float32)

    # Call the _embed_boxes method
    box_embeddings = prompt_encoder._embed_boxes(boxes)

    print(f"Box embeddings: {box_embeddings}")
    print(f"Box embeddings shape: {box_embeddings.shape}")

# test _embed_masks function in prompt_encoder.py
def test_embed_masks():
    embed_dim = 256
    input_image_size = (1024, 1024)
    prompt_encoder = PromptEncoder(embed_dim, (64, 64), input_image_size, 16)

    # Simulate some masks
    masks = torch.randn(2, 1, 4 * 64, 4 * 64)  # Batch size of 2, single channel mask

    # Call the _embed_masks method
    mask_embeddings = prompt_encoder._embed_masks(masks)

    print(f"Mask embeddings: {mask_embeddings}")
    print(f"Mask embeddings shape: {mask_embeddings.shape}")

# test the PromptEncoder forward
def test_forward():
    embed_dim = 256
    image_embedding_size = (64, 64)  # Size of the image embedding grid
    input_image_size = (1024, 1024)
    prompt_encoder = PromptEncoder(embed_dim, image_embedding_size, input_image_size, 16)

    # Simulate some points and labels
    points = torch.tensor([[[100, 200], [300, 400]], [[500, 600], [700, 800]]], dtype=torch.float32)
    labels = torch.tensor([[0, 1], [1, 1]], dtype=torch.int64)
    points = (points, labels)
    # Simulate some boxes
    boxes = torch.tensor([[[100, 200, 300, 400]], [[220, 320, 420, 520]]], dtype=torch.float32)

    # Simulate some masks
    masks = torch.randn(2, 1, 4 * 64, 4 * 64)  # Batch size of 2

    # Call the forward method
    outputs = prompt_encoder(points=points, boxes=boxes, masks=masks)

    print(f"Outputs: {outputs}")
    print(f"sparse_embeddings shape: {outputs[0].shape}")
    print(f"dense_embeddings shape: {outputs[1].shape}")

if __name__ == "__main__":
    # test_embed_points()
    # test_embed_boxes()
    # test_embed_masks()
    test_forward()