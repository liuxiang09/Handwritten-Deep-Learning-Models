import torch

from models.SAM.model.prompt_encoder import PositionEmbeddingRandom

def test_positional_encoding_random():
    pe = PositionEmbeddingRandom()
    coords = torch.rand(2, 5, 2)  # (B, N, 2)
    image_size = (128, 128)
    output_grid = pe.forward(image_size)
    output_coord = pe.forward_with_coords(coords, image_size)
    print(f"Output grid shape: {output_grid.shape}")
    print(f"Output coord shape: {output_coord.shape}")


if __name__ == "__main__":
    test_positional_encoding_random()