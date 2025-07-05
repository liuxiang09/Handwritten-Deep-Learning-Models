import torch
import numpy as np
import torch.nn as nn
from typing import Tuple, Type
from .common import LayerNorm2d

class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding for 2D images, using random values.
    """
    def __init__(self,
                 num_pos_feats: int = 128,
                 scale: float = 1.0):
        
        super().__init__()
        # Sample a fixed, non-learned random matrix from a Gaussian distribution
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)), # [2, num_pos_feats]
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Positionally encode points that are normalized to [0,1].
        Args:
            coords: (B, N, 2) or (H, W, 2) coordinates in [0, 1] range
        Returns:
            pe: (B, N, 2 * num_pos_feats) positional encoding
        """
        coords = 2 * coords - 1 # normalize to [-1, 1]  
        # use gaussian matrix to compute positional encoding
        coords = coords @ self.positional_encoding_gaussian_matrix # [B, N, 2] @ [2, num_pos_feats] -> [B, N, num_pos_feats]
        coords = 2 * np.pi * coords  # scale the coordinates
        # adopt a sinusoidal encoding scheme
        pe = torch.cat((coords.sin(), coords.cos()), dim=-1) # [B, N, 2 * num_pos_feats]
        return pe
    
    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """
        Generate positional encoding for a grid of the specified size.
        Args:
            size: (H, W) size of the coordinates grid
        Returns:
            pe: (H, W, 2 * num_pos_feats) positional encoding
        """
        h, w = size
        device = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5  # [H, W]
        x_embed = grid.cumsum(dim=1) - 0.5  # [H, W]
        y_embed = y_embed / h
        x_embed = x_embed / w
        # (H, W) -> (H, W, 2) -> (H * W, 2)
        coords = torch.stack((x_embed, y_embed), dim=-1)
        coords = self._pe_encoding(coords).permute(2, 0, 1)  # [2 * num_pos_feats, H, W]

        return coords
    
    def forward_with_coords(self, 
                            coords_input: torch.Tensor,
                            image_size: Tuple[int, int]) -> torch.Tensor:
        """
        Positionally encode points that are not normalized to [0,1].
        Args:
            coords_input: (B, N, 2)
            image_size: (H, W) size of the image
        Returns:
            pe: (B, N, 2 * num_pos_feats) positional encoding
        """
        # normalize into [0, 1]
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1] # normalize x
        coords[:, :, 1] = coords[:, :, 1] / image_size[0] # normalize y
        return self._pe_encoding(coords.to(torch.float)) # [B, N, 2 * num_pos_feats]
    
class PromptEncoder(nn.Module):
    
    def __init__(self,
                 embed_dim: int,
                 image_embedding_size: Tuple[int, int],
                 input_image_size: Tuple[int, int],
                 mask_in_channels: int,
                 activation: Type[nn.Module] = nn.GELU):
        """
        Encodes prompts for input to SAM's mask decoder.

        Args:
          embed_dim (int): The prompts' embedding dimension
          image_embedding_size (tuple(int, int)): The spatial size of the image embedding, as (H, W).
          input_image_size (tuple(int, int)): The padded size of the image as input to the image encoder, as (H, W).
          mask_in_chans (int): The number of hidden channels used for encoding input masks.
          activation (nn.Module): The activation to use when encoding input masks
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.image_embedding_size = image_embedding_size
        self.input_image_size = input_image_size
        self.pe_layer = PositionEmbeddingRandom(num_pos_feats=embed_dim // 2)

        self.num_point_embeddings = 4 # pos/neg point + 2 box corners
        point_embeddings = [nn.Embedding(1, embed_dim) for _ in range(self.num_point_embeddings)]
        self.point_embeddings = nn.ModuleList(point_embeddings)
        self.not_a_point_embedding = nn.Embedding(1, embed_dim)

        self.mask_input_size = (4 * image_embedding_size[0], 4 * image_embedding_size[1]) # [H, W] -> [4H, 4W]
        # downscale mask input to match the image embedding size
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_channels // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_channels // 4, eps=1e-6),
            activation(),
            nn.Conv2d(mask_in_channels // 4, mask_in_channels, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_channels, eps=1e-6),
            activation(),
            nn.Conv2d(mask_in_channels, embed_dim, kernel_size=1)
        )
        self.no_mask_embedding = nn.Embedding(1, embed_dim)

    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape [1, 2 * embed_dim, H, W]
        """
        return self.pe_layer.forward(self.image_embedding_size).unsqueeze(0)\
        
    def _embed_points(self,
                      points: torch.Tensor,
                      labels: torch.Tensor,
                      pad: bool) -> torch.Tensor:
        """
        Embeds point prompts.
        Args:
            points (torch.Tensor): Tensor of shape (B, N, 2) containing point coordinates.
            labels (torch.Tensor): Tensor of shape (B, N) containing point labels.
            pad (bool): Whether to pad the points and labels with a zero point and -1 label.
        """
        points = points + 0.5 # shift to center of pixel
        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)