import torch
import numpy as np
import torch.nn as nn
from typing import Tuple, Type, Optional
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
        coords = 2 * coords - 1 # normalize from [0, 1] to [-1, 1]
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
        H, W = image_size
        coords[:, :, 0] = coords[:, :, 0] / W # normalize x
        coords[:, :, 1] = coords[:, :, 1] / H # normalize y
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
        point_embeddings = [nn.Embedding(1, embed_dim) for _ in range(self.num_point_embeddings)] # Embedding weight shape: [1, embed_dim]
        self.point_embeddings = nn.ModuleList(point_embeddings)
        self.not_a_point_embed = nn.Embedding(1, embed_dim)

        self.mask_input_size = (4 * image_embedding_size[0], 4 * image_embedding_size[1]) # [4*embed_H, 4*embed_W]
        # downscale mask input to match the image embedding size
        # e.g., [4 * 64, 4 * 64] -> [64, 64], it's up to the image encoder's embedding size
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_channels // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_channels // 4, eps=1e-6),
            activation(),
            nn.Conv2d(mask_in_channels // 4, mask_in_channels, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_channels, eps=1e-6),
            activation(),
            nn.Conv2d(mask_in_channels, embed_dim, kernel_size=1)
        )
        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape [1, 2*embed_dim, embed_H, embed_W]
        """
        return self.pe_layer.forward(self.image_embedding_size).unsqueeze(0)
        
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
        Returns:
            torch.Tensor: Embedded points of shape (B, N, embed_dim).
        """
        points = points + 0.5 # shift to center of pixel
        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)
        point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)  # [B, N, 2 * embed_dim]
        point_embedding[labels == -1] = 0.0
        point_embedding[labels == -1] += self.not_a_point_embed.weight
        point_embedding[labels == 0] += self.point_embeddings[0].weight
        point_embedding[labels == 1] += self.point_embeddings[1].weight
        return point_embedding
    
    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """
        Embeds box prompts.
        Args:
            boxes (torch.Tensor): Tensor of shape (B, N, 4) containing box coordinates.
        Returns:
            torch.Tensor: Embedded boxes of shape (B * N, 2, embed_dim).
        """
        boxes = boxes + 0.5 # Shift to center of pixel
        coords = boxes.reshape(-1, 2, 2)  # [B * N, 2, 2]
        corner_embedding = self.pe_layer.forward_with_coords(coords, self.input_image_size)
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight # the left top corner
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight # the right bottom corner
        return corner_embedding
    
    def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """
        Embeds mask prompts.
        Args:
            masks (torch.Tensor): Tensor of shape (B, 1, 4*embed_H, 4*embed_W) containing binary masks.
        Returns:
            torch.Tensor: Embedded masks of shape (B, embed_dim, embed_H, embed_W).
        """
        mask_embedding = self.mask_downscaling(masks)
        return mask_embedding
    
    def _get_batch_size(self,
                        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
                        boxes: Optional[torch.Tensor],
                        masks: Optional[torch.Tensor]) -> int:
        """
        Returns the batch size of the input tensors.
        Args:
            points (Optional[Tuple[torch.Tensor, torch.Tensor]]): Point coordinates and labels.
            boxes (Optional[torch.Tensor]): Box coordinates.
            masks (Optional[torch.Tensor]): Mask tensors.
        Returns:
            int: Batch size.
        """
        if points is not None:
            return points[0].shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        else:
            return 1
    
    def _get_device(self) -> torch.device:
        return self.point_embeddings[0].weight.device
    
    def forward(self,
                points: Optional[Tuple[torch.Tensor, torch.Tensor]],
                boxes: Optional[torch.Tensor],
                masks: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        
        """
        Embeds different types of prompts, returning both sparse and dense embeddings.

        Arguments:
          points : point coordinates and labels to embed
                    points shape [B, N, 2] and labels shape [B, N].
          boxes : boxes to embed in the shape [B, N, 4].
          masks : masks to embed in the shape [B, 1, 4*embed_H, 4*embed_W].

        Returns:
          torch.Tensor: sparse embeddings for the points and boxes, with shape [B, N, embed_dim], 
                        where N is determined by the number of input points and boxes.
          torch.Tensor: dense embeddings for the masks, in the shape
                        [B, embed_dim, embed_H, embed_W]
        """
        bs = self._get_batch_size(points, boxes, masks)
        device = self._get_device()
        sparse_embeddings = torch.empty((bs, 0, self.embed_dim), device=device)
        if points is not None:
            point_embeddings = self._embed_points(points[0], points[1], pad=(boxes is None))
            sparse_embeddings = torch.cat((sparse_embeddings, point_embeddings), dim=1)
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = torch.cat((sparse_embeddings, box_embeddings), dim=1)
        
        if masks is not None:
            dense_embeddings = self._embed_masks(masks) # [B, embed_dim, embed_H, embed_W]
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            ) # [B, embed_dim, embed_H, embed_W]

        return sparse_embeddings, dense_embeddings