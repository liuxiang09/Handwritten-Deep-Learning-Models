import torch
import numpy as np
import torch.nn as nn
from typing import Tuple, Type, Optional
from .common import LayerNorm2d

class PositionEmbeddingRandom(nn.Module):
    """
    用于2D图像的位置编码，使用随机值。
    """
    def __init__(self,
                 num_pos_feats: int = 128,
                 scale: float = 1.0):
        
        super().__init__()
        # 从高斯分布中采样一个固定的、非学习的随机矩阵
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)), # [2, num_pos_feats]
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """
        对归一化到[0,1]的点进行位置编码。
        Args:
            coords: (B, N, 2) 或 (H, W, 2) 坐标，范围在[0, 1]
        Returns:
            pe: (B, N, 2 * num_pos_feats) 位置编码
        """
        coords = 2 * coords - 1 # 从[0, 1]归一化到[-1, 1]
        # 使用高斯矩阵计算位置编码
        coords = coords @ self.positional_encoding_gaussian_matrix # [B, N, 2] @ [2, num_pos_feats] -> [B, N, num_pos_feats]
        coords = 2 * np.pi * coords  # 缩放坐标
        # 采用正弦编码方案
        pe = torch.cat((coords.sin(), coords.cos()), dim=-1) # [B, N, 2 * num_pos_feats]
        return pe
    
    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """
        为指定大小的网格生成位置编码。
        Args:
            size: (H, W) 坐标网格的大小
        Returns:
            pe: (H, W, 2 * num_pos_feats) 位置编码
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
        对未归一化到[0,1]的点进行位置编码。
        Args:
            coords_input: (B, N, 2)
            image_size: (H, W) 图像的大小
        Returns:
            pe: (B, N, 2 * num_pos_feats) 位置编码
        """
        # 归一化到[0, 1]
        coords = coords_input.clone()
        H, W = image_size
        coords[:, :, 0] = coords[:, :, 0] / W # 归一化x
        coords[:, :, 1] = coords[:, :, 1] / H # 归一化y
        return self._pe_encoding(coords.to(torch.float)) # [B, N, 2 * num_pos_feats]
    
class PromptEncoder(nn.Module):
    
    def __init__(self,
                 embed_dim: int,
                 image_embedding_size: Tuple[int, int],
                 input_image_size: Tuple[int, int],
                 mask_in_channels: int,
                 activation: Type[nn.Module] = nn.GELU):
        """
        对输入到SAM掩码解码器的提示进行编码。

        Args:
          embed_dim (int): 提示的嵌入维度
          image_embedding_size (tuple(int, int)): 图像嵌入的空间大小，格式为(H, W)。
          input_image_size (tuple(int, int)): 输入到图像编码器的图像的填充大小，格式为(H, W)。
          mask_in_chans (int): 用于编码输入掩码的隐藏通道数。
          activation (nn.Module): 编码输入掩码时使用的激活函数
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.image_embedding_size = image_embedding_size
        self.input_image_size = input_image_size
        self.pe_layer = PositionEmbeddingRandom(num_pos_feats=embed_dim // 2)

        self.num_point_embeddings = 4 # 正/负点 + 2个框角点
        point_embeddings = [nn.Embedding(1, embed_dim) for _ in range(self.num_point_embeddings)] # 嵌入权重形状: [1, embed_dim]
        self.point_embeddings = nn.ModuleList(point_embeddings)
        self.not_a_point_embed = nn.Embedding(1, embed_dim)

        self.mask_input_size = (4 * image_embedding_size[0], 4 * image_embedding_size[1]) # [4*embed_H, 4*embed_W]
        # 下采样掩码输入以匹配图像嵌入大小
        # 例如，[4 * 64, 4 * 64] -> [64, 64]，这取决于图像编码器的嵌入大小
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
        返回用于编码点提示的位置编码，
        应用于图像编码形状的密集点集。

        Returns:
          torch.Tensor: 形状为[1, 2*embed_dim, embed_H, embed_W]的位置编码
        """
        return self.pe_layer.forward(self.image_embedding_size).unsqueeze(0)
        
    def _embed_points(self,
                      points: torch.Tensor,
                      labels: torch.Tensor,
                      pad: bool) -> torch.Tensor:
        """
        嵌入点提示。
        Args:
            points (torch.Tensor): 形状为(B, N, 2)的张量，包含点坐标。
            labels (torch.Tensor): 形状为(B, N)的张量，包含点标签。
            pad (bool): 是否用零点和-1标签填充点和标签。
        Returns:
            torch.Tensor: 形状为(B, N, embed_dim)的嵌入点。
        """
        points = points + 0.5 # 移动到像素中心
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
        嵌入框提示。
        Args:
            boxes (torch.Tensor): 形状为(B, N, 4)的张量，包含框坐标。
        Returns:
            torch.Tensor: 形状为(B * N, 2, embed_dim)的嵌入框。
        """
        boxes = boxes + 0.5 # 移动到像素中心
        coords = boxes.reshape(-1, 2, 2)  # [B * N, 2, 2]
        corner_embedding = self.pe_layer.forward_with_coords(coords, self.input_image_size)
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight # 左上角
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight # 右下角
        return corner_embedding
    
    def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """
        嵌入掩码提示。
        Args:
            masks (torch.Tensor): 形状为(B, 1, 4*embed_H, 4*embed_W)的张量，包含二进制掩码。
        Returns:
            torch.Tensor: 形状为(B, embed_dim, embed_H, embed_W)的嵌入掩码。
        """
        mask_embedding = self.mask_downscaling(masks)
        return mask_embedding
    
    def _get_batch_size(self,
                        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
                        boxes: Optional[torch.Tensor],
                        masks: Optional[torch.Tensor]) -> int:
        """
        返回输入张量的批大小。
        Args:
            points (Optional[Tuple[torch.Tensor, torch.Tensor]]): 点坐标和标签。
            boxes (Optional[torch.Tensor]): 框坐标。
            masks (Optional[torch.Tensor]): 掩码张量。
        Returns:
            int: 批大小。
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
        嵌入不同类型的提示，返回稀疏和密集嵌入。

        Arguments:
          points : 要嵌入的点坐标和标签
                    点形状[B, N, 2]，标签形状[B, N]。
          boxes : 要嵌入的框，形状[B, N, 4]。
          masks : 要嵌入的掩码，形状[B, 1, 4*embed_H, 4*embed_W]。

        Returns:
          torch.Tensor: 点和框的稀疏嵌入，形状[B, N, embed_dim]，
                        其中N由输入点和框的数量决定。
          torch.Tensor: 掩码的密集嵌入，形状
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