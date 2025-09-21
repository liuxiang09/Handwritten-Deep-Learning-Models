import torch
import torch.nn as nn
import timm
from typing import Type
from .common import LayerNorm2d
from torchinfo import summary


class ImageEncoderViT(nn.Module):
    def __init__(self,
                 model_name: str = 'vit_base_patch16_224',
                 image_size: int = 1024,
                 embed_dim: int = 768,
                 output_channels: int = 256,
                 norm_layer: Type[nn.Module] = LayerNorm2d):
        """
        Args:
            model_name (str): timm库中的模型名称，例如 'vit_base_patch16_224', 'vit_large_patch16_224', 'vit_huge_patch14_224'。
            image_size (int): 输入图像大小（假定为正方形）。
            embed_dim (int): ViT嵌入维度。'vit_base' 是 768，'vit_large' 是 1024，'vit_huge' 是 1280。
            output_channels (int): 最终特征图中的输出通道数。
            norm_layer (Type[nn.Module]): 要使用的归一化层。
        """
        super().__init__()
        self.image_size = image_size
        # ===== 从timm加载预训练的ViT模型 =====
        self.vit = timm.create_model(
            model_name, 
            pretrained=True, 
            num_classes=0,
            img_size=self.image_size)
        self.patch_size = self.vit.patch_embed.patch_size[0]

        # ===== 定义SAM特定的Neck结构 =====
        self.neck = nn.Sequential(
            nn.Conv2d(embed_dim, output_channels, kernel_size=1, bias=False),
            norm_layer(output_channels),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(output_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 形状为(B, C, H, W)的输入张量
        Returns:
            image_embedding: 形状为(B, output_channels, H_feat, W_feat)的输出张量
        """
        features = self.vit.forward_features(x) # x形状: (B, num_patches+1, embed_dim)
        # 将特征重塑为(B, embed_dim, H_feat, W_feat)
        B, num_patches, embed_dim = features.shape
        H_feat = W_feat = self.image_size // self.patch_size
        features = features[:, 1:, :].permute(0, 2, 1).reshape(B, embed_dim, H_feat, W_feat)  # (B, embed_dim, H_feat, W_feat)
        # 通过Neck结构
        image_embedding = self.neck(features)  # (B, output_channels, H_feat, W_feat)

        return image_embedding
