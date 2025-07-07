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
            model_name (str): model name from timm library, e.g., 'vit_base_patch16_224', 'vit_large_patch16_224', 'vit_huge_patch14_224'.
            image_size (int): input image size (assumed to be square).
            embed_dim (int): ViT embedding dimension. 'vit_base' is 768, 'vit_large' is 1024, 'vit_huge' is 1280.
            output_channels (int): number of output channels in the final feature map.
            norm_layer (Type[nn.Module]): normalization layer to use.
        """
        super().__init__()
        self.image_size = image_size
        # ===== Load pretrained ViT model from timm =====
        self.vit = timm.create_model(
            model_name, 
            pretrained=True, 
            num_classes=0,
            img_size=self.image_size)
        self.patch_size = self.vit.patch_embed.patch_size[0]

        # ===== Define SAM specific Neck structure =====
        self.neck = nn.Sequential(
            nn.Conv2d(embed_dim, output_channels, kernel_size=1, bias=False),
            norm_layer(output_channels),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(output_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor of shape (B, C, H, W)
        Returns:
            image_embedding: output tensor of shape (B, output_channels, H_feat, W_feat)
        """
        features = self.vit.forward_features(x) # x shape: (B, num_patches+1, embed_dim)
        # reshape features to (B, embed_dim, H_feat, W_feat)
        B, num_patches, embed_dim = features.shape
        H_feat = W_feat = self.image_size // self.patch_size
        features = features[:, 1:, :].permute(0, 2, 1).reshape(B, embed_dim, H_feat, W_feat)  # (B, embed_dim, H_feat, W_feat)
        # pass through Neck structure
        image_embedding = self.neck(features)  # (B, output_channels, H_feat, W_feat)

        return image_embedding
