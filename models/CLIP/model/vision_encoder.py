import torch
import torch.nn as nn

# 无预训练权重的VisionEncoder
class VisionEncoder(nn.Module):
    def __init__(
            self,
            image_size: int,
            patch_size: int,
            embed_dim: int,
            n_head: int,
            n_layer: int,
    ):
        super().__init__()

        # 将输入的图像分割成多个patch，每个patch的大小为 patch_size x patch_size
        self.patch_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) # [1, 1, D]
        nn.init.normal_(self.cls_token, mean=0.0, std=0.01)
        # 位置编码
        num_patches = (image_size // patch_size) ** 2
        self.positional_embedding = nn.Parameter(torch.empty(1, num_patches + 1, embed_dim)) # [num_patches + 1, D]
        nn.init.normal_(self.positional_embedding, mean=0.0, std=0.01)

        # 使用PyTorch内置的TransformerEncoderLayer构建编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=n_head,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_layer,
        )
        self.ln_final = nn.LayerNorm(embed_dim)
        
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image (torch.Tensor): 输入图片, shape: (N, 3, H, W)

        Returns:
            torch.Tensor: 图像特征, shape: (N, D)
        """
        x = self.patch_embedding(image) # [N, D, H/P, W/P]
        x = x.flatten(2) # [N, D, H/P * W/P]
        x = x.transpose(1, 2) # [N, H/P * W/P, D] == [N, num_patches, D]

        # 添加 CLS token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1) # [N, 1, D]
        x = torch.cat([cls_tokens, x], dim=1) # [N, 1 + num_patches, D]
        
        # 位置编码
        x = x + self.positional_embedding # [N, 1 + num_patches, D]

        # 编码器
        x = self.transformer_encoder(x) # [N, 1 + num_patches, D]

        # 取 CLS token 的输出作为图像特征
        x = x[:, 0, :] # [N, D]

        x = self.ln_final(x) # [N, D]
        return x
    
# 有预训练权重的VisionEncoder
import timm
class VisionEncoderPretrained(nn.Module):
    def __init__(
            self,
            pretrained_model_name: str,
            pretrained: bool = True,
            embed_dim: int = 768, # Base ViT 的 embed_dim 是 768
    ):
        """
        使用 timm 加载预训练的 Vision Transformer 模型。

        Args:
            pretrained_model_name (str): timm 库中的模型名称。
            pretrained (bool): 是否加载预训练权重。
            embed_dim (int): 模型的输出维度。
        """
        super().__init__()
        self.embed_dim = embed_dim
        # timm 的 ViT 模型内部已经包含了 CLS token、位置编码和最后的 LayerNorm
        # 所以我们不再需要手动定义 self.cls_token, self.positional_embedding 和 self.ln_final
        # 它们的结构已经和预训练权重完美匹配
        self.vit = timm.create_model(
            model_name=pretrained_model_name,
            pretrained=pretrained,
            num_classes=0,
        )
        
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image (torch.Tensor): 输入图片, shape: (N, 3, H, W)
                                  注意：H 和 W 需要与模型匹配，例如 224x224

        Returns:
            torch.Tensor: 图像特征, shape: (N, embed_dim)
        """
        # timm 的 ViT 模型直接返回cls token的特征
        cls_token_features = self.vit(image) # [N, D]
        # 仅在第一次前向传播时打印shape
        # if not hasattr(self, '_shape_printed'):
        #     print("cls_token_features.shape:", cls_token_features.shape)
        #     self._shape_printed = True
        return cls_token_features
