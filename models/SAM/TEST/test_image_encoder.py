import torch
from torchinfo import summary

from models.SAM.model.image_encoder import ImageEncoderViT

image_encoder = ImageEncoderViT(
    model_name='vit_base_patch16_224',
    image_size=1024,
    embed_dim=768,
    output_channels=256
)

x = torch.randn(4, 3, 1024, 1024)  # 模拟输入图像
output = image_encoder(x)
print(f"Output shape: {output.shape}")