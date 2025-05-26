import torch.nn as nn
import torch
from layernorm import LayerNorm
from transformer import Transformer


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        # width相当于transform中的d_model
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        # x=[1,3,224,224]
        x = self.conv1(x)  # shape = [*, width, grid, grid], 将图片分成[32,32]个patch  [1,768,7,7]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width] 添加一个 [CLS] token [1,50,7*7]
        x = x + self.positional_embedding.to(x.dtype) # 可学习的位置编码
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :]) # 将所有信息汇聚到cls token中，只需前面来做下游任务 [1,768]

        if self.proj is not None: # self.proj是可学习参数，维度为[768,512]
            x = x @ self.proj  # 通过学习参数将维度再次融合变成512特征，最终为[1,512]

        return x
    
if __name__ == '__main__':
        
    # 测试代码开始
    print("开始测试 VisionTransformer...")

    # 输入图像张量 (Batch_size, Channels, Height, Width)
    x = torch.randn(1, 3, 224, 224)
    print(f"输入张量形状: {x.shape}")

    # VisionTransformer 参数 (示例值，请根据你的模型配置调整)
    input_resolution = 224
    patch_size = 32 # 例如 ViT-Base-Patch32
    width = 768     # 例如 ViT-Base 的隐藏层维度
    layers = 12     # 例如 ViT-Base 的层数
    heads = 12      # 例如 ViT-Base 的注意力头数 (通常 width // 64)
    output_dim = 512 # CLIP 模型的嵌入维度

    # 实例化 VisionTransformer
    try:
        vit = VisionTransformer(
            input_resolution=input_resolution,
            patch_size=patch_size,
            width=width,
            layers=layers,
            heads=heads,
            output_dim=output_dim
        )
        print("VisionTransformer 实例化成功.")

        # 进行前向传播
        output = vit(x)

        print(f"输出张量形状: {output.shape}")
        print("VisionTransformer 测试完成.")

    except Exception as e:
        print(f"测试 VisionTransformer 时发生错误: {e}")
