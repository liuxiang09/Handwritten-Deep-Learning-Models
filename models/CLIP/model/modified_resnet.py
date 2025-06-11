import torchvision
import torch
import torch.nn as nn

class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        # x shape: (N, C, H, W)
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # (H*W, N, C)
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (H*W+1, N, C)
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # Add pos embedding
        
        # Multi-head attention
        x, _ = F.multi_head_attention_forward(
            query=x[:1], # 使用全局平均池化作为 query
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0.0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0) # (N, C)

# 需要导入 F
from torch.nn import functional as F

class ModifiedResNet(nn.Module):
    def __init__(self, embed_dim: int, n_head: int):
        super().__init__()
        
        # 加载预训练的 ResNet50
        self.resnet = torchvision.models.resnet50(weights='IMAGENET1K_V2')
        
        # 去掉最后的平均池化层和全连接层
        self.resnet.avgpool = nn.Identity()
        self.resnet.fc = nn.Identity()
        
        # ResNet50 最后输出的通道数是 2048，特征图尺寸是 7x7
        self.attnpool = AttentionPool2d(7, 2048, n_head, embed_dim)

    def forward(self, image: torch.Tensor):
        """
        Args:
            image (torch.Tensor): 输入图片, shape: (N, 3, 224, 224)

        Returns:
            torch.Tensor: 图像特征, shape: (N, embed_dim)
        """
        # ResNet backbone
        x = self.resnet.conv1(image)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x) # (N, 2048, 7, 7)
        
        # Attention Pooling
        x = self.attnpool(x) # (N, embed_dim)
        
        return x