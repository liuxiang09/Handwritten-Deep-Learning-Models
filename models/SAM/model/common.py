import torch
import torch.nn as nn


class LayerNorm2d(nn.Module):
    """
    一个为 2D 图像数据 (N, C, H, W) 设计的 LayerNorm。
    它会在通道维度 (C) 上进行归一化。
    """
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x 的形状是 (B, C, H, W)
        # 计算通道维度的均值和方差
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        
        # 归一化
        x = (x - u) / torch.sqrt(s + self.eps)
        
        # 应用可学习的 gain 和 bias
        # self.weight 和 self.bias 的形状是 (C,)
        # 需要重塑为 (1, C, 1, 1) 以便和 x 进行广播
        x = self.weight.view(1, -1, 1, 1) * x + self.bias.view(1, -1, 1, 1)
        
        return x