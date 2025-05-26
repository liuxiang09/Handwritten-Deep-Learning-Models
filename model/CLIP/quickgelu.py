import torch.nn as nn
import torch

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x) # torch.sigmoid(1.702 * x)用于近似标准正态分布