import torch.nn as nn
import torch

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""
    """目的是处理fp16数据类型时，计算LayNorm仍使用float32，保证数值稳定性"""
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)