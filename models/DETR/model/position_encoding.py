import math
import torch
from torch import nn
from typing import Optional
class PositionEmbeddingSine(nn.Module):
    """
    使用正弦位置编码来编码位置信息。
    """
    def __init__(self, num_pos_feats: int = 128, temperature: int = 10000, normalize: bool = False, scale: Optional[float] = None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        
