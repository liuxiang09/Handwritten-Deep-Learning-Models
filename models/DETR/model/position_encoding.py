import math
import torch
from torch import nn
from typing import Optional

from ..utils.utils import NestedTensor

class PositionEmbeddingSine(nn.Module):
    """
    使用正弦位置编码来编码位置信息。
    Google gemini canvas 文档: https://gemini.google.com/share/8c880bebdc06
    """
    def __init__(self, num_pos_feats: int = 128, temperature: int = 10000, normalize: bool = False, scale: Optional[float] = None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        
    def forward(self, x: NestedTensor):
        """
        Args:
            x: NestedTensor对象，包含'tensor'和'mask'属性
        Returns:
            pos: [B, hidden_dim, H, W] 位置编码
        """
        mask = x.mask  # [B, H, W]，True表示padding
        not_mask = ~mask  # 反转mask，False表示padding位置
        y_embed = not_mask.cumsum(1, dtype=torch.float32)  # 累加得到y坐标
        x_embed = not_mask.cumsum(2, dtype=torch.float32)  # 累加得到x坐标
        
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale  # 归一化
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale  # 归一化
        
        # 生成频率张量 
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=mask.device)  # [num_pos_feats]
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)  # [num_pos_feats]
        
        # 广播机制，将维度扩展为[B, H, W, num_pos_feats]
        # x_embed 和 y_embed 的维度是[B, H, W]，dim_t 的维度是[num_pos_feats]
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3) # stack + flatten 实现交错排列
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    可学习的位置编码。
    Google gemini canvas 文档: https://gemini.google.com/share/6d99e5178acd
    """
    def __init__(self, num_pos_feats: int = 128):
        super().__init__()
        # 创建两个查询表，一个给行（高度），一个给列（宽度）
        # 50是一个预设的最大值，表示模型能处理的最大高/宽
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        # 用均匀分布初始化查询表里的权重
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x: NestedTensor) -> torch.Tensor:
        assert len(x.mask.shape) == 3, "输入的 NestedTensor 必须有3个维度 [B, H, W]"
        h, w = x.mask.shape[1:]
        i = torch.arange(w, device=x.mask.device) # 列索引: [0, 1, ..., w-1]
        j = torch.arange(h, device=x.mask.device) # 行索引: [0, 1, ..., h-1]

        # 从查询表中查找每个行/列索引对应的向量
        x_emb = self.col_embed(i)  # [W, num_pos_feats]
        y_emb = self.row_embed(j)  # [H, num_pos_feats]

        # 将行、列向量组合成一个 [H, W, 2*num_pos_feats] 的位置图，然后调整维度并重复以适应批次大小
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1), # [H, W, num_pos_feats]
            y_emb.unsqueeze(1).repeat(1, w, 1), # [H, W, num_pos_feats]
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.mask.shape[0], 1, 1, 1)

        return pos
