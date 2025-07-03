import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

from .backbone import Backbone
from .transformer import Transformer
from .position_encoding import PositionEmbeddingSine
from ..utils.utils import NestedTensor

class MLP(nn.Module):
    """简单的多层感知器"""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, ..., input_dim]
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        # 输出: [batch_size, ..., output_dim]
        return x

class DETR(nn.Module):
    """
    DETR: DEtection TRansformer - 使用Transformer进行端到端目标检测
    文档中给出了DETR的张量数据流，很大程度帮助理解DETR的整个流程
    Google gemini canvas 文档: https://gemini.google.com/share/40c494881840
    """
    def __init__(self,
                 backbone: Backbone,
                 transformer: Transformer,
                 num_classes: int,
                 num_queries: int,
                 return_intermediate_dec: bool = False):
        
        super().__init__()
        self.backbone = backbone
        self.transformer = transformer
        self.num_classes = num_classes
        self.num_queries = num_queries
        hidden_dim = transformer.d_model
        
        # 预测头
        self.class_head = nn.Linear(hidden_dim, num_classes + 1)  # +1表示"无目标"类别
        self.bbox_head = MLP(hidden_dim, hidden_dim, 4, 3)  # 预测归一化的边界框坐标(cx,cy,w,h)
        
        # 可学习的对象查询
        self.query_embed = nn.Embedding(num_queries, hidden_dim) # [100, hidden_dim]
        
        # 特征投影和位置编码
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.pos_encoding = PositionEmbeddingSine(num_pos_feats=hidden_dim // 2, normalize=True)
        
        # 确保transformer的d_model等于位置编码num_pos_feats的2倍
        assert hidden_dim == self.pos_encoding.num_pos_feats * 2, f"Transformer的hidden_dim({hidden_dim})必须等于位置编码num_pos_feats({self.pos_encoding.num_pos_feats})的2倍"
        
        if return_intermediate_dec:
            self.transformer.decoder.return_intermediate = return_intermediate_dec
        self.return_intermediate_dec = return_intermediate_dec

    def forward(self, x: NestedTensor):
        """
        Args:
            x: NestedTensor包含:
               - tensors: [B, 3, H, W] - 输入图像
               - mask: [B, H, W] - 填充掩码(1表示填充区域)
        
        Returns:
            dict: 包含预测结果
        """
        # 1. 特征提取
        features = self.backbone(x)  # 返回字典
        src = features['0']  # [B, C, H/32, W/32] 特征图
        
        # 2. 位置编码--不会用到src.tensors，只会用到src.mask，得到的通道数与src.tensors的C通道数无关
        pos_embed = self.pos_encoding(src)  # [B, hidden_dim, H/32, W/32]
        
        # 3. Transformer处理
        # src: [B, hidden_dim, H/32, W/32]
        # mask: [B, H/32, W/32]
        # query_embed: [num_queries, hidden_dim]
        # pos_embed: [B, hidden_dim, H/32, W/32]
        hs, _ = self.transformer(src=self.input_proj(src.tensors), mask=src.mask, query_embed=self.query_embed.weight, pos_embed=pos_embed)
        # hs: [num_decoder_layers 或 1, B, num_queries, hidden_dim]
        
        # 4. 预测头输出
        outputs_class = self.class_head(hs)  # [num_decoder_layers, B, num_queries, num_classes+1]
        outputs_coord = self.bbox_head(hs).sigmoid()  # [num_decoder_layers, B, num_queries, 4]

        # 5. 只取最后一层输出
        out = {
            'pred_labels': outputs_class[-1],  # [B, num_queries, num_classes+1]
            'pred_boxes': outputs_coord[-1]    # [B, num_queries, 4]
        }
            
        return out
    