import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.backbone import Backbone
from model.transformer import Transformer
from model.position_encoding import PositionEmbeddingSine
from utils.utils import NestedTensor

class MLP(nn.Module):
    """
    一个简单的多层感知器(MLP)。
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x) # 最后一层不使用ReLU激活函数
        return x

class DETR(nn.Module):
    """
    DETR 主模块，负责整合所有组件
    """
    def __init__(self,
                 backbone: Backbone,
                 transformer: Transformer,
                 num_classes: int,
                 num_queries: int,
                 return_intermediate_dec: bool = False,
                 ):
        """
        Args:
            backbone: 骨干网络
            transformer: 变压器
            num_classes: 类别数
            num_queries: 查询数
            return_intermediate_dec: 是否返回中间解码层的输出
        """
        super().__init__()
        self.backbone = backbone
        self.transformer = transformer
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.return_intermediate_dec = return_intermediate_dec

        # 从transformer中获取隐藏维度
        hidden_dim = transformer.d_model
        
        # --- 预测头 (Prediction Heads) ---
        # 类别预测头: +1 是为了 "no object" (无目标) 这个背景类别
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        
        # 边界框预测头: 使用一个3层的MLP
        self.bbox_embed = MLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=4, num_layers=3)
        
        # --- 组件 ---
        # Object Queries: 可学习的查询向量
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        # 输入投影层: 将backbone的输出通道数 (如ResNet50的2048) 降维到transformer的输入维度
        # 使用1x1卷积实现
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)

    