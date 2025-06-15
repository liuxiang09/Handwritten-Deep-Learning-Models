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

    def forward(self, x: NestedTensor):
        """
        模型的完整前向传播流程。
        
        Args:
            x (NestedTensor): 包含 'tensors' 和 'mask' 的输入对象。
                - tensors: [B, 3, H, W]
                - mask: [B, H, W]
        
        Returns:
            dict: 包含预测结果的字典
                - 'pred_logits': [B, num_queries, num_classes + 1] 分类预测
                - 'pred_boxes': [B, num_queries, 4] 边界框预测 (cx, cy, w, h)
                - 'aux_outputs' (if training): 一个列表，包含解码器每个中间层的预测结果
        """
        # 1. 通过Backbone提取特征
        features = self.backbone(x) # features 是一个字典
        
        # 只取最后一层的特征和mask
        # src: NestedTensor(tensor, mask)
        src = features['0']
        
        # 2. 生成位置编码 (这里直接在外面构建好传入，或者在模型内部构建)
        # 假设位置编码模块已经构建好并命名为 pos_encoding
        # pos_embed = self.pos_encoding(src) # 这通常在主训练循环中完成，然后传入
        
        # 3. 将特征图、mask和位置编码传入Transformer
        # 注意：这里的 pos_embed 需要在外部生成并传入
        # 为了演示，我们暂时在这里创建一个（实际应在外部build）
        pos_encoding_module = PositionEmbeddingSine(num_pos_feats=self.transformer.d_model // 2, normalize=True)
        pos_embed = pos_encoding_module(src)
        
        hs, _ = self.transformer(self.input_proj(src.tensors), src.mask, self.query_embed.weight, pos_embed)
        # hs shape: [num_layers, B, num_queries, C]
        
        # 4. 通过预测头得到最终输出
        outputs_class = self.class_embed(hs) # 分类 logits
        outputs_coord = self.bbox_embed(hs).sigmoid() # 边界框坐标 (归一化到 0-1)
        
        # 整理输出格式
        # out['pred_logits'] 是最后一层decoder的输出
        # out['pred_boxes'] 是最后一层decoder的输出
        out = {
            'pred_logits': outputs_class[-1], 
            'pred_boxes': outputs_coord[-1]
        }
        
        # 如果需要返回中间层结果（用于辅助损失计算）
        if self.return_intermediate_dec:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
            
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # 将解码器除最后一层外的所有中间层输出打包，用于计算辅助损失
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
    