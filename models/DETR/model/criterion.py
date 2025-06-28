import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
from typing import List, Dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.utils import cxcywh_to_xyxy, generalized_box_iou
from model.matcher import HungarianMatcher

class SetCriterion(nn.Module):
    """
    DETR损失函数，计算类别损失、L1边界框损失和GIoU损失。
    """
    def __init__(self, num_classes: int, matcher: HungarianMatcher, weight_dict: Dict, eos_coef: float):
        """
        初始化损失函数。
        Args:
            num_classes (int): 类别总数 (不包括背景)
            matcher (HungarianMatcher): 用于匹配预测和目标的模块
            weight_dict (Dict): 一个字典，包含各类损失的权重
            eos_coef (float): "无目标"类别的分类损失权重，用于平衡正负样本
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        
        # 定义一个特殊的权重张量，用于处理类别不平衡问题
        # "无目标"类别的权重是eos_coef，其他类别的权重都是1.0
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight) # 注册为buffer，这样它就不会被视为模型参数

    # --- 单项损失计算函数 ---

    def _loss_labels(self, outputs: Dict, targets: List[Dict], indices: List[tuple], num_boxes: int):
        """
        计算类别损失 (Classification Loss)
        Args:
            outputs (Dict): 模型的输出，包含预测的类别 logits，形状为 [B, num_queries, num_classes + 1]
            targets (List[Dict]): 真实标签列表, 每个字典包含目标框和标签
            indices (List[tuple]): 匹配结果，包含预测和目标的索引
            num_boxes (int): Batch中所有目标框的总数，用于归一化损失
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits'] # [B, num_queries, num_classes + 1]

        # 把indices展开为batch索引和预测索引
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        # 对每个batch，根据indices的真实类别索引取出真实类别标签
        target_classes_o = torch.cat([target["labels"][J] for target, (_, J) in zip(targets, indices)]) # [B, num_matched_boxes]
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[batch_idx, src_idx] = target_classes_o

        # 计算交叉熵损失，使用前面定义的`empty_weight`来平衡正负样本
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        return {'loss_ce': loss_ce}

    def _loss_boxes(self, outputs: Dict, targets: List[Dict], indices: List[tuple], num_boxes: int):
        """计算边界框损失 (Bounding Box Loss)，包含L1损失和GIoU损失"""
        assert 'pred_boxes' in outputs
        # 直接展开索引逻辑
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        src_boxes = outputs['pred_boxes'][batch_idx, src_idx] # [num_matched_boxes, 4]
        target_boxes = torch.cat([target['boxes'][i] for target, (_, i) in zip(targets, indices)], dim=0)

        # L1 损失
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        
        # GIoU 损失
        loss_giou = 1 - torch.diag(generalized_box_iou(
            cxcywh_to_xyxy(src_boxes),
            cxcywh_to_xyxy(target_boxes)
        ))

        return {'loss_bbox': loss_bbox.sum() / num_boxes, 'loss_giou': loss_giou.sum() / num_boxes}
    
    # --- 前向传播 ---

    def forward(self, outputs: Dict, targets: List[Dict]):
        """
        计算总损失。
        Args:
            outputs (Dict): 模型的原始输出，可能包含用于辅助损失的中间层输出
            targets (List[Dict]): 真实标签列表
        Returns:
            losses (Dict): 包含所有损失项和其加权和的字典
        """
        # 1. 匹配
        # 我们只对最后一层的输出进行匹配
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        indices = self.matcher(outputs_without_aux, targets)

        # 2. 计算Batch中所有目标框的总数，用于归一化
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        # TODO: 处理分布式训练时的num_boxes同步问题
        
        # 3. 计算各项损失
        losses = {}
        # -- 类别损失 --
        losses.update(self._loss_labels(outputs, targets, indices, num_boxes))
        # -- 边界框损失 --
        losses.update(self._loss_boxes(outputs, targets, indices, num_boxes))

        # 4. 计算辅助损失 (如果存在)
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                # 对每一层中间输出都进行匹配和损失计算
                indices_aux = self.matcher(aux_outputs, targets)
                
                # 计算类别损失
                l_dict = self._loss_labels(aux_outputs, targets, indices_aux, num_boxes)
                # 计算边界框损失
                l_dict.update(self._loss_boxes(aux_outputs, targets, indices_aux, num_boxes))
                
                # 将辅助损失的键名加上后缀
                l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses