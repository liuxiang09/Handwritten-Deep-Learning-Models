import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List
from .matcher import HungarianMatcher
from ..utils.utils import generalized_box_iou, cxcywh_to_xyxy




class SetCriterion(nn.Module):
    """
    1. 类别损失 (Classification Loss)
    2. L1 边界框损失 (L1 Bbox Loss)
    3. GIoU 边界框损失 (Generalized IoU Loss)
    """
    def __init__(self, 
                 num_classes: int, 
                 matcher: HungarianMatcher, 
                 weight_dict: Dict[str, float], 
                 eos_coef: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef

        # 创建一个用于类别损失的权重张量
        background_weight = torch.ones(num_classes + 1)
        background_weight[-1] = eos_coef  # 最后一类是背景类
        self.register_buffer("background_weight", background_weight)


    def forward(self,
                outputs: Dict[str, torch.Tensor],
                targets: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Args:
            outputs: 模型的输出，包含:
                - "pred_labels": [B, num_queries, num_classes + 1]
                - "pred_boxes": [B, num_queries, 4](cx, cy, w, h) 且已归一化到[0, 1]
            targets: 真实标签列表，每个元素是一个字典，包含:
                - "labels": [num_boxes]
                - "boxes": [num_boxes, 4](x_min, y_min, x_max, y_max) 且未归一化

        Returns:
            dict: 包含损失值的字典
        """
        # =====匈牙利匹配 =====
        indices = self.matcher(outputs, targets)

        # ===== 计算损失 =====
        num_total_boxes = sum(len(t["boxes"]) for t in targets)

        losses = {}
        loss_ce = 0.0
        loss_bbox = 0.0
        loss_giou = 0.0
        for i, (matched_pred_idx, matched_target_idx) in enumerate(indices):
            # 获取当前batch的预测和目标
            pred_labels = outputs["pred_labels"][i]
            pred_boxes = outputs["pred_boxes"][i]
            target = targets[i]

            # 计算类别损失
            target_labels = torch.full((pred_labels.shape[0],), self.num_classes, dtype=torch.int64, device=pred_labels.device)
            target_labels[matched_pred_idx] = target["labels"][matched_target_idx]
            loss_ce += F.cross_entropy(pred_labels, target_labels, weight=self.background_weight)
            loss_ce_test = F.cross_entropy(pred_labels, target_labels)

            # 计算边界框损失
            if len(matched_target_idx) > 0:
                matched_pred_boxes = pred_boxes[matched_pred_idx]
                matched_target_boxes = target["boxes"][matched_target_idx]
                loss_bbox += F.l1_loss(matched_pred_boxes, matched_target_boxes, reduction="sum")

            # 计算GIoU损失
            if len(matched_target_idx) > 0:
                matched_pred_boxes = pred_boxes[matched_pred_idx]
                matched_target_boxes = target["boxes"][matched_target_idx]
                giou = generalized_box_iou(
                    cxcywh_to_xyxy(matched_pred_boxes), 
                    cxcywh_to_xyxy(matched_target_boxes))
                loss_giou += (1 - torch.diag(giou)).sum()

        losses = {
            "loss_ce": loss_ce * self.weight_dict.get("loss_ce", 1.0) / len(targets),
            "loss_bbox": loss_bbox * self.weight_dict.get("loss_bbox", 1.0) / num_total_boxes,
            "loss_giou": loss_giou * self.weight_dict.get("loss_giou", 1.0) / num_total_boxes,
        }
        return losses