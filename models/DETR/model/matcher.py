import torch
import torch.nn as nn
from typing import List, Dict
from scipy.optimize import linear_sum_assignment

from ..utils.utils import cxcywh_to_xyxy, generalized_box_iou

class HungarianMatcher(nn.Module):
    """
    DETR中的核心组件之一：匈牙利匹配器。
    该模块负责在模型的预测结果和真实标签之间找到一个最优的二分匹配。
    匹配的依据是类别预测的置信度和边界框的相似度。
    """
    def __init__(self, 
                 cost_class: float, 
                 cost_L1: float, 
                 cost_giou: float):
        """
        初始化匈牙利匹配器。
        Args:
            cost_class (float): 类别匹配的权重
            cost_L1 (float): L1边界框损失的权重
            cost_giou (float): GIoU损失的权重
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_L1 = cost_L1
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, 
                outputs, 
                targets) -> List[tuple]:
        """
        Args:
            outputs (Dict): 模型的输出，包含:
                           - "pred_labels": [B, num_queries, num_classes + 1]
                           - "pred_boxes": [B, num_queries, 4]
            targets (List[Dict]): 真实标签列表，每个元素是一个字典，包含:
                                - "labels": [num_target_boxes]
                                - "boxes": [num_target_boxes, 4]
        Returns:
            List[tuple]: 一个列表，长度为batch_size。每个元素是一个元组(row_ind, col_ind)，
                         代表成功匹配的(预测索引, 真实标签索引)。
        """
        batch_size, num_queries = outputs["pred_labels"].shape[:2]

        # 计算代价矩阵 (Cost Matrix) --> [B, num_queries, num_target_boxes]

        # 将预测结果展平，方便后续计算
        pred_labels = outputs["pred_labels"].flatten(0, 1).softmax(-1)  # [B * num_queries, num_classes + 1]
        pred_boxes = outputs["pred_boxes"].flatten(0, 1)  # [B * num_queries, 4]

        # 将target列表的所有样本拼接起来
        labels = torch.cat([t["labels"] for t in targets]) # [total_num_boxes]
        boxes = torch.cat([t["boxes"] for t in targets]) # [total_num_boxes, 4]

        # 计算类别代价 (Classification Cost)
        cost_class = -pred_labels[:, labels] # [B * num_queries, total_num_boxes]

        # 计算L1 BBox代价 (L1 BBox Cost)
        # cdist计算所有预测框和所有真实框之间的L1距离
        cost_L1 = torch.cdist(pred_boxes, boxes, p=1) # [B * num_queries, total_num_boxes]

        # 计算GIoU BBox代价 (GIoU BBox Cost)
        cost_giou = -generalized_box_iou(cxcywh_to_xyxy(pred_boxes), cxcywh_to_xyxy(boxes)) # [B * num_queries, total_num_boxes]

        # 最终代价是三者的加权和
        C = self.cost_L1 * cost_L1 + self.cost_class * cost_class + self.cost_giou * cost_giou # [B * num_queries, total_num_boxes]
        C = C.view(batch_size, num_queries, -1).cpu() # [B, num_queries, total_num_boxes]

        # 匈牙利算法求解最优匹配
        sizes = [len(v["boxes"]) for v in targets] # 每个batch中目标框的数量
        # 根据每个batch的目标框数量分割代价矩阵
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))] # List[tuple(num_boxes, num_boxes)]
        
        # 把匹配结果转换为tensor格式
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]