import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
from typing import List, Dict
from torch import Tensor
from scipy.optimize import linear_sum_assignment

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.utils import cxcywh_to_xyxy, generalized_box_iou

class HungarianMatcher(nn.Module):
    """
    DETR中的核心组件之一：匈牙利匹配器。
    该模块负责在模型的预测结果和真实标签之间找到一个最优的二分匹配。
    匹配的依据是类别预测的置信度和边界框的相似度。
    """
    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """
        初始化匈牙利匹配器。
        Args:
            cost_class (float): 类别匹配的权重
            cost_bbox (float): L1边界框损失的权重
            cost_giou (float): GIoU损失的权重
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "所有cost权重不能都为0"

    @torch.no_grad()
    def forward(self, outputs: Dict[str, Tensor], targets: List[Dict[str, Tensor]]) -> List[tuple]:
        """
        执行匹配过程。
        Args:
            outputs (Dict): 模型的输出，包含:
                           - "pred_logits": [B, num_queries, num_classes + 1]
                           - "pred_boxes": [B, num_queries, 4]
            targets (List[Dict]): 真实标签列表，每个元素是一个字典，包含:
                                - "labels": [num_target_boxes]
                                - "boxes": [num_target_boxes, 4]
        Returns:
            List[tuple]: 一个列表，长度为batch_size。每个元素是一个元组(row_ind, col_ind)，
                         代表匹配上的(预测索引, 真实标签索引)。
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # 1. 计算代价矩阵 (Cost Matrix) --> [B, num_queries, num_target_boxes]

        # 将预测结果展平，方便后续计算
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [B * num_queries, num_classes + 1]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [B * num_queries, 4]

        # 将target列表的所有labels和boxes拼接起来
        tgt_ids = torch.cat([v["labels"] for v in targets]) # [total_num_boxes]
        tgt_bbox = torch.cat([v["boxes"] for v in targets]) # [total_num_boxes, 4]

        # 计算类别代价 (Classification Cost)
        # out_prob[:, tgt_ids]可以得到每个查询对应的真实类别的概率,为了适配匈牙利算法，使用-号将概率转换为"代价"
        cost_class = -out_prob[:, tgt_ids]

        # 计算L1 BBox代价 (L1 BBox Cost)
        # cdist计算所有预测框和所有真实框之间的L1距离
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1) # [B * num_queries, total_num_boxes]

        # 计算GIoU BBox代价 (GIoU BBox Cost)
        cost_giou = -generalized_box_iou(cxcywh_to_xyxy(out_bbox), cxcywh_to_xyxy(tgt_bbox)) # [B * num_queries, total_num_boxes]

        # 最终代价是三者的加权和
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou # [B * num_queries, total_num_boxes]
        C = C.view(bs, num_queries, -1).cpu() # [B, num_queries, total_num_boxes]

        # 2. 匈牙利算法求解最优匹配
        sizes = [len(v["boxes"]) for v in targets] # 每个batch中目标框的数量

        # linear_sum_assignment的输入必须是[num_queries, total_num_boxes]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        
        # 3. 整理返回结果
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]