import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple

from ..utils.utils import cxcywh_to_xyxy, generalized_box_iou
from .matcher import HungarianMatcher

class SetCriterion(nn.Module):
    """
    DETR损失函数类。

    该类通过匈牙利匹配器将模型的预测结果与真实标签进行一对一匹配，
    然后计算三项损失：
    1. 类别损失 (Classification Loss): 使用交叉熵，惩罚错误的类别预测。
       对于未匹配的预测，其目标是 "无目标" (no object) 类别。
    2. L1 边界框损失 (L1 Box Loss): 惩罚匹配上的预测框与真实框之间的 L1 距离。
    3. GIoU 边界框损失 (Generalized IoU Loss): 惩罚匹配上的预测框与真实框之间的 GIoU 距离，
       它对框的尺度不敏感，且能更好地处理框不重叠的情况。
    """
    def __init__(self, num_classes: int, matcher: HungarianMatcher, weight_dict: Dict, eos_coef: float):
        """
        初始化损失函数。

        Args:
            num_classes (int): 数据集中的类别总数 (不包括背景或"无目标"类)。
            matcher (HungarianMatcher): 用于在预测和真实目标之间寻找最佳匹配的模块。
            weight_dict (Dict): 一个字典，包含 'loss_ce', 'loss_bbox', 'loss_giou' 等键，值为它们各自的权重。
            eos_coef (float): "无目标" (end-of-sentence) 类别的分类损失相对权重。
                              由于匹配上的框（正样本）远少于未匹配的框（负样本），
                              使用一个较低的权重可以防止模型过于倾向于预测“无目标”，从而稳定训练。
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        
        # 定义一个特殊的权重张量，用于处理类别不平衡问题
        # "无目标"类别的权重是eos_coef，其他所有真实类别的权重都是1.0
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        # register_buffer将其注册为模型状态的一部分，但不是模型参数（即不会被优化器更新）
        # 这确保了它会随着模型一起被移动到GPU/CPU
        self.register_buffer('empty_weight', empty_weight)

    def _get_permutation_indices(self, indices: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将匹配器输出的索引列表转换为可用于张量索引的格式。

        Args:
            indices (List[Tuple]): 匈牙利匹配器的输出。
                                   一个列表，长度为batch_size。每个元素是一个元组 (src_idx, tgt_idx)，
                                   分别表示匹配上的预测索引和目标索引。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 两个扁平化的张量，第一个是batch内各样本的索引，第二个是匹配上的预测索引。
        """
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_target_classes(self, outputs: Dict, targets: List[Dict], indices: List[Tuple]) -> torch.Tensor:
        """
        构建用于分类损失的目标张量。

        所有未匹配的预测查询，其目标类别都将被设为 "无目标" 类 (self.num_classes)。
        匹配上的预测查询，其目标类别为对应的真实物体类别。
        """
        # 1. 获取匹配上的预测的索引 (batch_idx, src_idx)
        batch_idx, src_idx = self._get_permutation_indices(indices)
        
        # 2. 获取这些预测对应的真实目标的类别
        #    zip(targets, indices) -> (单个样本的target字典, 单个样本的匹配结果(src, tgt))
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        
        # 3. 创建一个形状为 [B, num_queries] 的张量，并用 "无目标" 类别填充，作为赋值的背景板，没有匹配的查询默认对应“”无目标”类别。
        src_logits = outputs['pred_logits']
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        
        # 4. 在目标张量的对应位置填上真实类别
        target_classes[batch_idx, src_idx] = target_classes_o # [B, num_queries]
        return target_classes

    def _get_matched_outputs_and_targets(self, outputs: Dict, targets: List[Dict], indices: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取匹配上的预测框和对应的真实框。"""
        # 1. 获取匹配上的预测的索引
        batch_idx, src_idx = self._get_permutation_indices(indices)
        
        # 2. 根据索引选出匹配上的预测框
        matched_pred_boxes = outputs['pred_boxes'][batch_idx, src_idx] # [total_num_matches, 4]
        
        # 3. 选出匹配上的真实框
        matched_target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0) # [total_num_matches, 4]
        
        return matched_pred_boxes, matched_target_boxes

    def calculate_losses(self, outputs: Dict, targets: List[Dict], indices: List[Tuple], num_boxes: int) -> Dict[str, torch.Tensor]:
        """
        为单层网络输出计算所有损失项。

        Args:
            outputs (Dict): 模型单层的输出，包含 'pred_logits' 和 'pred_boxes'。
            targets (List[Dict]): 真实标签列表，包含'labels' 和 'boxes'。
            indices (List[Tuple]): 该层输出对应的匹配结果。
            num_boxes (int): 用于归一化的总目标框数。

        Returns:
            Dict[str, torch.Tensor]: 包含该层各项损失的字典。
        """
        # --- 1. 类别损失 (Classification Loss) ---
        src_logits = outputs['pred_logits']
        target_classes = self._get_target_classes(outputs, targets, indices)
        # PyTorch的CrossEntropyLoss期望的输入形状是 (N, C, ...) 和 (N, ...)，所以需要转置
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)

        # --- 2. 边界框损失 (L1 Loss 和 GIoU Loss) ---
        matched_pred_boxes, matched_target_boxes = self._get_matched_outputs_and_targets(outputs, targets, indices)
        
        # L1 损失
        loss_bbox = F.l1_loss(matched_pred_boxes, matched_target_boxes, reduction='none')
        
        # GIoU 损失
        # 注意：diag只取对角线元素，因为我们只关心匹配上的 (pred, target) 对的IoU
        loss_giou = 1 - torch.diag(generalized_box_iou(
            cxcywh_to_xyxy(matched_pred_boxes),
            cxcywh_to_xyxy(matched_target_boxes)
        ))
        
        # 将各项损失打包并按目标框总数进行归一化
        losses = {}
        losses['loss_ce'] = loss_ce
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        
        return losses

    def forward(self, outputs: Dict, targets: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        执行完整的前向损失计算。

        Args:
            outputs (Dict): 模型的完整输出，可能包含 'aux_outputs' (辅助损失)。
            targets (List[Dict]): 真实标签列表。

        Returns:
            Dict[str, torch.Tensor]: 包含所有损失项（包括辅助损失）的字典。
        """
        # 1. 仅对解码器最后一层的输出进行匹配，以获得最终的匹配结果
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        indices = self.matcher(outputs_without_aux, targets)

        # 2. 计算Batch中所有目标框的总数，用于归一化损失。
        #    这使得损失的大小与batch中物体的数量无关。
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes_tensor = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        # TODO: 在分布式训练中，需要同步所有GPU上的num_boxes_tensor以获得全局总数

        # 3. 计算最后一层的损失
        losses = self.calculate_losses(outputs, targets, indices, num_boxes)

        # 4. 如果存在辅助输出 (来自解码器的中间层)，则为它们计算辅助损失
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                # 每一层都需要独立的匹配
                aux_indices = self.matcher(aux_outputs, targets)
                aux_losses = self.calculate_losses(aux_outputs, targets, aux_indices, num_boxes)
                
                # 为辅助损失的键名添加后缀，以便区分
                # 例如 'loss_ce' -> 'loss_ce_0', 'loss_ce_1', ...
                renamed_aux_losses = {k + f'_{i}': v for k, v in aux_losses.items()}
                losses.update(renamed_aux_losses)

        return losses