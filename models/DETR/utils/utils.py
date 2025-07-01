from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.DETR.model.backbone import Backbone
from models.DETR.model.transformer import Transformer
from models.DETR.model.detr import DETR
from models.DETR.model.matcher import HungarianMatcher
from models.DETR.model.criterion import SetCriterion


class NestedTensor(object):
    """
    一个包装类，将一个tensor和它的padding mask包装在一起。
    """
    def __init__(self, tensors, mask: Optional[Tensor] = None):
        self.tensors = tensors
        self.mask = mask

    def to(self, device: torch.device, dtype: torch.dtype = None):
        # type: (torch.device, torch.dtype) -> NestedTensor
        """
        将tensor和mask都转换到指定的设备和数据类型。
        """
        return NestedTensor(self.tensors.to(device, dtype), self.mask)
        
    def decompose(self):
        """
        将NestedTensor分解为tensor和mask。
        """
        return self.tensors, self.mask
    
    @property
    def shape(self):
        # tensors: [B, C, H, W]
        # mask: [B, H, W]
        assert self.tensors.shape[-2:] == self.mask.shape[-2:]
        return self.tensors.shape
    
def cxcywh_to_xyxy(x: Tensor) -> Tensor:
    """
    [cx, cy, w, h] -> [x1, y1, x2, y2]
    """
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """
    计算两个box的IoU。
    Args:
        boxes1: 形状为(N, 4)的tensor，表示N个box。
        boxes2: 形状为(M, 4)的tensor，表示M个box。
    Returns:
        iou: 形状为(N, M)的tensor，表示N个box和M个box的IoU。
        union: 形状为(N, M)的tensor，表示N个box和M个box的并集。
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1]) # [N]
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1]) # [M]
    # 找到交集框的位置
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2]) # [N, M, 2] 即每个boxes1与每个boxes2都对比一次，找到交集框的左上角
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) # [N, M, 2] 同理找到交集框的右下角

    wh = (rb - lt).clamp(min=0) # clamp(min=0) 确保wh不会为负
    inter = wh[:, :, 0] * wh[:, :, 1] # [N, M] 即每个boxes1与每个boxes2的交集面积
    
    union = area1[:, None] + area2 - inter # [N, M]
    iou = inter / (union + 1e-6)
    return iou, union
    
def generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """
    计算两个box的Generalized IoU(广义交并比)。
    Args:
        boxes1: 形状为(N, 4)的tensor，表示N个box。
        boxes2: 形状为(M, 4)的tensor，表示M个box。
    Returns:
        GIoU: 形状为(N, M)的tensor，表示N个box和M个box的广义IoU。
    """
    # 确保boxes1和boxes2都不为空
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)
    # 注意这里计算方式与上面有所不同，这里是找到最小闭包框
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0) # clamp(min=0) 确保wh不会为负
    area = wh[:, :, 0] * wh[:, :, 1]
    return iou - (area - union) / (area + 1e-6)

def build_model(args):
    """构建DETR模型"""
    # 构建backbone
    backbone = Backbone(
        name='resnet50',
        train_backbone=True,
        return_interm_layers=False,
        dilation=False
    )
    
    # 构建transformer
    transformer = Transformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=2048,
        dropout=args.dropout,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=True
    )
    
    # 构建DETR模型
    model = DETR(
        backbone=backbone,
        transformer=transformer,
        num_classes=20,  # Pascal VOC有20个类别
        num_queries=args.num_queries,
        return_intermediate_dec=True
    )
    
    return model


def build_criterion(args):
    """构建损失函数"""
    # 构建匈牙利匹配器
    matcher = HungarianMatcher(
        cost_class=args.set_cost_class,
        cost_bbox=args.set_cost_bbox,
        cost_giou=args.set_cost_giou
    )
    
    # 损失权重
    weight_dict = {
        'loss_ce': args.loss_ce,
        'loss_bbox': args.loss_bbox,
        'loss_giou': args.loss_giou
    }
    
    # 添加辅助损失权重
    aux_weight_dict = {}
    for i in range(args.num_decoder_layers - 1):
        aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
    weight_dict.update(aux_weight_dict)
    
    # 构建损失函数
    criterion = SetCriterion(
        num_classes=20,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=args.eos_coef
    )
    
    return criterion
