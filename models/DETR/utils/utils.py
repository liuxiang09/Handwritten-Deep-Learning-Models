from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


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
    转换中心点-宽高格式的边界框 [cx, cy, w, h] 到左上右下角点格式 [x1, y1, x2, y2]
    这个函数假设输入的坐标已经归一化到[0,1]范围内
    """
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    # 确保坐标在[0,1]范围内
    result = torch.stack(b, dim=-1)
    return torch.clamp(result, min=0.0, max=1.0)

def xyxy_to_cxcywh(x: Tensor) -> Tensor:
    """
    [x1, y1, x2, y2] -> [cx, cy, w, h]
    """
    x1, y1, x2, y2 = x.unbind(-1)
    b = [(x1 + x2) / 2, (y1 + y2) / 2,
         (x2 - x1), (y2 - y1)]
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
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)
    # 注意这里计算方式与上面有所不同，这里是找到最小闭包框
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0) # clamp(min=0) 确保wh不会为负
    area = wh[:, :, 0] * wh[:, :, 1]
    return iou - (area - union) / (area + 1e-6)

def create_nested_tensor(tensors: Tensor, mask: Optional[Tensor] = None) -> NestedTensor:
    """
    创建一个NestedTensor对象。
    
    Args:
        tensors: 输入的张量，形状为[B, C, H, W]。
        mask: 可选的掩码，形状为[B, H, W]。
    
    Returns:
        NestedTensor对象。
    """
    if mask is None:
        mask = torch.zeros(tensors.shape[:2], dtype=torch.bool, device=tensors.device)
    return NestedTensor(tensors=tensors, mask=mask)