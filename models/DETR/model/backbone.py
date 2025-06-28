import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models._utils import IntermediateLayerGetter
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from DETR.utils.utils import NestedTensor

class Backbone(nn.Module):
    def __init__(self, name: str = "resnet50", train_backbone: bool = True, return_interm_layers: bool = False, dilation: bool = False):
        """
        Args:
            name: 使用的backbone模型，如"resnet50"
            train_backbone: 是否训练backbone
            return_interm_layers: 是否返回中间层的特征
            dilation: 是否使用dilation卷积
        """
        super().__init__()
        # 记录输出通道数和步长
        self.num_channels = 2048  # C = 2048
        self.strides = [32]

        # 使用torchvision.models的模型作为backbone
        backbone_fn = getattr(torchvision.models, name)
        
        self.body = backbone_fn(
            weights="IMAGENET1K_V2",
            replace_stride_with_dilation=[False, False, dilation],  # 对应layer2,3,4
            norm_layer=nn.BatchNorm2d
        )
        # 官方简洁写法，表示若训练backbone，则只训练layer2, layer3, layer4的参数
        for name, parameter in self.body.named_parameters():
            if not train_backbone or ('layer2' not in name and 'layer3' not in name and 'layer4' not in name):
                parameter.requires_grad_(False)
        
        # 为FPN设计，但是暂未使用，return_interm_layers=False
        if return_interm_layers:
            # 返回多个特征层用于FPN
            # layer1返回: [B, 256, 56, 56]
            # layer2返回: [B, 512, 28, 28]
            # layer3返回: [B, 1024, 14, 14]
            # layer4返回: [B, 2048, 7, 7]
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            # 只返回最后一层特征
            return_layers = {'layer4': "0"}
            
        self.body = IntermediateLayerGetter(self.body, return_layers=return_layers)
        
        
    def forward(self, x: NestedTensor):
        """
        Args:
            x: NestedTensor包含:
               - tensors: [B, 3, H, W] - 输入图像
               - mask: [B, H, W] - 填充掩码(1表示填充区域)
        
        Returns:
            out: 包含多个特征层的NestedTensor字典
               - 键为特征层名称，值为NestedTensor对象，包含:
                 - tensors: [B, C, H/32, W/32] - 特征图
                 - mask: [B, H/32, W/32] - 填充掩码
        """
        # 处理图像特征
        tensors = x.tensors
        features = self.body(tensors) # [B, C, H, W]
        
        # 处理mask
        out = {}
        for name, feature in features.items():
            # 获取特征图的空间尺寸
            _, _, h, w = feature.shape
            
            if x.mask is not None:
                # 下采样mask到特征图的尺寸，使用最近邻插值以保持二值性质
                mask = F.interpolate(x.mask[None].float(), size=(h, w), mode="nearest")[0].to(torch.bool)
            else:
                # 如果没有mask，创建一个全为False的mask
                mask = torch.zeros((h, w), dtype=torch.bool, device=feature.device)
            
            # 创建NestedTensor结果
            out[name] = NestedTensor(feature, mask)
        
        return out
                    
