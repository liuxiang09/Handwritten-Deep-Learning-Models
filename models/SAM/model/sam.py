import torch
from torch import nn
from torch.nn import functional as F

from typing import Any, Dict, List, Tuple

from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder


class Sam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"
    def __init__(self,
                 image_encoder: ImageEncoderViT,
                 prompt_encoder: PromptEncoder,
                 mask_decoder: MaskDecoder,
                 pixel_mean: List[float] = [123.675, 116.28, 103.53],
                 pixel_std: List[float] = [58.395, 57.12, 57.375]):
        """
        SAM从图像和输入提示预测对象掩码。

        Arguments:
          image_encoder (ImageEncoderViT): 用于将图像编码为图像嵌入的主干网络，以便高效进行掩码预测。
          prompt_encoder (PromptEncoder): 编码各种类型的输入提示。
          mask_decoder (MaskDecoder): 从图像嵌入和编码的提示预测掩码。
          pixel_mean (list(float)): 用于归一化输入图像像素的均值。
          pixel_std (list(float)): 用于归一化输入图像像素的标准差。
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self) -> Any:
        return self.pixel_mean.device
    

    def preprocess(self,
                   x: torch.Tensor) -> torch.Tensor:
        """归一化像素值并填充为正方形输入。"""
        # 归一化颜色
        x = (x - self.pixel_mean) / self.pixel_std
        
        # 填充到1024x1024大小
        h, w = x.shape[-2:]
        padh = self.image_encoder.image_size - h
        padw = self.image_encoder.image_size - w
        x = F.pad(x, (0, padw, 0, padh), mode="constant", value=0.0)
        return x
    
    def postprocess_masks(self,
                          masks: torch.Tensor,
                          input_size: Tuple[int, int],
                          original_size: Tuple[int, int]) -> torch.Tensor:
        """
        移除填充并将掩码上采样到原始图像大小。

        Arguments:
          masks (torch.Tensor): 来自mask_decoder的批量掩码，
            格式为BxCxHxW。
          input_size (tuple(int, int)): 输入到模型的图像大小，
            格式为(H, W)。用于移除填充。
          original_size (tuple(int, int)): 在调整大小输入到模型之前的原始图像大小，
            格式为(H, W)。

        Returns:
          (torch.Tensor): 格式为BxCxHxW的批量掩码，其中(H, W)
            由original_size给出。
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.image_size, self.image_encoder.image_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., :input_size[0], :input_size[1]]
        masks = F.interpolate(
            masks,
            original_size,
            mode="bilinear",
            align_corners=False,
        )
        return masks

    @torch.no_grad()
    def forward(self,
                batched_input: List[Dict[str, Any]],
                multimask_output: bool = False) -> List[Dict[str, torch.Tensor]]:
        """
        从提供的图像和提示端到端预测掩码。
        如果提示事先未知，建议使用SamPredictor而不是直接调用模型。

        Arguments:
          batched_input (list(dict)): 输入图像列表，每个都是
            具有以下键的字典。如果不存在，可以排除提示键。
              'image': 作为torch张量的图像，格式为3xHxW，
                已为输入到模型进行转换。
              'original_size': (tuple(int, int)) 转换前的原始图像大小，
                格式为(H, W)。
              'point_coords': (torch.Tensor) 此图像的批量点提示，
                形状为BxNx2。已转换到模型的输入框架。
              'point_labels': (torch.Tensor) 点提示的批量标签，
                形状为BxN。
              'boxes': (torch.Tensor) 批量框输入，形状为Bx4。
                已转换到模型的输入框架。
              'mask_inputs': (torch.Tensor) 模型的批量掩码输入，
                格式为Bx1xHxW。
          multimask_output (bool): 模型是否应预测多个消歧掩码，
            或返回单个掩码。

        Returns:
          (list(dict)): 输入图像列表，其中每个元素是
            具有以下键的字典。
              'masks': (torch.Tensor) 批量二进制掩码预测，
                形状为BxCxHxW，其中B是输入提示数量，
                C由multimask_output确定，(H, W)是原始图像大小。
              'iou_predictions': (torch.Tensor) 模型对掩码质量的预测，
                形状为BxC。
              'low_res_logits': (torch.Tensor) 低分辨率logits，
                形状为BxCxHxW，其中H=W=256。可以作为掩码输入
                传递给后续预测迭代。
        """
        input_images = torch.stack([self.preprocess(x['image']) for x in batched_input], dim=0)
        image_embeddings = self.image_encoder(input_images)

        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )

            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )

            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )
            masks = masks > self.mask_threshold
            outputs.append(
                {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                }
            )
        return outputs