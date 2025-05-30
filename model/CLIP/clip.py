import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer, CLIPTextModel, CLIPVisionModel
from tqdm import tqdm
import math


class CLIPContrastiveModel(nn.Module):
    """
    CLIP 对比学习模型，包含图像编码器、文本编码器和投影层。
    """
    def __init__(self, model_name, projection_dim):
        super().__init__()
        # 加载预训练的 CLIP 文本编码器和视觉编码器
        clip_model = CLIPModel.from_pretrained(model_name)
        self.text_encoder = clip_model.text_model
        self.vision_encoder = clip_model.vision_model

        # 投影层，将文本和图像特征投影到同一维度空间
        self.text_projection = nn.Linear(self.text_encoder.config.hidden_size, projection_dim)
        self.vision_projection = nn.Linear(self.vision_encoder.config.hidden_size, projection_dim)

        # 归一化层
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07)) # 初始化为 1/0.07，对应 CLIP 论文中的初始化

    def forward(self, input_ids, attention_mask, pixel_values):
        # 获取文本特征
        text_features = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        text_features = self.text_projection(text_features)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True) # 归一化

        # 获取图像特征
        vision_features = self.vision_encoder(pixel_values=pixel_values).pooler_output
        vision_features = self.vision_projection(vision_features)
        vision_features = vision_features / vision_features.norm(dim=-1, keepdim=True) # 归一化

        return text_features, vision_features, self.logit_scale.exp()
    
# 对比损失函数
class ContrastiveLoss(nn.Module):
    """
    CLIP 对比损失函数。
    计算图像和文本嵌入之间的对称交叉熵损失。
    """
    def __init__(self):
        super().__init__()

    def forward(self, image_features, text_features, logit_scale):
        # 计算相似度矩阵
        # (batch_size, projection_dim) x (projection_dim, batch_size) -> (batch_size, batch_size)
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logit_scale * text_features @ image_features.T

        # 构建目标标签
        # 对应位置是正样本，其他位置是负样本
        labels = torch.arange(len(image_features), device=image_features.device)

        # 计算交叉熵损失
        loss_i = nn.functional.cross_entropy(logits_per_image, labels) # 图像到文本的损失
        loss_t = nn.functional.cross_entropy(logits_per_text, labels) # 文本到图像的损失

        # 对称损失
        total_loss = (loss_i + loss_t) / 2
        return total_loss