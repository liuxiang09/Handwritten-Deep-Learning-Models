import torch
import torch.nn as nn

class CLIP(nn.Module):
    def __init__(
            self,
            image_encoder: nn.Module,
            text_encoder: nn.Module,
            vision_feature_dim: int, # 视觉编码器输出维度
            text_feature_dim: int,   # 文本编码器输出维度
            embed_dim: int           # 最终共享嵌入维度
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder

        # 线性投影层，将编码器输出投影到共享嵌入空间
        self.image_projection = nn.Linear(vision_feature_dim, embed_dim)
        self.text_projection = nn.Linear(text_feature_dim, embed_dim)
        
        self.logit_scale = nn.Parameter(torch.tensor(2.6592)) # ln(1/0.07)

    def encode_image(self, image):
        return self.image_projection(self.image_encoder(image))

    def encode_text(self, text):
        return self.text_projection(self.text_encoder(text))
        
    def forward(self, image: torch.Tensor, text: torch.Tensor):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # 归一化
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return image_features, text_features

        
        
            
