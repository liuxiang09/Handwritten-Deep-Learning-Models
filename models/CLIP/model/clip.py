import torch
import torch.nn as nn

class CLIP(nn.Module):
    def __init__(
            self,
            image_encoder: nn.Module,
            text_encoder: nn.Module,
            embed_dim: int
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder

        # 获取编码器输出的维度
        # 假设ViT和TextEncoder输出维度已是embed_dim
        # ResNet输出维度是2048，需要投影
        # 为简化，我们假设所有编码器直接输出 embed_dim
        # image_feat_dim = 2048 if isinstance(image_encoder, ModifiedResNet) else embed_dim
        # text_feat_dim = embed_dim
        
        # self.image_projection = nn.Parameter(torch.empty(image_feat_dim, embed_dim))
        # self.text_projection = nn.Parameter(torch.empty(text_feat_dim, embed_dim))
        # nn.init.normal_(self.image_projection, std=embed_dim ** -0.5)
        # nn.init.normal_(self.text_projection, std=embed_dim ** -0.5)

        # 在我们上面的实现中，编码器已直接输出 embed_dim，所以不需要额外的线性投影层。
        # 如果需要，可以像注释中那样添加。
        
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592) # ln(1/0.07)

        def encode_image(self, image):
            return self.image_encoder(image)

        def encode_text(self, text):
            return self.text_encoder(text)
        
        def forward(self, image: torch.Tensor, text: torch.Tensor):
            image_features = self.encode_image(image)
            text_features = self.encode_text(text)

            # 归一化
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # logit_scale 是一个可学习的温度参数
            # scaled_logits = (image_features @ text_features.T) * self.logit_scale.exp()
            # 完整的相似度计算和 loss 在训练时使用，这里只返回特征
            return image_features, text_features

        
        
            
