import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type

from .common import LayerNorm2d


class MLP(nn.Module):
    """
    具有可配置层数和激活函数的灵活MLP模块。
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int,
                 sigmoid_output: bool = False):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


class MaskDecoder(nn.Module):
    def __init__(self,
                  *,
                  transformer_dim: int,
                  transformer: nn.Module,
                  num_multimask_outputs: int = 3,
                  activation: Type[nn.Module] = nn.GELU,
                  iou_head_depth: int = 3,
                  iou_head_hidden_dim: int = 256,):
        """
        给定图像和提示嵌入，使用Transformer架构预测掩码。

        Arguments:
          transformer_dim (int): Transformer的通道维度
          transformer (nn.Module): 用于预测掩码的Transformer
          num_multimask_outputs (int): 在消歧掩码时要预测的掩码数量
          activation (nn.Module): 上采样掩码时使用的激活函数类型
          iou_head_depth (int): 用于预测掩码质量的MLP深度
          iou_head_hidden_dim (int): 用于预测掩码质量的MLP隐藏维度
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs
        
        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1 # +1为主掩码令牌
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        # [B, transformer_dim, embed_H, embed_W] -> [B, transformer_dim//8, embed_H*4, embed_W*4]
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim//4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim//4),
            activation(),
            nn.ConvTranspose2d(transformer_dim//4, transformer_dim//8, kernel_size=2, stride=2),
            activation()
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, num_layers=3)
                for i in range(self.num_mask_tokens)
            ]
        )
        self.iou_prediction_head = MLP(
            transformer_dim,
            iou_head_hidden_dim,
            self.num_mask_tokens,
            num_layers=iou_head_depth,
        )


    def predict_masks(self,
                      image_embeddings: torch.Tensor,
                      image_pe:torch.Tensor,
                      sparse_prompt_embeddings: torch.Tensor,
                      dense_prompt_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        预测掩码。详见'forward'方法。
        一张图像 --- 多个提示（点、框、掩码）--> 多个掩码。
        Args:
            image_embeddings (torch.Tensor): 来自图像编码器的嵌入，形状为[1, transformer_dim, embed_H, embed_W]
            image_pe (torch.Tensor): 位置编码，形状为[B, transformer_dim, embed_H, embed_W]
            sparse_prompt_embeddings (torch.Tensor): 点和框的嵌入，形状为[B, N, transformer_dim]
            dense_prompt_embeddings (torch.Tensor): 掩码输入的嵌入，形状为[B, transformer_dim, embed_H, embed_W]
        """
        # 连接输出令牌
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0) # [1+num_mask_tokens, transformer_dim]
        output_tokens = output_tokens.unsqueeze(0).repeat(sparse_prompt_embeddings.shape[0], 1, 1) # [B, 1+num_mask_tokens, transformer_dim]
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)  # [B, 1+num_mask_tokens+N, transformer_dim]

        # 在批次方向上扩展每图像数据以成为每掩码
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0) # [B, transformer_dim, embed_H, embed_W]
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0) # [B, transformer_dim, embed_H, embed_W]
        b, c, h, w = src.shape

        # 运行transformer
        # hs: [B, 1+num_mask_tokens+N, transformer_dim]
        # src: [B, embed_H*embed_W, transformer_dim] (粗糙掩码)
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :] # [B, transformer_dim]
        mask_tokens_out = hs[:, 1:(1+self.num_mask_tokens), :] # [B, num_mask_tokens, transformer_dim]

        # 上采样掩码嵌入并使用掩码令牌预测掩码
        src = src.transpose(1, 2).view(b, c, h, w) # [B, transformer_dim, embed_H, embed_W]
        upscaled_embedding = self.output_upscaling(src) # [B, transformer_dim//8, embed_H*4, embed_W*4] (精细掩码)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])) # [B, transformer_dim//8]
        hyper_in = torch.stack(hyper_in_list, dim=1) # [B, num_mask_tokens, transformer_dim//8]
        b, c, h, w = upscaled_embedding.shape

        # [B, num_mask_tokens, transformer_dim//8] @ [B, transformer_dim//8, embed_H*4*embed_W*4] -> [B, num_mask_tokens, embed_H*4, embed_W*4]
        masks = (hyper_in @ upscaled_embedding.view(b, c, h*w)).view(b, -1, h, w) # (最终掩码)
        # 生成掩码质量预测
        iou_pred = self.iou_prediction_head(iou_token_out) # [B, num_mask_tokens]
        return masks, iou_pred
    

    def forward(self,
                image_embeddings: torch.Tensor,
                image_pe: torch.Tensor,
                sparse_prompt_embeddings: torch.Tensor,
                dense_prompt_embeddings: torch.Tensor,
                multimask_output: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        给定图像和提示嵌入预测掩码。

        Arguments:
          image_embeddings (torch.Tensor): 来自图像编码器的嵌入
          image_pe (torch.Tensor): 与image_embeddings形状相同的位置编码
          sparse_prompt_embeddings (torch.Tensor): 点和框的嵌入
          dense_prompt_embeddings (torch.Tensor): 掩码输入的嵌入
          multimask_output (bool): 是否返回多个掩码或单个掩码。

        Returns:
          torch.Tensor: 批量预测的掩码
          torch.Tensor: 批量掩码质量预测
        """
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # 选择正确的掩码进行输出
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        # 准备输出
        return masks, iou_pred
