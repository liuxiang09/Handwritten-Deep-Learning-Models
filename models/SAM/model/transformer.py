import torch
from torch import Tensor, nn

import math
from typing import Tuple, Type

from .common import MLPBlock

class Attention(nn.Module):
    """
    一个注意力层，允许在投影到查询、键和值后对嵌入大小进行下采样。
    """
    def __init__(self,
                 embedding_dim: int,
                 num_heads: int,
                 downsample_rate: int = 1,):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, \
            f"internal_dim {self.internal_dim} must be divisible by num_heads {num_heads}"

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        """
        Args:
            x: 形状为(B, num_tokens, internal_dim)的输入张量
            num_heads: 注意力头数。
        Returns:
            形状为(B, num_heads, num_tokens, internal_dim // num_heads)的张量
        """
        bs, n_head, dim = x.shape
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        x = x.reshape(bs, n_head, num_heads, dim // num_heads)
        return x.transpose(1, 2)  # (B, num_heads, num_tokens, internal_dim // num_heads)


    def _recombine_heads(self, x: Tensor) -> Tensor:
        """
        Args:
            x: 形状为(B, num_heads, num_tokens, internal_dim // num_heads)的输入张量
        Returns:
            形状为(B, num_tokens, internal_dim)的张量
        """
        bs, n_head, n_token, dim = x.shape
        return x.transpose(1, 2).reshape(bs, n_token, n_head * dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # 输入投影
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # 分离头
        q = self._separate_heads(q, self.num_heads) # [B, num_heads, num_tokens, internal_dim // num_heads]
        k = self._separate_heads(k, self.num_heads) # [B, num_heads, num_tokens, internal_dim // num_heads]
        v = self._separate_heads(v, self.num_heads) # [B, num_heads, num_tokens, internal_dim // num_heads]

        # 注意力
        dim_per_head = q.shape[-1]
        attn = q @ k.permute(0, 1, 3, 2) / math.sqrt(dim_per_head) # [B, num_heads, num_tokens, num_tokens]
        attn = attn.softmax(dim=-1)

        # 获取输出
        out = attn @ v # [B, num_heads, num_tokens, internal_dim // num_heads]
        out = self._recombine_heads(out)
        out = self.out_proj(out)
        return out # [B, num_tokens, internal_dim]
    
class TwoWayAttentionBlock(nn.Module):
    def __init__(self,
                 embedding_dim: int,
                 num_heads: int,
                 mlp_dim: int=2048,
                 activation: Type[nn.Module] = nn.ReLU,
                 attention_downsample_rate: int = 2,
                 skip_first_layer_pe: bool = False):
        """
        具有四层的Transformer块：(1)稀疏输入的自注意力，(2)稀疏输入到密集输入的交叉注意力，
        (3)稀疏输入的MLP块，(4)密集输入到稀疏输入的交叉注意力。

        Arguments:
          embedding_dim (int): 嵌入的通道维度
          num_heads (int): 注意力层中的头数
          mlp_dim (int): MLP块的隐藏维度
          activation (nn.Module): MLP块的激活函数
          attention_downsample_rate (int): 注意力层的下采样率，
            例如，2表示注意力层将输入下采样2倍。
          skip_first_layer_pe (bool): 跳过第一层的PE
        """
        super().__init__()
        self.skip_first_layer_pe = skip_first_layer_pe

        self.self_attn = Attention(embedding_dim, num_heads, downsample_rate=1)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, act=activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm4 = nn.LayerNorm(embedding_dim)

    def forward(self,
                queries: Tensor,
                keys: Tensor,
                query_pe: Tensor,
                key_pe: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            queries: 形状为(B, num_queries, embedding_dim)的输入张量
            keys: 形状为(B, num_keys, embedding_dim)的输入张量
            query_pe: 查询的位置编码，形状为(B, num_queries, embedding_dim)
            key_pe: 键的位置编码，形状为(B, num_keys, embedding_dim)
        """
        # 自注意力块
        if self.skip_first_layer_pe:
            queries = self.self_attn(queries, queries, queries)
        else:
            self_attn_out = self.self_attn(queries + query_pe, queries + query_pe, queries)
            queries = self_attn_out + queries # 残差连接
        queries = self.norm1(queries)

        # 交叉注意力块，令牌关注图像嵌入
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q, k, keys)
        queries = attn_out + queries # 残差连接
        queries = self.norm2(queries)

        # MLP块
        mlp_out = self.mlp(queries)
        queries = mlp_out + queries
        queries = self.norm3(queries)

        # 交叉注意力块，图像嵌入关注令牌
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(k, q, queries)
        keys = attn_out + keys  # 残差连接
        keys = self.norm4(keys)

        return queries, keys
    

class TwoWayTransformer(nn.Module):
    def __init__(self,
                 depth: int,
                 embedding_dim: int,
                 num_heads: int,
                 mlp_dim: int = 2048,
                 activation: Type[nn.Module] = nn.ReLU,
                 attention_downsample_rate: int = 2):
        """
        一个Transformer解码器，使用提供位置嵌入的查询来关注输入图像。

        Args:
          depth (int): Transformer的层数
          embedding_dim (int): 输入嵌入的通道维度
          num_heads (int): 多头注意力的头数。必须能整除embedding_dim
          mlp_dim (int): MLP块内部的通道维度
          activation (nn.Module): MLP块中使用的激活函数
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=self.embedding_dim,
                    num_heads=self.num_heads,
                    mlp_dim=self.mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0) # 跳过第一个块的第一层PE
                )
            )
        
        self.final_attn_token_to_image = Attention(
            embedding_dim=self.embedding_dim,
            num_heads=self.num_heads,
            downsample_rate=attention_downsample_rate
        )
        self.final_norm = nn.LayerNorm(self.embedding_dim)

    def forward(self,
                image_embedding: Tensor,
                image_pe: Tensor,
                point_embedding: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
          image_embedding (torch.Tensor): 要关注的图像。应为任意h和w的形状
            B x embedding_dim x h x w。
          image_pe (torch.Tensor): 要添加到图像的位置编码。必须
            与image_embedding具有相同的形状。
          point_embedding (torch.Tensor): 要添加到查询点的嵌入。
            对于任意N_points，必须具有形状B x N_points x embedding_dim。

        Returns:
          torch.Tensor: 处理后的point_embedding
          torch.Tensor: 处理后的image_embedding
        """
        # [B, embedding_dim, h, w] -> [B, h*w, embedding_dim]
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).transpose(1, 2)  # (bs, h*w, c)
        image_pe = image_pe.flatten(2).transpose(1, 2)

        # 准备查询和键
        queries = point_embedding
        keys = image_embedding
        
        # 应用Transformer块和最终层归一化
        for layer in self.layers:
            queries, keys = layer(queries, keys, query_pe=point_embedding, key_pe=image_pe)

        # 从点到图像应用最终注意力层
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = attn_out + queries  # 残差连接
        queries = self.final_norm(queries)

        return queries, keys  # queries是处理后的点嵌入，keys是处理后的图像嵌入