import torch
from torch import Tensor, nn

import math
from typing import Tuple, Type

from .common import MLPBlock

class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
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
            x: Input tensor of shape (B, num_tokens, internal_dim)
            num_heads: Number of attention heads.
        Returns:
            Tensor of shape (B, num_heads, num_tokens, internal_dim // num_heads)
        """
        bs, n_head, dim = x.shape
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        x = x.reshape(bs, n_head, num_heads, dim // num_heads)
        return x.transpose(1, 2)  # (B, num_heads, num_tokens, internal_dim // num_heads)


    def _recombine_heads(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape (B, num_heads, num_tokens, internal_dim // num_heads)
        Returns:
            Tensor of shape (B, num_tokens, internal_dim)
        """
        bs, n_head, n_token, dim = x.shape
        return x.transpose(1, 2).reshape(bs, n_token, n_head * dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input Projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Seperate heads
        q = self._separate_heads(q, self.num_heads) # [B, num_heads, num_tokens, internal_dim // num_heads]
        k = self._separate_heads(k, self.num_heads) # [B, num_heads, num_tokens, internal_dim // num_heads]
        v = self._separate_heads(v, self.num_heads) # [B, num_heads, num_tokens, internal_dim // num_heads]

        # Attention
        dim_per_head = q.shape[-1]
        attn = q @ k.permute(0, 1, 3, 2) / math.sqrt(dim_per_head) # [B, num_heads, num_tokens, num_tokens]
        attn = attn.softmax(dim=-1)

        # Get output
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
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          attention_downsample_rate (int): the downsample rate of the attention
            layers, e.g., 2 means that the attention layer will downsample the
            input by a factor of 2.
          skip_first_layer_pe (bool): skip the PE on the first layer
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
            queries: Input tensor of shape (B, num_queries, embedding_dim)
            keys: Input tensor of shape (B, num_keys, embedding_dim)
            query_pe: Positional encoding for queries of shape (B, num_queries, embedding_dim)
            key_pe: Positional encoding for keys of shape (B, num_keys, embedding_dim)
        """
        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn(queries, queries, queries)
        else:
            self_attn_out = self.self_attn(queries + query_pe, queries + query_pe, queries)
            queries = self_attn_out + queries # residual connection
        queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q, k, keys)
        queries = attn_out + queries # residual connection
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = mlp_out + queries
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(k, q, queries)
        keys = attn_out + keys  # residual connection
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
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
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
                    skip_first_layer_pe=(i == 0) # skip the first Block's first layer PE
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
          image_embedding (torch.Tensor): image to attend to. Should be shape
            B x embedding_dim x h x w for any h and w.
          image_pe (torch.Tensor): the positional encoding to add to the image. Must
            have the same shape as image_embedding.
          point_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.

        Returns:
          torch.Tensor: the processed point_embedding
          torch.Tensor: the processed image_embedding
        """
        # [B, embedding_dim, h, w] -> [B, h*w, embedding_dim]
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).transpose(1, 2)  # (bs, h*w, c)
        image_pe = image_pe.flatten(2).transpose(1, 2)

        # Prepare queries and keys
        queries = point_embedding
        keys = image_embedding
        
        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries, keys = layer(queries, keys, query_pe=point_embedding, key_pe=image_pe)

        # Apply the final attention layer from the points to the image
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = attn_out + queries  # residual connection
        queries = self.final_norm(queries)

        return queries, keys  # queries are the processed point embeddings, keys are the processed image embeddings