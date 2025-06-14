import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
import copy

# ==============================================================================
# 辅助函数 (Helper Functions)
# ==============================================================================

def _get_clones(module, N):
    """创建N个相同的模块副本"""
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """根据字符串返回对应的激活函数"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


# ==============================================================================
# Transformer 核心层 (Core Layers)
# ==============================================================================

class EncoderLayer(nn.Module):
    """
    Transformer Encoder的基础层。
    包含：自注意力 (Self-Attention) -> Add&Norm -> 前馈网络 (FFN) -> Add&Norm
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # 前馈网络 (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            _get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def add_pos_encoding(self, tensor: torch.Tensor, pos: torch.Tensor):
        """将位置编码添加到输入张量中。"""
        return tensor + pos

    def forward(
                self,
                src: torch.Tensor,
                pos: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None
        ):
        # 在输入给自注意力前加上位置编码
        q = k = self.add_pos_encoding(src, pos)
        # 自注意力
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        # ADD & Norm
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # 前馈网络
        src2 = self.ffn(src)
        # ADD & Norm
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class DecoderLayer(nn.Module):
    """
    Transformer Decoder的基础层。
    包含：自注意力 -> Add&Norm -> 交叉注意力 -> Add&Norm -> 前馈网络 -> Add&Norm
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu"):
        super().__init__()
        # Decoder的自注意力，关注对象是Object Queries
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Encoder-Decoder的交叉注意力
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # 前馈网络 (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            _get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def add_pos_encoding(self, tensor: torch.Tensor, pos: torch.Tensor):
        """将位置编码添加到输入张量中。"""
        return tensor + pos

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                     pos: torch.Tensor, query_pos: torch.Tensor,
                     tgt_mask: Optional[torch.Tensor] = None,
                     memory_mask: Optional[torch.Tensor] = None,
                     tgt_key_padding_mask: Optional[torch.Tensor] = None,
                     memory_key_padding_mask: Optional[torch.Tensor] = None):
        # 1. Decoder自注意力 (Q, K, V都是Object Queries)
        # 在输入给注意力层之前，给Q和K加上Object Queries的位置编码(query_pos)
        q = k = self.add_pos_encoding(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        # Add & Norm
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # 2. 交叉注意力 (Q来自上一步输出, K,V来自Encoder输出)
        # Q加上Object Queries的位置编码(query_pos)
        # K加上图像特征的位置编码(pos)
        tgt2 = self.multihead_attn(query=self.add_pos_encoding(tgt, query_pos),
                                   key=self.add_pos_encoding(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        # Add & Norm
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # 3. 前馈网络
        tgt2 = self.ffn(tgt)
        # Add & Norm
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


# ==============================================================================
# Encoder & Decoder 模块
# ==============================================================================

class Encoder(nn.Module):
    """
    由多个EncoderLayer堆叠而成的完整Encoder。
    """
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: torch.Tensor, pos: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, pos, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class Decoder(nn.Module):
    """
    由多个DecoderLayer堆叠而成的完整Decoder。
    """
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, 
                pos: torch.Tensor, query_pos: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None):
        output = tgt
        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, pos=pos, query_pos=query_pos, 
                           tgt_mask=tgt_mask, memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask)
            if self.return_intermediate:
                # 将每一层decoder的输出都保存下来
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


# ==============================================================================
# 完整的 Transformer 主模块
# ==============================================================================

class Transformer(nn.Module):
    """
    DETR中完整的Transformer模块，整合了Encoder和Decoder。
    """
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", return_intermediate_dec=False):
        super().__init__()

        # --- Encoder ---
        encoder_layer = EncoderLayer(d_model, nhead, dim_feedforward,
                                     dropout, activation)
        self.encoder = Encoder(encoder_layer, num_encoder_layers)

        # --- Decoder ---
        decoder_layer = DecoderLayer(d_model, nhead, dim_feedforward,
                                     dropout, activation)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = Decoder(decoder_layer, num_decoder_layers, decoder_norm,
                               return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        """
        Args:
            src (Tensor): Backbone的输出特征. [B, C, H, W]
            mask (Tensor): 用于区分padding的mask. [B, H, W]
            query_embed (Tensor): Object Queries, 可学习的查询向量. [num_queries, C]
            pos_embed (Tensor): 位置编码. [B, C, H, W]

        Returns:
            hs (Tensor): Decoder各层输出的集合. [num_layers, B, num_queries, C]
            memory (Tensor): Encoder最后一层的输出. [H*W, B, C]
        """
        # --- 数据预处理 ---
        # 将输入特征图和位置编码从 [B, C, H, W] 展平为 [H*W, B, C]
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)  # [H*W, B, C]
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1) # [H*W, B, C]
        
        # 将Object Queries从 [num_queries, C] 扩展为 [num_queries, B, C]
        # 这是Decoder的初始输入
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        
        # 将mask从 [B, H, W] 展平为 [B, H*W]
        mask = mask.flatten(1)

        # Decoder的初始输入，在DETR中是全零向量
        tgt = torch.zeros_like(query_embed)

        # --- Encoder前向传播 ---
        # memory 是编码后的图像特征
        memory = self.encoder(src, pos=pos_embed, src_key_padding_mask=mask)

        # --- Decoder前向传播 ---
        # hs 是每一层decoder的输出
        hs = self.decoder(tgt, memory, pos=pos_embed, query_pos=query_embed, memory_key_padding_mask=mask)
        
        # --- 输出格式整理 ---
        # hs: [num_layers, B, num_queries, C]
        # memory: [B, C, H, W]
        return hs, memory.permute(1, 2, 0).view(bs, c, h, w)


# ==============================================================================
# 构建函数 (Builder Function)
# ==============================================================================

def build_transformer(args):
    """根据参数构建Transformer模型"""
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        return_intermediate_dec=True,
    )
