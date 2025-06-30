import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
import copy

# ==============================================================================
# 辅助函数 (Helper Functions)
# ==============================================================================

def _get_clones(module, N):
    """
    创建N个相同的模块副本。
    Args:
        module (nn.Module): 需要复制的模块。
        N (int): 副本数量。
    Returns:
        nn.ModuleList: 包含N个深拷贝模块的ModuleList。
    """
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """
    根据字符串返回对应的激活函数。
    Args:
        activation (str): 激活函数名称（'relu', 'gelu', 'glu'）。
    Returns:
        Callable: 对应的激活函数。
    """
    if activation == "relu":
        return nn.ReLU()
    if activation == "gelu":
        return nn.GELU()
    if activation == "glu":
        return nn.GLU()
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


# ==============================================================================
# Transformer 核心层 (Core Layers)
# ==============================================================================

class EncoderLayer(nn.Module):
    """
    Transformer Encoder的基础层。
    包含：自注意力 (Self-Attention) -> Add&Norm -> 前馈网络 (FFN) -> Add&Norm
    Args:
        d_model (int): 特征维度。
        nhead (int): 多头注意力头数。
        dim_feedforward (int): 前馈网络隐藏层维度。
        dropout (float): dropout概率。
        activation (str): 激活函数类型。
    """
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1, activation: str = "relu"):
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
        """
        将位置编码添加到输入张量中。
        """
        return tensor + pos

    def forward(self,
                src: torch.Tensor,
                pos: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None):
        """
        前向传播。
        Args:
            src (Tensor): 输入特征。[H*W, B, D]
            pos (Tensor): 位置编码。[H*W, B, D]
            src_mask (Tensor, optional): 注意力mask。
            src_key_padding_mask (Tensor, optional): padding mask。
        Returns:
            Tensor: 输出特征。[H*W, B, D]
        """
        # 在输入给自注意力前加上位置编码
        q = k = self.add_pos_encoding(src, pos)
        # 自注意力
        src2 = self.self_attn(query=q, 
                              key=k, 
                              value=src, 
                              attn_mask=src_mask, 
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
    Args:
        d_model (int): 特征维度。
        nhead (int): 多头注意力头数。
        dim_feedforward (int): 前馈网络隐藏层维度。
        dropout (float): dropout概率。
        activation (str): 激活函数类型。
    """
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1, activation: str = "relu"):
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
        """
        将位置编码添加到输入张量中。
        """
        return tensor + pos

    def forward(self, 
                tgt: torch.Tensor, 
                memory: torch.Tensor,
                query_pos: torch.Tensor,
                pos: torch.Tensor, 
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None):
        """
        前向传播。
        Args:
            tgt (Tensor): Decoder输入。[num_queries, B, D]
            memory (Tensor): Encoder输出。[H*W, B, D]
            query_pos (Tensor): Object Queries的位置编码。[num_queries, B, D]
            pos (Tensor): 图像特征的位置编码。[H*W, B, D]
            tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask: mask参数。
        Returns:
            Tensor: Decoder输出。[num_queries, B, D]
        """
        # 1. Decoder自注意力 (Q, K, V都是Object Queries)
        # 在输入给注意力层之前，给Q和K加上Object Queries的位置编码(query_pos)
        q = k = self.add_pos_encoding(tgt, query_pos)
        tgt2 = self.self_attn(query=q, 
                              key=k, 
                              value=tgt, 
                              attn_mask=tgt_mask, 
                              key_padding_mask=tgt_key_padding_mask)[0]
        # Add & Norm
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # 2. 交叉注意力 (Q来自上一步输出, K,V来自Encoder输出)
        # Q加上Object Queries的位置编码(query_pos)
        # K加上图像特征的位置编码(pos)
        tgt2 = self.multihead_attn(query=self.add_pos_encoding(tgt, query_pos),
                                   key=self.add_pos_encoding(memory, pos),
                                   value=memory, 
                                   attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0] # [num_queries, B, D]
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
    Args:
        encoder_layer (EncoderLayer): 单层Encoder。
        num_layers (int): 层数。
        norm (nn.Module, optional): 层归一化。
    """
    def __init__(self, 
                 encoder_layer: EncoderLayer, 
                 num_layers: int, 
                 norm: Optional[nn.Module] = None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, 
                src: torch.Tensor, 
                pos: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None):
        """
        前向传播。
        Args:
            src (Tensor): 输入特征。[H*W, B, D]
            pos (Tensor): 位置编码。[H*W, B, D]
            mask (Tensor, optional): 注意力mask。
            src_key_padding_mask (Tensor, optional): padding mask。
        Returns:
            Tensor: 输出特征。[H*W, B, D]
        """
        output = src
        # 多层EncoderLayer堆叠
        for layer in self.layers:
            output = layer(output, 
                           pos, 
                           src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output # [H*W, B, D]


class Decoder(nn.Module):
    """
    由多个DecoderLayer堆叠而成的完整Decoder。
    Args:
        decoder_layer (DecoderLayer): 单层Decoder。
        num_layers (int): 层数。
        norm (nn.Module, optional): 层归一化。
        return_intermediate_dec (bool): 是否返回所有中间层输出。
    """
    def __init__(self, 
                 decoder_layer: DecoderLayer, 
                 num_layers: int, 
                 norm: Optional[nn.Module] = None, 
                 return_intermediate_dec: bool = False):
        
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate_dec = return_intermediate_dec

    def forward(self, 
                tgt: torch.Tensor, 
                memory: torch.Tensor, 
                query_pos: torch.Tensor,
                pos: torch.Tensor, 
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None):
        """
        前向传播。
        Args:
            tgt (Tensor): Decoder输入。[num_queries, B, D]
            memory (Tensor): Encoder输出。[H*W, B, D]
            query_pos (Tensor): Object Queries的位置编码。[num_queries, B, D]
            pos (Tensor): 图像特征的位置编码。[H*W, B, D]
            tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask: mask参数。
        Returns:
            Tensor: Decoder输出。[num_layers, B, num_queries, D] 或 [1, B, num_queries, D]
        """
        output = tgt
        intermediate = []

        for layer in self.layers:
            output = layer(output, # [num_queries, B, D]
                           memory, 
                           query_pos=query_pos, 
                           pos=pos,
                           tgt_mask=tgt_mask, 
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask)
            
            if self.return_intermediate_dec:
                # 将每一层decoder的输出都保存下来
                intermediate.append(self.norm(output))
        # 应用层归一化（如果存在）
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate_dec:
                # 如果需要返回中间结果，则用最终归一化后的输出替换最后一个中间结果
                intermediate[-1] = output
        
        # 根据配置返回不同格式的结果
        if self.return_intermediate_dec:
            # 将所有中间层输出堆叠成一个张量 [num_layers, B, num_queries, D]
            return torch.stack(intermediate, dim=0)
        else:
            # 如果不需要中间结果，则只返回最后一层的输出 [1, B, num_queries, D]
            return output.unsqueeze(0)


# ==============================================================================
# 完整的 Transformer 主模块
# ==============================================================================

class Transformer(nn.Module):
    def __init__(self, 
                 d_model: int = 256, # 原始论文就是d_model=256
                 nhead: int = 8, 
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, 
                 dim_feedforward: int = 2048, # DETR原始论文中，并没有设置为d_model的4倍，而是2048
                 dropout: float = 0.1,
                 activation: str = "relu",  # 激活函数
                 return_intermediate_dec: bool = False): # 是否返回解码器中间结果
        super().__init__()

        # --- Encoder ---
        encoder_layer = EncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = Encoder(encoder_layer, num_encoder_layers, norm=encoder_norm)

        # --- Decoder ---
        decoder_layer = DecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = Decoder(decoder_layer, num_decoder_layers, norm=decoder_norm, return_intermediate_dec=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        """
        初始化所有参数。
        """
        for p in self.parameters():
            # dim>1 说明不是bias
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: torch.Tensor, mask: torch.Tensor, query_embed: torch.Tensor, pos_embed: torch.Tensor):
        """
        Transformer前向传播。
        Args:
            src (Tensor): Backbone的输出特征. [B, D, H, W]
            mask (Tensor): 用于区分padding的mask. [B, H, W]
            query_embed (Tensor): Object Queries, 可学习的查询向量. [num_queries, D]
            pos_embed (Tensor): 位置编码. [B, D, H, W]
        Returns:
            hs (Tensor): Decoder各层输出的集合. [num_layers, B, num_queries, D]
            memory (Tensor): Encoder最后一层的输出. [B, D, H, W]
        """
        # --- 数据预处理 ---
        # 将输入特征图和位置编码从 [B, D, H, W] 展平为 [H*W, B, D]
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)  # [H*W, B, D]
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1) # [H*W, B, D]
        
        # 将Object Queries从 [num_queries, D] 扩展为 [num_queries, B, D]
        # 这是Decoder的初始输入
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        
        # 将mask从 [B, H, W] 展平为 [B, H*W]
        mask = mask.flatten(1)

        # Decoder的初始输入，在DETR中是全零向量
        tgt = torch.zeros_like(query_embed)

        # --- Encoder前向传播 ---
        # memory 是编码后的图像特征 [H*W, B, D]
        memory = self.encoder(src, pos=pos_embed, src_key_padding_mask=mask)

        # --- Decoder前向传播 ---
        # hidden_states 是每一层decoder的输出，形状为 [num_layers 或 1, num_queries, B, D]
        # 关键：查询嵌入 query_embed 是以位置编码的形式注入到Decoder中的⚠
        hidden_states = self.decoder(tgt, memory, query_pos=query_embed, pos=pos_embed, memory_key_padding_mask=mask)
        
        # --- 输出格式整理 ---
        # hidden_states -> [num_layers 或 1, B, num_queries, D]
        # memory -> [B, D, H, W]
        return hidden_states.transpose(1, 2), memory.permute(1, 2, 0).contiguous().view(bs, c, h, w)
