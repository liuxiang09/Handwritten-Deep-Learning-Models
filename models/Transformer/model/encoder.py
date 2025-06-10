import torch
import torch.nn as nn
from model.feedforward import FeedForward
from model.multihead_attention import MultiHeadAttention
from model.positional_encoding import TokenEmbedding, PositionalEncoding

class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_head: int, d_ff: int, dropout: float):
        """
        初始化编码器层
        Args:
            d_model (int): 模型的隐藏维度 (例如 512)
            n_head (int): 多头注意力的头数 (例如 8)
            d_ff (int): 前馈网络的内部隐藏维度 (例如 2048)
            dropout_p (float): dropout 概率
        """
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, n_head)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor):
        """
        编码器层:src -> self_attention -> Add & Norm -> feed_forward -> Add & Norm -> enc_output
        Args:
            src (torch.Tensor): 输入张量 [batch_size, seq_len, d_model]
            src_mask (torch.Tensor): 掩码 [batch_szie, seq_len, seq_len],一般是padding mask
        Returns:
            enc_output (torch.Tensor): 编码器输出 [batch_size, seq_len, d_model]
            attention_weights (torch.Tensor): 注意力权重 [batch_size, n_head, seq_len, seq_len]
        """
        # 1.多头注意力子层 + Add&Norm
        residual = src
        # self_attention输出形状: [batch_size, seq_len, d_model]
        attn_output, attn_weights = self.self_attention(src, src, src, src_mask)
        normalized_attn_output = self.norm1(self.dropout1(attn_output) + residual)

        # 2.前馈神经网路 + Add&Norm
        residual = normalized_attn_output
        ffn_output = self.feed_forward(normalized_attn_output)
        enc_output = self.norm2(self.dropout2(ffn_output) + residual)
        # print(f"{'enc_output:':<{print_width}}{enc_output.shape}")
        return enc_output, attn_weights

class Encoder(nn.Module):
    """
    实现 Transformer 编码器 (Encoder)
    由 N 个相同的 EncoderLayer 堆叠而成。
    Args:
        vocab_size (int): 词汇表大小
        d_model (int): 模型的隐藏维度 (例如 512)
        n_head (int): 多头注意力的头数 (例如 8)
        d_ff (int): 前馈网络的内部隐藏维度 (例如 2048)
        n_layer (int): 编码器层的数量 (N)
        dropout_p (float): dropout 概率
        max_len (int): 最大序列长度，用于位置编码
    """
    def __init__(self, vocab_size: int, d_model: int, n_head: int, d_ff: int, n_layer: int, dropout: float, max_len: int):
        super(Encoder, self).__init__()
        self.d_model = d_model
        # 词嵌入层
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        # 位置编码层
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len)
        # 堆叠的EncoderLayer
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_head, d_ff, dropout)
            for _ in range(n_layer)
        ])

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor=None):
        """
        前向传播
        Args:
            src (torch.Tensor): 输入源序列的整数索引张量。
                                形状确定为 (batch_size, seq_len)。
            src_mask (torch.Tensor, optional): 源序列的注意力掩码。
                                             通常是 padding mask，形状 (batch_size, seq_len, seq_len)。
                                             用于屏蔽 padding token。
        Returns:
            torch.Tensor: 编码器的最终输出, 形状 (batch_size, seq_len, d_model)
            list[torch.Tensor]: 每层自注意力权重列表。
                                每个元素的形状是 (batch_size, n_head, seq_len, seq_len)。
        """
        # 1.词嵌入 
        # [batch_size, seq_len] -> [batch_size, seq_len, d_model]
        x = self.token_embedding(src) 
        # 2.位置编码(形状不变)
        # [batch_size, seq_len, d_model] -> [batch_size, seq_len, d_model]
        x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)
        # 3.逐层通过EncoderLayer
        all_self_attn_weights = []
        for layer_idx, layer in enumerate(self.layers):
            x, layer_attn_weights = layer(x, src_mask)
            all_self_attn_weights.append(layer_attn_weights)
        return x, all_self_attn_weights