import torch
import torch.nn as nn
from model.feedforward import FeedForward
from model.multihead_attention import MultiHeadAttention
from model.positional_encoding import TokenEmbedding, PositionalEncoding


class DecoderLayer(nn.Module):
    """
    Transformer 解码器单层实现
    包含：
    1. 遮盖式多头自注意力
    2. 多头交叉注意力（Encoder-Decoder Attention）
    3. 前馈神经网络
    每个子层后都有残差连接和层归一化
    """
    def __init__(self, d_model: int, n_head: int, d_ff: int, dropout: float):
        """
        初始化解码器层
        Args:
            d_model (int): 模型的隐藏维度
            n_head (int): 注意力头的数量
            d_ff (int): 前馈网络内部维度
            dropout (float): Dropout 概率
        """
        super(DecoderLayer, self).__init__()
        # 1.遮盖式多头注意力
        self.masked_self_attention = MultiHeadAttention(d_model, n_head)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        # 2.多头交叉注意力
        self.cross_attention = MultiHeadAttention(d_model, n_head)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        # 3.前馈神经网络
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, trg: torch.Tensor, enc_output: torch.Tensor, trg_mask: torch.Tensor = None, dec_enc_mask: torch.Tensor = None):
        """
        解码器层的前向传播
        Args:
            trg (torch.Tensor): 目标序列输入（已生成部分或完整目标序列），形状 (batch_size, seq_len_trg, d_model)
            enc_output (torch.Tensor): 编码器的最终输出，形状 (batch_size, seq_len_src, d_model)
            trg_mask (torch.Tensor, optional): 目标序列的遮盖掩码 (Look-Ahead Mask + Padding Mask)。
                                             形状 (batch_size, seq_len_trg, seq_len_trg)。
                                             True表示不屏蔽，False表示屏蔽。
            dec_enc_mask (torch.Tensor, optional): 源序列的填充掩码 (Padding Mask)，用于交叉注意力。
                                             形状 (batch_size, 1, 1, seq_len_src) 或 (batch_size, seq_len_src, seq_len_src)
                                             在 MultiHeadAttention 内部会处理为 (batch_size, 1, seq_len_q, seq_len_k)。
                                             True表示不屏蔽，False表示屏蔽。
        Returns:
            torch.Tensor: 解码器层的输出, 形状 (batch_size, seq_len_trg, d_model)
            torch.Tensor: 遮盖式自注意力权重, 形状 (batch_size, n_head, seq_len_trg, seq_len_trg)
            torch.Tensor: 交叉注意力权重, 形状 (batch_size, n_head, seq_len_trg, seq_len_src)
        """
        # 1.遮盖式多头自注意力 + Add&Norm
        # self_attn_output shape: (batch_size, seq_len_trg, d_model)
        # masked_self_attn_weights shape: (batch_size, n_head, seq_len_trg, seq_len_trg)
        residual = trg
        self_attn_output, masked_self_attn_weights = self.masked_self_attention(trg, trg, trg, trg_mask)
        self_attn_output = self.dropout1(self_attn_output)
        trg = self.norm1(residual + self_attn_output)
        # 2.多头交叉注意力 + Add&Norm
        # cross_attn_output shape: (batch_size, seq_len_trg, d_model)
        # cross_attn_weights shape: (batch_size, n_head, seq_len_trg, seq_len_src)
        residual = trg
        cross_attn_output, cross_attn_weights = self.cross_attention(trg, enc_output, enc_output, dec_enc_mask)
        cross_attn_output = self.dropout2(cross_attn_output)
        trg = self.norm2(residual + cross_attn_output)
        # 3.前馈神经网路 + Add&Norm
        residual = trg
        ffn_output = self.feed_forward(trg) # ffn_output shape: (batch_size, seq_len_trg, d_model)
        ffn_output = self.dropout3(ffn_output) 
        dec_output = self.norm3(residual + ffn_output) # decoder_output shape: (batch_size, seq_len_trg, d_model)
        # print(f"{'dec_output:':<{print_width}}{dec_output.shape}")
        return dec_output, masked_self_attn_weights, cross_attn_weights

class Decoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_head: int, d_ff: int, n_layer: int, dropout: float, max_len: int):
        """
        Transformer 解码器 (Decoder) 实现
        由 N 个相同的 DecoderLayer 堆叠而成。
        Args:
            vocab_size (int): 目标词汇表大小
            d_model (int): 模型的隐藏维度
            n_head (int): 多头注意力的头数
            d_ff (int): 前馈网络的内部隐藏维度
            num_layers (int): 解码器层的数量 (N)
            dropout_p (float): dropout 概率
            max_len (int): 最大序列长度，用于位置编码
        """
        super(Decoder, self).__init__()
        self.d_model = d_model

        # 1.词嵌入层
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        # 2.位置编码层
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len)
        # 3.堆叠的DecoderLayer
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_head, d_ff, dropout)
            for _ in range(n_layer)
        ])
        # 4.最终的线性层和 Softmax (用于生成词汇表概率)
        self.fc_out = nn.Linear(d_model, vocab_size)


    def forward(self, trg: torch.Tensor, enc_output: torch.Tensor, trg_mask: torch.Tensor, dec_enc_mask: torch.Tensor):
        """
        前向传播
        Args:
            trg (torch.Tensor): 目标序列的整数索引张量 (通常是已生成的部分)。
                                形状 (batch_size, seq_len_trg)。
            enc_output (torch.Tensor): 编码器的最终输出，形状 (batch_size, seq_len_src, d_model)。
            trg_mask (torch.Tensor): 目标序列的注意力掩码，包含 Look-Ahead Mask 和 Padding Mask。
                                    形状 (batch_size, seq_len_trg, seq_len_trg)。
                                    True表示不屏蔽，False表示屏蔽。
            dec_enc_mask (torch.Tensor): 源序列的填充掩码，用于交叉注意力。
                                    形状 (batch_size, 1, 1, seq_len_src) 或 (batch_size, seq_len_src, seq_len_src)。
                                    True表示不屏蔽，False表示屏蔽。
        Returns:
            torch.Tensor: 最终词汇表概率分布的对数，形状 (batch_size, seq_len_trg, vocab_size)。
                          通常会使用 CrossEntropyLoss，它内部会进行 Softmax。
            list[torch.Tensor]: 每层遮盖式自注意力权重列表。
                                每个元素的形状是 (batch_size, n_head, seq_len_trg, seq_len_trg)。
            list[torch.Tensor]: 每层交叉注意力权重列表。
                                每个元素的形状是 (batch_size, n_head, seq_len_trg, seq_len_src)。
        """
        # 1.词嵌入
        trg_emb = self.token_embedding(trg)
        # 2.位置编码
        trg_emb = self.pos_encoding(trg_emb.transpose(0, 1)).transpose(0, 1)

        all_masked_self_attention_weights = []
        all_cross_attention_weights = []
        # 3.逐层通过EncoderLayer
        dec_output = trg_emb
        for layer_idx, layer in enumerate(self.layers):
            # decoder_output shape: (batch_size, seq_len_trg, d_model)
            # layer_masked_self_attn_weights shape: (batch_size, n_head, seq_len_trg, seq_len_trg)
            # layer_cross_attn_weights shape: (batch_size, n_head, seq_len_trg, seq_len_src)
            dec_output, masked_self_attn_weights, cross_attn_weights = layer(dec_output, enc_output, trg_mask, dec_enc_mask)
            all_masked_self_attention_weights.append(masked_self_attn_weights)
            all_cross_attention_weights.append(cross_attn_weights)

        # 4.最终的线性层投影到词汇表大小
        # output_logits shape: (batch_size, seq_len_trg, vocab_size)
        output_logits = self.fc_out(dec_output)

        # output_logits 通常是 logits，在计算交叉熵损失时会自动应用 softmax
        return output_logits, all_masked_self_attention_weights, all_cross_attention_weights