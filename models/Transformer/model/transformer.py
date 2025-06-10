import torch
import torch.nn as nn
from model.encoder import Encoder
from model.decoder import Decoder
from utils.utils import *
from configs.config import *

class Transformer(nn.Module):
    """
    Transformer 模型实现
    Args:
        src_vocab_size (int): 源语言词汇表大小
        trg_vocab_size (int): 目标语言词汇表大小
        d_model (int): 模型的隐藏维度 (默认 512)
        n_head (int): 多头注意力的头数 (默认 8)
        d_ff (int): 前馈网络的内部隐藏维度 (默认 2048)
        num_encoder_layers (int): 编码器层的数量 (默认 6)
        num_decoder_layers (int): 解码器层的数量 (默认 6)
        dropout_p (float): Dropout 概率 (默认 0.1)
        max_len (int): 最大序列长度，用于位置编码 (默认 5000)
        pad_idx (int): 填充 token 的索引，用于生成掩码 (默认 0)
    """
    def __init__(self, 
                 src_vocab_size: int, 
                 trg_vocab_size: int,
                 d_model: int = 512,
                 n_head: int = 8,
                 d_ff: int = 2048,
                 n_encoder_layer: int = 6,
                 n_decoder_layer: int = 6,
                 dropout: float = 0.1,
                 max_len: int = 5000,
                 pad_idx: int = 0,
                 ):
        super(Transformer, self).__init__()
        self.pad_idx = pad_idx
        # 编码器部分
        self.encoder = Encoder(src_vocab_size, d_model, n_head, d_ff, n_encoder_layer, dropout, max_len)
        # 解码器部分
        self.decoder = Decoder(trg_vocab_size, d_model, n_head, d_ff, n_decoder_layer, dropout, max_len)

    def forward(self, src: torch.Tensor, trg: torch.Tensor):
        """
        Transformer 模型的前向传播。

        Args:
            src (torch.Tensor): 源序列的整数索引张量，形状为 [batch_size, seq_len_src]。
            trg (torch.Tensor): 目标序列的整数索引张量，形状为 [batch_size, seq_len_trg]。
                                (在训练时，通常是 <sos> token + 部分或全部目标序列；
                                 在推理时，通常是 <sos> token + 已生成的序列)。

        Returns:
            torch.Tensor: 解码器最终输出的 logits，形状为 [batch_size, seq_len_trg, trg_vocab_size]。
                          这些 logits 可以直接传递给交叉熵损失函数。
        """
        # 1.生成掩码
        src_mask = get_encoder_self_attention_mask(src, self.pad_idx)
        trg_mask = get_decoder_self_attention_mask(trg, self.pad_idx)
        dec_enc_mask = get_decoder_cross_attention_mask(src, trg, self.pad_idx)# 注意：这里和 src_mask 是同一个掩码，因为都是针对源序列的填充。名字不同只是为了语义清晰，实际传递的是同一个张量。
        # 2.编码器操作
        enc_output, _ = self.encoder(src, src_mask)
        # 3.解码器操作
        dec_output_logits, _, _ = self.decoder(trg, enc_output, trg_mask, dec_enc_mask)

        return dec_output_logits


