import torch

def get_encoder_self_attention_mask(src_seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    生成用于 Transformer 编码器自注意力的掩码。
    该掩码同时考虑 Query 和 Key 的填充，形状为 [batch_size, seq_len_src, seq_len_src]。

    Args:
        src_seq (torch.Tensor): 输入源序列的批次，形状为 [batch_size, seq_len_src]。
                                  包含 token 的整数索引。
        pad_idx (int): 用于填充的 token 的整数索引。默认为 0。

    Returns:
        torch.Tensor: 编码器自注意力掩码，形状为 [batch_size, seq_len_src, seq_len_src]。
                      值为 True 表示允许注意力（不遮蔽），False 表示禁止注意力（遮蔽）。
    """
    batch_size, seq_len_src = src_seq.shape
    device = src_seq.device

    # 1. 生成源序列的填充掩码
    # 形状: [batch_size, seq_len_src]
    src_padding_mask = (src_seq != pad_idx).to(device) # [batch_size, seq_len_src]

    # 2. 扩展维度
    encoder_self_attn_mask = src_padding_mask.unsqueeze(1).expand(batch_size, seq_len_src, seq_len_src)
    
    return encoder_self_attn_mask


def get_decoder_cross_attention_mask(src_seq: torch.Tensor, trg_seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    生成用于解码器交叉注意力的掩码，形状为 [batch_size, seq_len_trg, seq_len_src]。
    只遮蔽 src_seq 中的填充。
    """
    batch_size, seq_len_src = src_seq.shape
    _, seq_len_trg = trg_seq.shape # 获取 seq_len_trg
    device = src_seq.device

    # 1. 生成源序列的 1D 填充掩码
    src_padding_mask = (src_seq != pad_idx).to(device) # [batch_size, seq_len_src]

    # 2. 扩展维度以匹配交叉注意力的分数矩阵形状
    # 将 src_padding_mask_1d 从 [batch_size, seq_len_src] 扩展到 [batch_size, 1, seq_len_src]
    # 然后再广播到 [batch_size, seq_len_trg, seq_len_src]
    cross_attn_mask = src_padding_mask.unsqueeze(1).expand(batch_size, seq_len_trg, -1)

    return cross_attn_mask

def get_decoder_self_attention_mask(trg_seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    生成用于 Transformer 解码器自注意力的序列掩码。
    该掩码结合了前瞻掩码（Look-Ahead Mask）和填充掩码（Padding Mask）。
    形状：[batch_size, seq_len_trg, seq_len_trg]
    """
    batch_size, seq_len_trg = trg_seq.shape
    device = trg_seq.device

    # 1. 前瞻掩码
    look_ahead_mask = torch.tril(torch.ones(seq_len_trg, seq_len_trg, dtype=torch.bool, device=device))
    look_ahead_mask = look_ahead_mask.unsqueeze(0).expand(batch_size, -1, -1) 

    # 2. 填充掩码
    padding_mask = (trg_seq != pad_idx).unsqueeze(1).expand(batch_size, seq_len_trg, -1).to(device)

    # 3. 组合
    final_mask = look_ahead_mask & padding_mask
    return final_mask