import torch
import torch.nn as nn
import math


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        """
        初始化缩放点积注意力
        """
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:
        """
        前向传播
        Args:
            query (torch.Tensor): 查询张量，形状为 [batch_size, n_head, seq_len_q, d_k]
            key (torch.Tensor):   键张量，形状为 [batch_size, n_head, seq_len_k, d_k]
            value (torch.Tensor): 值张量，形状为 [batch_size, n_head, seq_len_v, d_v]
            mask (torch.Tensor, optional):  掩码张量，形状为 [batch_size, 1, seq_len_q, seq_len_k]
        Returns:
            torch.Tensor: 上下文向量，形状为 [batch_size, n_head, seq_len_q, d_v]
            torch.Tensor: 注意力权重，形状为 [batch_size, n_head, seq_len_q, seq_len_k]
        """
        # 1. 计算点积：QK^T
        # q: (batch_size, n_head, seq_len_q, d_k)
        # k: (batch_size, n_head, seq_len_k, d_k)
        # k.transpose(-2, -1) 将 k 的最后两个维度交换，变成 (batch_size, n_head, d_k, seq_len_k)
        # 矩阵乘法结果: (batch_size, n_head, seq_len_q, seq_len_k)
        scores = torch.matmul(query, key.transpose(-2, -1))
        
        # 2. 缩放
        # d_k 是 k 向量的维度（Q 和 K 的隐藏维度）
        d_k = query.size(-1)
        scores = scores / math.sqrt(d_k)

        # 3. 掩码
        if mask is not None:
            # mask 中为 True 的位置会被屏蔽（设置为-inf），这样在 softmax 后对应的权重变为 0
            # mask: (batch_size, n_head, seq_len_q, seq_len_k)
            scores = scores.masked_fill(mask==0, float('-inf'))

        # 4. softmax
        # scores: (batch_size, n_head, seq_len_q, seq_len_k)
        # 在最后一个维度上进行 softmax 操作，得到注意力权重
        attention_weights = torch.softmax(scores, dim=-1)

        # 5. 加权求和：Attention(Q, K, V) = attention_weights * V
        # attention_weights: (batch_size, n_head, seq_len_q, seq_len_k)
        # value: (batch_size, n_head, seq_len_v, d_v),其中seq_len_v = seq_len_k
        # 计算结果: (batch_size, n_head, seq_len_q, d_v)
        output = torch.matmul(attention_weights, value)
        return output, attention_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        """
        初始化多头注意力
        Args:
            d_model (int): 嵌入维度(通常是512)
            n_head (int): 注意力头数(通常是8)
        """
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.d_k = d_model // n_head
        # 定义线性变换层
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        # 定义缩放点积注意力
        self.attention = ScaledDotProductAttention()

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:
        """
        前向传播
        Args:
            query (torch.Tensor): 查询张量，形状为 [batch_size, seq_len_q, d_model]
            key (torch.Tensor):   键张量，形状为 [batch_size, seq_len_k, d_model]
            value (torch.Tensor): 值张量，形状为 [batch_size, seq_len_v, d_model]
            mask (torch.Tensor, optional):  掩码张量，形状为 [batch_size, seq_len_q, seq_len_k]
        Returns:
            torch.Tensor: 多头注意力的输出，形状为 [batch_size, seq_len_q, d_model]
            torch.Tensor: 所有头的平均注意力权重(可选返回)，形状为 [batch_size, n_head, seq_len_q, seq_len_k]
        """
        batch_size = query.size(0)

        # 1. 线性变换
        # q: (batch_size, seq_len_q, d_model)
        # k: (batch_size, seq_len_k, d_model)
        # v: (batch_size, seq_len_v, d_model)
        q = self.w_q(query).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        k = self.w_k(key).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        v = self.w_v(value).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)

        # 2.应用注意力掩码
        if mask is not None:
            # mask: (batch_size, seq_len_q, seq_len_k)
            # 在这里我们需要扩展 mask 的维度以匹配 q 和 k 的形状
            mask = mask.unsqueeze(1)

        # 3.缩放点积注意力
        # output: (batch_size, n_head, seq_len_q, d_k)
        # attention_weights: (batch_size, n_head, seq_len_q, seq_len_k)
        output, attention_weights = self.attention(q, k, v, mask)

        # 4.拼接所有头的输出
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 5.线性变换
        output = self.w_o(output)
        return output, attention_weights