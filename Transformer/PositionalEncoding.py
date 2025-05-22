import torch
import torch.nn as nn
import math
from config import *

class TokenEmbedding(nn.Module):
    """
    词嵌入类
    """
    def __init__(self, vocab_size: int, d_model: int):
        """
        初始化词嵌入
        Args:
            vocab_size (int): 词汇表大小
            d_model (int): 嵌入维度
        """
        super(TokenEmbedding, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)  # 维护一个大小为 [vocab_size, d_model] 的词嵌入矩阵

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        词嵌入前向传播
        Args:
            x (torch.Tensor): 输入张量，形状为 [seq_len, batch_size]
        Returns:
            torch.Tensor: 嵌入后的张量，形状为 [seq_len, batch_size, d_model]
        """
        emb_x = self.embedding(x.long()) * math.sqrt(self.d_model)  # 词嵌入
        print(f"{'token_embedding:':<{print_width}}{emb_x.shape}") # 词嵌入形状
        return emb_x

class PositionalEncoding(nn.Module):
    """
    位置编码类
    """
    def __init__(self, d_model: int, dropout: float, max_len: int) -> torch.Tensor:
        """
        初始化位置编码
        Args:
            d_model (int): 嵌入维度
            dropout (float): Dropout概率
            max_len (int): 最大序列长度
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # [max_len, 1],为了方便后续的广播运算
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len, 1, d_model]
        self.register_buffer('pe', pe)  # 注册为缓冲区，不会被视为模型参数

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            x (torch.Tensor): 输入张量，形状为 [seq_len, batch_size, d_model]
        Returns:
            torch.Tensor: 经过位置编码的张量
        """
        x = x + self.pe[:x.size(0), :]
        print(f"{'pos_embedding:':<{print_width}}{x.shape}") # 位置编码形状
        return self.dropout(x)
