import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, max_length: int, n_head: int, n_layer: int, dropout: float = 0.1):
        super().__init__()
        """
        TextEncoder 类定义了 CLIP 模型的文本编码器部分。
        
        Args:
            vocab_size (int): 词汇表大小
            embed_dim (int): 词嵌入维度
            max_length (int): 最大序列长度
            n_head (int): 多头注意力机制的头数
            n_layer (int): 编码器层数
            dropout (float): 丢弃率
        """
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_embedding = nn.Parameter(torch.empty(max_length, embed_dim)) # [max_L, D]
        nn.init.normal_(self.positional_embedding, mean=0.0, std=0.01) # normal_代表原位操作

        # 使用PyTorch内置的TransformerEncoderLayer构建编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_head,
            dim_feedforward=4 * embed_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True, # 输入的维度顺序为 (N, L, D)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_layer,
        )
        self.ln_final = nn.LayerNorm(embed_dim)
        self.max_length = max_length

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数，接受一个输入张量 x，并返回文本编码器的输出。
        
        """
        # text: [N, L]
        # 检查序列长度
        if text.size(1) > self.max_length:
            text = text[:, :self.max_length]
        
        # 词嵌入
        x = self.token_embedding(text) # [N, L, D]
        x = x + self.positional_embedding[:text.size(1)] # [N, L, D] + [L, D] -> [N, L, D] 广播机制

        # 编码器
        x = self.transformer_encoder(x) # [N, L, D]

        # 取 [EOS] token 的输出作为文本特征, [EOS] token 是每个序列的最后一个有效 token。
        # eot_token_pos shape: (N,)
        eot_token_pos = text.argmax(dim=-1) # 找到每行中ID最大的值的位置，一般来说词汇表中[EOS] token 的ID最大
        x = x[torch.arange(x.shape[0]), eot_token_pos] # [N, D]

        x = self.ln_final(x) # [N, D]
        return x