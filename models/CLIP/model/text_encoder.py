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

    def forward(self, text: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播函数，接受一个输入张量和注意力掩码，并返回文本编码器的输出。
        
        Args:
            text (torch.Tensor): 输入文本的token ids，形状为 [N, L]
            attention_mask (torch.Tensor, optional): 注意力掩码，形状为 [N, L]，
                                                   1表示需要注意的token，0表示padding token
        
        Returns:
            torch.Tensor: 文本特征向量，形状为 [N, D]
        """
        # text: [N, L]
        # 检查序列长度
        if text.size(1) > self.max_length:
            text = text[:, :self.max_length]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :self.max_length]
        
        # 词嵌入
        x = self.token_embedding(text) # [N, L, D]
        x = x + self.positional_embedding[:text.size(1)] # [N, L, D] + [L, D] -> [N, L, D] 广播机制

        
        # 如果提供了attention_mask，需要将其转换为PyTorch transformer期望的格式
        if attention_mask is not None:
            attention_mask = attention_mask.bool()
            attention_mask = ~attention_mask  # 反转掩码，False表示参与计算

        # 编码器（带掩码）
        x = self.transformer_encoder(x, src_key_padding_mask=attention_mask) # [N, L, D]

        # 取 [EOS] token 的输出作为文本特征
        # 注意：attention_mask已经被反转，False表示有效token
        if attention_mask is not None:
            # (~attention_mask)中1表示有效token，找到最后一个1的位置
            eot_token_pos = (~attention_mask).sum(dim=1) - 1  # [N]
        else:
            # 如果没有掩码，就找最后一个位置
            eot_token_pos = torch.full((text.shape[0],), text.shape[1]-1, device=text.device)  # [N]
            
        x = x[torch.arange(x.shape[0]), eot_token_pos] # [N, D]
        x = self.ln_final(x) # [N, D]
        return x