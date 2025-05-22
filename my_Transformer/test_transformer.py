import torch
from config import *
from Transformer import transformer



# 输入矩阵
input = torch.randint(0, 1000, (64, 100))  # [batch_size, seq_len]
print(f"{'input:':<{print_width}}{input.shape}")  # 输入形状

# 目标输出矩阵
output = torch.randint(0, 1000, (64, 200)) # [batch_size, seq_len_trg]
print(f"{'output:':<{print_width}}{output.shape}")  # 目标输出形状

transformer = transformer(src_vocab_size, trg_vocab_size)
logits = transformer(input, output)
print(f"{'logits:':<{print_width}}{logits.shape}")
