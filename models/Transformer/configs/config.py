import torch

# 模型超参数
d_model = 256
max_len = 128
dropout = 0.1
n_head = 4
n_layer = 3
d_ff = 1024


# 训练超参数
batch_size = 64
learning_rate = 5e-4
num_epochs = 50


# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 计算最长标签的宽度，用于对齐输出
labels = [
    "input:",
    "token_embedding:",
    "pos_embedding:",
    "mha_output:",
    "attention_weights:",
    "all_self_attn_weights:",
]
print_width = max(len(label) for label in labels) + 2