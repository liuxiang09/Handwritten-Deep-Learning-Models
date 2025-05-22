import torch

# 模型超参数
d_model = 512
max_len = 128
dropout = 0.05
n_head = 8
n_layer = 6
d_ff = 2048


# 训练超参数
batch_size = 64
learning_rate = 0.0005
num_epochs = 10


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