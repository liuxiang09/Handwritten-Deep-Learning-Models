d_model = 512
max_len = 5000
dropout = 0.1
n_head = 8
n_layer = 6
src_vocab_size = 1000
trg_vocab_szie = 2000
d_ff = 2048


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