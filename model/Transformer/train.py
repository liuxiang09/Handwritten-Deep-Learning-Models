import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from collections import Counter
from tqdm import tqdm
import os
import math
import time
from transformer import Transformer 
from functools import partial # 用于向 collate_fn 传递额外参数
from config import *
from utils.Multi30kDataset import Multi30kDataset, collate_fn
from torch.optim import optimizer
# ==============================================================================
# 0. 导入你的 Transformer 模型
# 请确保你的模型文件与此训练脚本在同一目录下，或者在PYTHONPATH中
# 例如：
# from your_transformer_model import Transformer, create_masks
# ==============================================================================

# ==============================================================================
# 1. 配置参数
# ==============================================================================
# 数据路径
data_dir = './model/Transformer/data/multi30k'
src_lang = 'en' # Source language (e.g., English)
trg_lang = 'de' # Target language (e.g., German)

# 文件名
train_src_file = os.path.join(data_dir, f'train.{src_lang}')
train_trg_file = os.path.join(data_dir, f'train.{trg_lang}')
val_src_file = os.path.join(data_dir, f'val.{src_lang}')
val_trg_file = os.path.join(data_dir, f'val.{trg_lang}')
test_src_file = os.path.join(data_dir, f'test_2016_flickr.{src_lang}')
test_trg_file = os.path.join(data_dir, f'test_2016_flickr.{trg_lang}')


# ==============================================================================
# 2. 分词器和词汇表构建
# ==============================================================================
# 加载huggingface bert分词器
print("Loading tokenizer...")
tokenizer_en = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenizer_de = AutoTokenizer.from_pretrained("bert-base-german-cased")
print("tokenizer loaded.")

# 添加开始，结束标记
tokenizer_en.add_special_tokens({'bos_token': '<sos>', 'eos_token': '<eos>'})
tokenizer_de.add_special_tokens({'bos_token': '<sos>', 'eos_token': '<eos>'})
# 特殊的token id
en_sos_idx = tokenizer_en.bos_token_id
en_eos_idx = tokenizer_en.eos_token_id
en_pad_idx = tokenizer_en.pad_token_id
de_sos_idx = tokenizer_de.bos_token_id
de_eos_idx = tokenizer_de.eos_token_id
de_pad_idx = tokenizer_de.pad_token_id
print(f"English tokenizer vocab size after adding: {len(tokenizer_en)}")
print(f"German tokenizer vocab size after adding: {len(tokenizer_de)}")

# ==============================================================================
# 3. 加载数据集
# ==============================================================================
print("\nInstantiating datasets...")
train_dataset = Multi30kDataset(
    src_filepath=train_src_file,
    trg_filepath=train_trg_file,
    src_tokenizer=tokenizer_en,
    trg_tokenizer=tokenizer_de,
    max_seq_len=max_len
)

val_dataset = Multi30kDataset(
    src_filepath=val_src_file,
    trg_filepath=val_trg_file,
    src_tokenizer=tokenizer_en,
    trg_tokenizer=tokenizer_de,
    max_seq_len=max_len
)
test_dataset = Multi30kDataset(
    src_filepath=test_src_file,
    trg_filepath=test_trg_file,
    src_tokenizer=tokenizer_en,
    trg_tokenizer=tokenizer_de,
    max_seq_len=max_len
)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=partial(collate_fn, src_pad_idx=en_pad_idx, trg_pad_idx=de_pad_idx)
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=partial(collate_fn, src_pad_idx=en_pad_idx, trg_pad_idx=de_pad_idx)
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=partial(collate_fn, src_pad_idx=en_pad_idx, trg_pad_idx=de_pad_idx)
)

for batch_idx, (src_batch, trg_batch) in enumerate(train_loader):
    print(f"Batch {batch_idx+1} Source Batch Shape: {src_batch.shape}")
    print(f"Batch {batch_idx+1} Target Batch Shape: {trg_batch.shape}")
    print(f"Example Source Sequence (first in batch):\n{src_batch[0]}")
    print(f"Example Target Sequence (first in batch):\n{trg_batch[0]}")
    print(f"Decoded Source (first in batch): {tokenizer_en.decode(src_batch[0], skip_special_tokens=False)}")
    print(f"Decoded Target (first in batch): {tokenizer_de.decode(trg_batch[0], skip_special_tokens=False)}")
    break # 只获取第一个批次进行演示

# ==============================================================================
# 3. 加载模型
# ==============================================================================
model = Transformer(len(tokenizer_en), len(tokenizer_de), d_model, n_head, d_ff, max_len=128, dropout=dropout).to(device)

model.to(device)
# ==============================================================================
# 4. 训练循环
# ==============================================================================
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss(ignore_index=de_pad_idx) # 忽略填充 token 的损失

for epoch in range(num_epochs):
    model.train()
    print("第{}个epoch".format(epoch + 1))
    running_loss = 0.0
    for batch_idx, (src_batch, trg_batch) in enumerate(tqdm(train_loader)):
        # 清除梯度
        optimizer.zero_grad()
        src, trg = src_batch.to(device), trg_batch.to(device)
        decoder_input = trg[:, :-1] # 解码器输入不要<eos>
        target_labels = trg[:, 1:] # 真值target不要<sos>
        logits = model(src, decoder_input)

        logits_flat = logits.contiguous().view(-1, len(tokenizer_de)) # [batch_size * seq_len_trg, vocab_size]
        target_labels_flat = target_labels.contiguous().view(-1) # [batch_size * seq_len_trg]
        loss = criterion(logits_flat, target_labels_flat)

        loss.backward()
        # 8. 梯度裁剪 (可选，但推荐用于 Transformer 避免梯度爆炸)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # max_norm 是裁剪阈值
        optimizer.step()

        running_loss += loss.item()

    # 打印每个 epoch 的平均损失
    avg_train_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch + 1} completed. Average Training Loss: {avg_train_loss:.4f}")

# 保存模型
save_path = "model/Transformer/transformer_test_3.pth"
torch.save(model.state_dict(), save_path)
print(f"模型已保存到 {save_path}")

