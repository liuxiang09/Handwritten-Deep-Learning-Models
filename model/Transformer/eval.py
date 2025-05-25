import torch
import torch.nn as nn
from transformers import AutoTokenizer  # 导入 Hugging Face 分词器
from transformer import Transformer
from config import *
from utils.Multi30kDataset import Multi30kDataset, collate_fn
import os
from torch.utils.data import DataLoader
from functools import partial  # 用于向 collate_fn 传递额外参数
import evaluate  # 导入 Hugging Face 的 evaluate 库
from tqdm import tqdm

print("加载自定义分词器")
tokenizer_de = AutoTokenizer.from_pretrained("bert-base-german-cased")
tokenizer_en = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenizer_de.add_special_tokens({'bos_token': '<sos>', 'eos_token': '<eos>'})
tokenizer_en.add_special_tokens({'bos_token': '<sos>', 'eos_token': '<eos>'})

SRC_SOS_IDX = tokenizer_en.bos_token_id
SRC_EOS_IDX = tokenizer_en.eos_token_id
SRC_PAD_IDX = tokenizer_en.pad_token_id

TRG_SOS_IDX = tokenizer_de.bos_token_id
TRG_EOS_IDX = tokenizer_de.eos_token_id
TRG_PAD_IDX = tokenizer_de.pad_token_id

SRC_VOCAB_SIZE = len(tokenizer_en)
TRG_VOCAB_SIZE = len(tokenizer_de)

model = Transformer(len(tokenizer_en), len(tokenizer_de), d_model, n_head, d_ff, max_len=128, dropout=dropout).to(device)

# 加载训练好的模型权重
MODEL_PATH = 'model/Transformer/transformer_test_3.pth' # 确保文件名和路径正确
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"模型权重已从 {MODEL_PATH} 加载成功。")
except FileNotFoundError:
    print(f"错误: 找不到模型权重文件 {MODEL_PATH}。请确保文件存在且路径正确。")
    print("将使用未训练的模型进行推理测试。")
except Exception as e:
    print(f"加载模型权重时发生错误: {e}")
    print("将使用未训练的模型进行推理测试。")

model.eval() # 设置模型为评估模式

print("\n模型和分词器已准备就绪。")

data_dir = 'model/Transformer/data/multi30k'
train_src_file = os.path.join(data_dir, f'train.en')
train_trg_file = os.path.join(data_dir, f'train.de')
test_src_file = os.path.join(data_dir, f'test_2016_flickr.en')
test_trg_file = os.path.join(data_dir, f'test_2016_flickr.de')

train_dataset = Multi30kDataset(
    src_filepath=train_src_file,
    trg_filepath=train_trg_file,
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

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=partial(collate_fn, src_pad_idx=SRC_PAD_IDX, trg_pad_idx=TRG_PAD_IDX)
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=partial(collate_fn, src_pad_idx=SRC_PAD_IDX, trg_pad_idx=TRG_PAD_IDX)
)

for batch_idx, (src_batch, trg_batch) in enumerate(test_loader):
    print(f"Batch {batch_idx+1} Source Batch Shape: {src_batch.shape}")
    print(f"Batch {batch_idx+1} Target Batch Shape: {trg_batch.shape}")
    print(f"Example Source Sequence (first in batch):\n{src_batch[0]}")
    print(f"Example Target Sequence (first in batch):\n{trg_batch[0]}")
    print(f"Decoded Source (first in batch): {tokenizer_en.decode(src_batch[0], skip_special_tokens=False)}")
    print(f"Decoded Target (first in batch): {tokenizer_de.decode(trg_batch[0], skip_special_tokens=False)}")
    break # 只获取第一个批次进行演示

def translate(model, src_tokens, src_tokenizer, trg_tokenizer, max_len=128, device=device):
    """
    给定一个源句子的token序列，通过模型生成目标语言的翻译
    """
    model.eval()
    src_tokens = src_tokens.unsqueeze(0).to(device) # [1, seq_len]
    trg_tokens = torch.tensor([[TRG_SOS_IDX]], dtype=torch.long, device=device) # shape: [1, 1]

    # 初始化目标序列
    for _ in range(max_len):
        with torch.no_grad():
            output = model(src_tokens, trg_tokens)

        # 预测下一个 token
        pred_token_id = output.argmax(2)[:, -1].item() # 获取最后一个时间步的token
        
        # 将预测的 token 添加到目标序列
        trg_tokens = torch.cat(
            (trg_tokens, torch.tensor([[pred_token_id]], dtype=torch.long, device=device)), dim=1
        )

        # 如果输出<eos>，则停止输出
        if pred_token_id == TRG_EOS_IDX:
            break
    
    # 解码生成的 token ID 序列
    # 对于BLEU，通常在解码后再处理，但我们这里先简单移除SOS/EOS
    decoded_sentence = trg_tokenizer.decode(
        trg_tokens[0],  # 取出批次维度，并只取第一个样本
        skip_special_tokens=True # 移除特殊标记，如 <sos>, <eos>, <pad>
    )
    return decoded_sentence

# --- 模型评估函数 ---
def evaluate_model(model, data_loader, src_tokenizer, trg_tokenizer, device, max_len=50):
    model.eval() # 设置模型为评估模式

    # 初始化 BLEU 评估器
    # 使用 sacrebleu 作为度量标准，因为它更标准化
    metric = evaluate.load("sacrebleu")

    predicted_sentences = []
    reference_sentences = []

    print("\n开始评估模型性能...")
    with torch.no_grad():
        for batch_idx, (src_batch, trg_batch) in enumerate(tqdm(data_loader)):
            # 将批次数据移动到指定设备
            src_batch = src_batch.to(device)
            trg_batch = trg_batch.to(device)

            # 遍历批次中的每个样本进行翻译
            for i in range(src_batch.shape[0]): # 遍历 batch_size
                src_example = src_batch[i]
                trg_example = trg_batch[i]

                # 生成机器翻译
                # translate_sentence 期望单个序列，所以这里不加 unsqueeze(0)
                # translate_sentence 内部会处理 unsqueeze
                translated_text = translate(
                    model,
                    src_example, # 传递单条源序列 (shape: [seq_len])
                    src_tokenizer,
                    trg_tokenizer,
                    max_len,
                    device
                )
                predicted_sentences.append(translated_text)

                # 解码参考译文，用于与机器译文进行比较
                # 移除 <sos>, <eos>, <pad> token
                ref_text = trg_tokenizer.decode(
                    trg_example[trg_example != TRG_PAD_IDX], # 过滤掉填充 token
                    skip_special_tokens=True # 移除 <sos>, <eos>
                )
                reference_sentences.append([ref_text]) # sacrebleu 期望参考译文是列表的列表

            # if (batch_idx + 1) % 10 == 0:
            #     print(f"已处理 {batch_idx + 1} 个批次...")

    print("\n所有批次处理完毕，开始计算 BLEU 分数...")
    # 计算 BLEU 分数
    # metric.compute() 期望 predictions 是一个字符串列表，references 是一个字符串列表的列表
    # 例如：predictions = ["hello world"], references = [["hello world"], ["hi there"]]
    results = metric.compute(predictions=predicted_sentences, references=reference_sentences)

    print("\n--- 评估结果 ---")
    print(f"BLEU 分数: {results['score']:.2f}")
    print(f"n-grams精度: {results['precisions']}") # N-gram 精度
    print(f"机器翻译文本词数: {results['sys_len']}")
    print(f"参考译文词数: {results['ref_len']}")
    print("----------------")

    return results['score']

# --- 执行评估 ---
if __name__ == "__main__":
    # 确保 device 在 config.py 中正确定义 (例如: device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    # 并且其他配置变量 (d_model, n_head, d_ff, n_layer, dropout, max_len, batch_size) 也已定义。

    # 运行评估
    bleu_score = evaluate_model(
        model,
        test_loader,
        tokenizer_en,
        tokenizer_de,
        device,
        max_len # 使用配置中的最大序列长度作为翻译时的最大生成长度
    )