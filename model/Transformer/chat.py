import torch
import torch.nn as nn
from transformers import AutoTokenizer # 导入 Hugging Face 分词器
from transformer import Transformer
from config import *

# ==============================================================================
# 0. 准备工作：加载模型、分词器和必要的ID
# ==============================================================================

# 请确保这些变量与你训练时使用的保持一致
# 1. 设备设置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. 加载分词器 (确保路径正确，或直接使用模型名称加载)
# 如果你保存了自定义分词器，请从保存路径加载
TOKENIZER_SAVE_DIR = "./my_custom_tokenizers" # 确保路径与你训练时保存的路径一致

# try:
#     tokenizer_de = AutoTokenizer.from_pretrained(f"{TOKENIZER_SAVE_DIR}/de_tokenizer")
#     tokenizer_en = AutoTokenizer.from_pretrained(f"{TOKENIZER_SAVE_DIR}/en_tokenizer")
#     print("已加载自定义分词器。")
# except Exception:
#     print("未能加载自定义分词器，尝试从预训练模型加载并添加特殊token。")
print("加载自定义分词器")
tokenizer_de = AutoTokenizer.from_pretrained("bert-base-german-cased")
tokenizer_en = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenizer_de.add_special_tokens({'bos_token': '<sos>', 'eos_token': '<eos>'})
tokenizer_en.add_special_tokens({'bos_token': '<sos>', 'eos_token': '<eos>'})

# 3. 获取特殊 Token ID
SRC_SOS_IDX = tokenizer_en.bos_token_id
SRC_EOS_IDX = tokenizer_en.eos_token_id
SRC_PAD_IDX = tokenizer_en.pad_token_id

TRG_SOS_IDX = tokenizer_de.bos_token_id
TRG_EOS_IDX = tokenizer_de.eos_token_id
TRG_PAD_IDX = tokenizer_de.pad_token_id

# 4. 定义模型结构 (已定义)

# 5. 实例化模型并加载权重
# 确保这里的参数与你训练时的 Transformer 模型定义一致
SRC_VOCAB_SIZE = len(tokenizer_en)
TRG_VOCAB_SIZE = len(tokenizer_de)

model = Transformer(
    src_vocab_size=SRC_VOCAB_SIZE,
    trg_vocab_size=TRG_VOCAB_SIZE,
    d_model=d_model,
    n_head=n_head,
    d_ff=d_ff,
    n_encoder_layer=n_layer,
    n_decoder_layer=n_layer,
    dropout=dropout,
    max_len=max_len
).to(DEVICE)

# 加载训练好的模型权重
MODEL_PATH = 'model/Transformer/transformer_test_1.pth' # 确保文件名和路径正确
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print(f"模型权重已从 {MODEL_PATH} 加载成功。")
except FileNotFoundError:
    print(f"错误: 找不到模型权重文件 {MODEL_PATH}。请确保文件存在且路径正确。")
    print("将使用未训练的模型进行推理测试。")
except Exception as e:
    print(f"加载模型权重时发生错误: {e}")
    print("将使用未训练的模型进行推理测试。")

model.eval() # 设置模型为评估模式

print("\n模型和分词器已准备就绪。")

# ==============================================================================
# 1. 翻译函数
# ==============================================================================

def translate_sentence(model, sentence, src_tokenizer, trg_tokenizer, device, max_len=128):
    model.eval() # 确保模型处于评估模式

    # 1. 预处理源句子
    # src_tokenizer.encode 不会自动添加特殊token，所以手动添加 EOS
    tokens = src_tokenizer.encode(sentence, add_special_tokens=False, truncation=True, max_length=max_len - 1)
    src_indexes = tokens + [SRC_EOS_IDX] # 添加 EOS
    src_tensor = torch.tensor(src_indexes, dtype=torch.long).unsqueeze(0).to(device) # [1, src_len]

    # 2. 初始化目标序列 (只包含 <sos>)
    # 解码器从 <sos> 开始预测第一个词
    trg_indexes = [TRG_SOS_IDX]
    
    # 贪婪解码循环
    for i in range(max_len):
        trg_tensor = torch.tensor(trg_indexes, dtype=torch.long).unsqueeze(0).to(device) # [1, current_trg_len]

        # 模型前向传播：获取当前解码步的 logits
        # 假设 model.forward 接收 src 和 trg_input
        with torch.no_grad():
            output = model(src_tensor, trg_tensor) # output 形状: [1, current_trg_len, trg_vocab_size]

        # 获取最后一个时间步的预测结果 (即预测下一个词的 logits)
        pred_token_logits = output[:, -1, :] # [1, trg_vocab_size]
        pred_token_id = pred_token_logits.argmax(1).item() # 获取概率最高的词的ID

        # 将预测的词添加到目标序列
        trg_indexes.append(pred_token_id)

        # 如果预测到 <eos>，则停止生成
        if pred_token_id == TRG_EOS_IDX:
            break
            
    # 3. 后处理输出序列
    # 解码并移除 <sos> 和 <eos>
    # 注意：tokenizer_en.decode 默认会移除特殊 token，但如果你想保留，可以设置 skip_special_tokens=False
    translated_sentence = trg_tokenizer.decode(trg_indexes[1:-1], skip_special_tokens=True)
    return translated_sentence

# ==============================================================================
# 2. 对话式测试循环
# ==============================================================================

print("\n进入对话模式。输入 'exit' 退出。")
while True:
    input_sentence = input("\n请输入英语句子 (或输入 'exit' 退出): ")
    if input_sentence.lower() == 'exit':
        print("退出对话模式。")
        break

    if not input_sentence.strip(): # 避免空输入
        print("输入不能为空，请重新输入。")
        continue


    translated_sentence = translate_sentence(
        model,
        input_sentence,
        tokenizer_en,
        tokenizer_de,
        DEVICE,
        max_len=max_len
    )
    print(f"英语原文: {input_sentence}")
    print(f"德语翻译: {translated_sentence}")