from transformers import AutoTokenizer


text = "Hello, world! This is an example sentence."
tokenizer_en = AutoTokenizer.from_pretrained("bert-base-uncased")

# 1. 基本编码 (返回字典)
encoded_input = tokenizer_en(text)
print(encoded_input)
# 输出示例：
# {'input_ids': [101, 7592, 1010, 2088, 100, 2003, 2003, 2058, 2003, 102],
#  'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}

# 2. 获取 token ID
input_ids = encoded_input['input_ids']
print("Input IDs:", input_ids) # [101, 7592, 1010, 2088, 100, 2003, 2003, 2058, 2003, 102]

# 3. 将 ID 转换回 token 字符串
tokens = tokenizer_en.convert_ids_to_tokens(input_ids)
print("Tokens:", tokens) # ['[CLS]', 'hello', ',', 'world', '!', 'this', 'is', 'an', 'example', 'sentence', '.', '[SEP]']

# 4. 将 token ID 转换回原始字符串 (解码) 
# 可以使用 skip_special_tokens 参数，来得到更干净的文本
decoded_text = tokenizer_en.decode(input_ids, skip_special_tokens=True)
print("Decoded text:", decoded_text) # [CLS] hello, world! this is an example sentence. [SEP]

# 5. 批处理 (处理多句话)
texts = ["First sentence.", "Second sentence here."]
batch_encoded = tokenizer_en(texts, padding=True, truncation=True, return_tensors="pt")
print("Batch Encoded Input IDs:\n", batch_encoded['input_ids'])
print("Batch Attention Mask:\n", batch_encoded['attention_mask'])
# return_tensors="pt" 会返回 PyTorch tensors
# return_tensors="tf" 会返回 TensorFlow tensors
# return_tensors="np" 会返回 NumPy arrays

# 6.特殊标记属性
print(tokenizer_en.pad_token) # [PAD]
print(tokenizer_en.pad_token_id)# 0

print(tokenizer_en.unk_token)# [UNK]
print(tokenizer_en.unk_token_id)# 100

print(tokenizer_en.cls_token)# [CLS]
print(tokenizer_en.cls_token_id)# 101

print(tokenizer_en.sep_token)# [SEP]
print(tokenizer_en.sep_token_id)# 102

print(tokenizer_en.mask_token)# [MASK]
print(tokenizer_en.mask_token_id)# 103

# 7.词汇表信息
print(tokenizer_en.vocab_size) # 30522
# print(tokenizer_en.vocab) # 返回词汇表字典（token -> id）
# print(tokenizer_en.get_vocab())

# 8.添加新token
print(tokenizer_en.all_special_tokens)
print(tokenizer_en.special_tokens_map)
tokenizer_en.add_tokens(['new_token1', 'new_token2'])
tokenizer_en.add_special_tokens({'bos_token': '<sos>', 'eos_token': '<eos>'})
print(tokenizer_en.vocab_size) # 30522
print(len(tokenizer_en)) # 30526
print(tokenizer_en.all_special_tokens)

# 重要！添加新token后，必须调用 model.resize_token_embeddings(len(tokenizer)) 来调整模型Embedding的尺寸，否则新token将无法正确嵌入