import requests
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from datetime import datetime
import torch

# 记录脚本开始时间
print(f"{datetime.now()} - 脚本开始")

# 准备图像和文本数据
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

texts = ["a photo of a cat", "a photo of a dog", "a photo of a person"] # 用于测试的文本列表

# 加载 CLIP 模型和处理器
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 记录模型加载完成时间
print(f"{datetime.now()} - 模型和处理器加载完成")

# 准备输入
# 记录输入准备开始时间
print(f"{datetime.now()} - 输入准备开始")
inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)

# 记录输入准备完成时间
print(f"{datetime.now()} - 输入准备完成")

# 进行前向传播（推理）
# 记录前向传播开始时间
print(f"{datetime.now()} - 前向传播开始")
outputs = model(**inputs)

# 计算图像-文本相似度
logits_per_image = outputs.logits_per_image # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1) # we can take softmax to get probabilities

# 记录推理完成时间
print(f"{datetime.now()} - 推理完成")

# 打印结果
print("图像-文本相似度得分 (logits):")
print(logits_per_image)

print("\n图像-文本相似度概率:")
# 将概率与文本对应打印
for text, prob in zip(texts, probs[0]):
    print(f"- \"{text}\": {prob.item():.4f}")

# 记录脚本结束时间
print(f"{datetime.now()} - 脚本结束") 