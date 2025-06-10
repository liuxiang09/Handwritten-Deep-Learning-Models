import requests
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, CLIPConfig
from datetime import datetime
import torch
from model.clip import CLIPContrastiveModel

# 记录脚本开始时间
print(f"{datetime.now()} - 脚本开始")

# 准备图像和文本数据
image_url = ".\data\clip_images\image_0.jpg"
# image = Image.open(requests.get(image_url, stream=True).raw)
image = Image.open(image_url)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
texts = ["红色", "绿色", "粉红色", "天蓝色"] # 用于测试的文本列表

# 加载 CLIP 模型和处理器
local_ckpt_path = "./model/CLIP/ckpt/clip_contrastive_model.pth" # 使用原始路径


model = CLIPContrastiveModel('openai/clip-vit-base-patch32', 512)
state_dict = torch.load(local_ckpt_path, map_location=device)
model.load_state_dict(state_dict)
print("成功加载本地模型！")


processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 记录模型加载完成时间
print(f"{datetime.now()} - 模型和处理器加载完成")

# 准备输入
inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)


# 进行前向传播（推理）
# 记录前向传播开始时间
print(f"{datetime.now()} - 前向传播开始")
outputs = model(**inputs)
text_features, vision_features, logit_scale = outputs
# 计算图像-文本相似度
logits_per_image = logit_scale * vision_features @ text_features.T

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