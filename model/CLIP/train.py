from transformers import CLIPConfig, CLIPModel
import torch


# 加载CLIP模型的配置
config = CLIPConfig.from_pretrained("openai/clip-vit-base-patch32")

# 使用配置实例化一个未训练的模型
# 模型具有随机初始化的权重
model = CLIPModel(config)

input()