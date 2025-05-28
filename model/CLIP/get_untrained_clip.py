from transformers import CLIPConfig, CLIPModel
from datetime import datetime
import torch

# 记录脚本开始时间
print(f"{datetime.now()} - 脚本开始")

# 1. 加载 CLIP 模型的配置
# 可以加载一个现有模型的配置来获取正确的架构参数，但不会加载权重。
# 例如，加载 "openai/clip-vit-base-patch32" 的配置
print(f"{datetime.now()} - 加载模型配置")
config = CLIPConfig.from_pretrained("openai/clip-vit-base-patch32")

# 2. 使用配置实例化一个未训练的模型
# 这个模型将具有随机初始化的权重
print(f"{datetime.now()} - 实例化未训练的模型")
model = CLIPModel(config)

print(f"{datetime.now()} - 未训练的 CLIP 模型已创建")

# 现在你可以检查模型的结构，或者准备数据进行训练
# 例如，打印模型的部分结构
# print(model)

# 记录脚本结束时间
print(f"{datetime.now()} - 脚本结束") 