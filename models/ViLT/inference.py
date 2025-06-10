import requests
from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image
from datetime import datetime

# 记录脚本开始时间
print(f"{datetime.now()} - 脚本开始")

# 准备图像和文本数据
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
test = requests.get(url, stream=True)
image = Image.open(requests.get(url, stream=True).raw)
text = "what's animal in there?"

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa") 

# 记录模型加载完成时间
print(f"{datetime.now()} - 模型和处理器加载完成")

# 准备输入
encoding = processor(image, text, return_tensors='pt')

# 记录前向传播开始时间
print(f"{datetime.now()} - 前向传播开始")
output = model(**encoding)
logits = output.logits
idx = logits.argmax(-1).item()

# 记录预测结果时间
print(f"{datetime.now()} - 预测结果： {model.config.id2label[idx]}")