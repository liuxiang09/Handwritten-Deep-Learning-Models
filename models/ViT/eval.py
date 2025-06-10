import logging
import torchvision
import numpy as np
import torch
from transformers import AutoModelForImageClassification, AutoImageProcessor, TrainingArguments, Trainer
import json



logging.basicConfig(level=logging.INFO, format='%(asctime)s == %(levelname)s == %(message)s')

# 定义 collate_fn，在这里使用 processor 处理图像批次
# 注意：processor 变量将在 __main__ 块中初始化，并在函数中作为全局变量使用。
def collate_fn_cifar(batch):
    # batch 是一个列表，每个元素是 (PIL_image, label)
    images = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch])

    # 使用 ViTImageProcessor 处理图像
    # `padding=True` (或 "max_length") 和 `truncation=True` 是好习惯
    # `return_tensors="pt"` 使其返回 PyTorch 张量
    inputs = processor(images=images, return_tensors="pt")
    inputs['labels'] = labels # 将标签添加到字典中，方便后续模型使用
    return inputs

# 定义评估指标计算函数
def compute_metrics(p):
    predictions = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(predictions, axis=1)
    return {"accuracy": (preds == p.label_ids).astype(float).mean()}


# 定义保存模型的路径
model_path = "./model/ViT/fine_tuned_vit_cifar10"
processor_name = "google/vit-base-patch16-224" # 确保与训练时使用的 processor_name 一致

# 加载模型
# 确保您使用的AutoModel类与您训练的模型类型匹配
model = AutoModelForImageClassification.from_pretrained(model_path)
processor = AutoImageProcessor.from_pretrained(processor_name, use_fast=True) # 使用 fast 处理器

logging.info(f"模型已从 {model_path} 加载。")
logging.info(f"处理器已从 {processor_name} 加载。")

# 加载CIFAR-10评估数据集
logging.info(f"Start downloading the CIFAR10 dataset.")
transform = None # 在 collate_fn 中处理图像，所以这里 transform 可以是 None
cifar10_val = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# 按照1/100的比率随机采样测试集 (与训练时保持一致)
# import random
# eval_indices = random.sample(range(len(cifar10_val)), len(cifar10_val) // 100)
# cifar10_val = torch.utils.data.Subset(cifar10_val, eval_indices)
logging.info(f"Completed downloading the CIFAR10 dataset. Eval set size: {len(cifar10_val)}")

eval_output_dir = "./model/ViT/eval_results"
# 定义 TrainingArguments，仅用于评估
eval_args = TrainingArguments(
    output_dir=eval_output_dir, # 评估结果输出目录
    per_device_eval_batch_size=32, # 评估批次大小
    report_to="none", # 不向任何hub报告
    remove_unused_columns=False # 保持 collate_fn 返回的所有列
)

# 实例化 Trainer
trainer = Trainer(
    model=model,
    args=eval_args,
    eval_dataset=cifar10_val,
    data_collator=collate_fn_cifar,
    compute_metrics=compute_metrics,
)

logging.info("Starting evaluation with Trainer...")
eval_results = trainer.evaluate()
with open(eval_output_dir, "w") as f:
    json.dump(eval_results, f, ident=4)
logging.info(f"Evaluation results saved to : {eval_output_dir}")