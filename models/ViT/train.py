import logging
from transformers import ViTImageProcessor, ViTForImageClassification, ViTConfig, TrainingArguments, Trainer
import torch
import torchvision
import numpy as np

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


if __name__ == '__main__':
    # 选择一个预训练的 ViT 模型对应的 processor
    # 'google/vit-base-patch16-224-in21k' 是一个常用模型，在 ImageNet-21k 上预训练
    # 'google/vit-base-patch16-224' 是在 ImageNet-1k 上微调过的
    processor_name = 'google/vit-base-patch16-224' # 或者 'google/vit-base-patch16-224-in21k'
    try:
        processor = ViTImageProcessor.from_pretrained(processor_name)
    except Exception as e:
        logging.error(f"Could not load processor {processor_name}. \nError: {e}")
        logging.info("Attempting to load a default ViT processor config if available.")
        # 作为后备，你可以尝试加载一个基础配置，但这可能不包含预训练的均值/标准差
        config = ViTConfig.from_pretrained('google/vit-base-patch16-224-in21k', trust_remote_code=True) # 示例
        processor = ViTImageProcessor(size={"height": config.image_size, "width": config.image_size},
                                    image_mean=config.image_mean if hasattr(config, 'image_mean') else [0.5, 0.5, 0.5], # 默认值
                                    image_std=config.image_std if hasattr(config, 'image_std') else [0.5, 0.5, 0.5])   # 默认值
        logging.info(f"Initialized processor with default ViT config: image_size={processor.size}")

    logging.info(f"Finish building the processor {processor_name}.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # CIFAR-10
    num_classes = 10
    cifar10_labels = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'] 

    # 加载CIFAR-10数据集
    logging.info(f"Start downloading the CIFAR10 dataset.")
    transform = None
    cifar10_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    cifar10_val = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # 按照1/100的比率随机采样训练集和测试集
    import random
    train_indices = random.sample(range(len(cifar10_train)), len(cifar10_train) // 100)
    cifar10_train = torch.utils.data.Subset(cifar10_train, train_indices)
    eval_indices = random.sample(range(len(cifar10_val)), len(cifar10_val) // 100)
    cifar10_val = torch.utils.data.Subset(cifar10_val, eval_indices)
    logging.info(f"Completed downloading the CIFAR10 dataset.")

    # Model initialization
    model_name = 'google/vit-base-patch16-224' # model 和 processor 对应
    # model_name = 'google/vit-base-patch16-224-in21k' # 如果想从 ImageNet-21k 预训练开始

    try:
        model = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True # 关键参数！允许加载预训练权重时替换分类头
                                         # 如果预训练模型的分类头和你的 num_labels 不匹配
        )
    except Exception as e:
        # 加载预训练模型出错，通过ViTConfig构建未训练的模型
        logging.error(f"Error loading pretrained model '{model_name}': {e}")
        logging.info(f"Attempting to initialize a new model from config.")
        config = ViTConfig.from_pretrained(model_name, num_labels=num_classes, trust_remote_code=True)
        model = ViTForImageClassification(config)
    logging.info(f"Finishing initialized a new ViT model.")
    model = model.to(device)

    # 定义 TrainingArguments
    training_args = TrainingArguments(
        output_dir="./model/ViT/output",     # 结果输出目录
        num_train_epochs=10,                 # 训练总轮数
        per_device_train_batch_size=32,      # 每个设备上的训练批次大小
        per_device_eval_batch_size=32,       # 每个设备上的评估批次大小
        learning_rate=5e-5,                  # 学习率
        weight_decay=0.01,                   # 权重衰减
        logging_dir="./model/ViT/logs",      # 日志目录
        logging_steps=100,                   # 每多少步记录一次日志
        eval_strategy="steps",               # 在每个 epoch 结束时进行评估
        eval_steps = 5,                      # 每5个steps就进行一次评估
        save_strategy="steps",               # 在每个 epoch 结束时保存检查点
        load_best_model_at_end=True,         # 训练结束时加载最佳模型
        metric_for_best_model="accuracy",    # 用于选择最佳模型的指标
        report_to="none",                    # 不向任何hub报告，或者可以设置为 "tensorboard"
        remove_unused_columns=False          # 保持 collate_fn 返回的所有列
    )

    # 实例化 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=cifar10_train,
        eval_dataset=cifar10_val,
        data_collator=collate_fn_cifar, # Trainer 会使用这个 collate 函数来批处理数据
        compute_metrics=compute_metrics, # 提供一个函数来计算评估指标
    )

    logging.info("Starting training with Trainer...")
    trainer.train(resume_from_checkpoint=True)
    logging.info("Training complete. Evaluating model...")

    # 在训练结束后评估模型
    eval_results = trainer.evaluate()
    logging.info(f"Evaluation results: {eval_results}")

    # 可以选择保存微调后的模型
    trainer.save_model("./checkpoints/vit-cifar10")
    processor.save_pretrained("./checkpoints/vit-cifar10")
    logging.info("Fine-tuned model saved.")


