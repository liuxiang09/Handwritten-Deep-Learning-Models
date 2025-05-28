import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer, CLIPTextModel, CLIPVisionModel
from PIL import Image
import os
import random
from tqdm import tqdm
import math

# 1. 配置类
class TrainingConfig:
    """
    训练配置类，管理所有可配置的参数。
    """
    def __init__(self):
        self.model_name = "openai/clip-vit-base-patch32"  # 可以选择其他 CLIP 模型，例如 "openai/clip-vit-large-patch14"
        self.image_dir = "./data/images"  # 假设图片数据在此目录下
        self.text_data_path = "./data/captions.txt"  # 假设文本数据在此文件中
        self.batch_size = 32
        self.num_epochs = 5
        self.learning_rate = 2e-5
        self.temperature = 0.07  # 对比学习中的温度参数
        self.projection_dim = 512 # 投影层的维度，与CLIP的嵌入维度一致
        self.max_seq_length = 77 # CLIP默认的文本最大序列长度
        self.num_workers = 4 # DataLoader的工作进程数
        self.save_model_path = "./model/CLIP/ckpt/clip_contrastive_model.pth"
        self.log_steps = 100 # 每隔多少步打印一次日志

# 2. 数据集类
class CustomCLIPDataset(Dataset):
    """
    自定义 CLIP 数据集。
    假设我们有一个图片文件夹，和一个文本文件，每行对应一个图片文件名和其描述，用制表符分隔。
    例如：
    image1.jpg\t一张狗的图片
    image2.png\t一只猫在睡觉
    """
    def __init__(self, image_dir, text_data_path, processor, max_seq_length):
        self.image_dir = image_dir
        self.processor = processor
        self.max_seq_length = max_seq_length
        self.data = self._load_data(text_data_path)

    def _load_data(self, text_data_path):
        data = []
        with open(text_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    image_filename, text_caption = parts
                    image_path = os.path.join(self.image_dir, image_filename)
                    if os.path.exists(image_path): # 检查图片是否存在
                        data.append({'image_path': image_path, 'text_caption': text_caption})
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item['image_path']
        text_caption = item['text_caption']

        # 加载图片
        image = Image.open(image_path).convert("RGB")

        # 处理图片和文本
        # processor 会处理图像大小调整、归一化等，并对文本进行分词和编码
        inputs = self.processor(
            text=text_caption,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_length
        )
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'pixel_values': inputs['pixel_values'].squeeze(),
            'image': image # 原始图像，方便在collate_fn中处理
        }


# 3. CLIP 对比学习模型
class CLIPContrastiveModel(nn.Module):
    """
    CLIP 对比学习模型，包含图像编码器、文本编码器和投影层。
    """
    def __init__(self, model_name, projection_dim):
        super().__init__()
        # 加载预训练的 CLIP 文本编码器和视觉编码器
        clip_model = CLIPModel.from_pretrained(model_name)
        self.text_encoder = clip_model.text_model
        self.vision_encoder = clip_model.vision_model

        # 投影层，将文本和图像特征投影到同一维度空间
        self.text_projection = nn.Linear(self.text_encoder.config.hidden_size, projection_dim)
        self.vision_projection = nn.Linear(self.vision_encoder.config.hidden_size, projection_dim)

        # 归一化层
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07)) # 初始化为 1/0.07，对应 CLIP 论文中的初始化

    def forward(self, input_ids, attention_mask, pixel_values):
        # 获取文本特征
        text_features = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        text_features = self.text_projection(text_features)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True) # 归一化

        # 获取图像特征
        vision_features = self.vision_encoder(pixel_values=pixel_values).pooler_output
        vision_features = self.vision_projection(vision_features)
        vision_features = vision_features / vision_features.norm(dim=-1, keepdim=True) # 归一化

        return text_features, vision_features, self.logit_scale.exp()


# 4. 对比损失函数
class ContrastiveLoss(nn.Module):
    """
    CLIP 对比损失函数。
    计算图像和文本嵌入之间的对称交叉熵损失。
    """
    def __init__(self):
        super().__init__()

    def forward(self, image_features, text_features, logit_scale):
        # 计算相似度矩阵
        # (batch_size, projection_dim) x (projection_dim, batch_size) -> (batch_size, batch_size)
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logit_scale * text_features @ image_features.T

        # 构建目标标签
        # 对应位置是正样本，其他位置是负样本
        labels = torch.arange(len(image_features), device=image_features.device)

        # 计算交叉熵损失
        loss_i = nn.functional.cross_entropy(logits_per_image, labels)
        loss_t = nn.functional.cross_entropy(logits_per_text, labels)

        # 对称损失
        total_loss = (loss_i + loss_t) / 2
        return total_loss


# 5. 训练函数
def train(config: TrainingConfig):
    """
    训练 CLIP 对比学习模型。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 准备数据和处理器
    processor = CLIPProcessor.from_pretrained(config.model_name)
    tokenizer = CLIPTokenizer.from_pretrained(config.model_name) # 用于获取max_seq_length

    # 创建虚拟数据（实际使用时请替换为你的真实数据）
    # 为了演示，我们在这里创建一个简单的数据集
    os.makedirs(config.image_dir, exist_ok=True)
    with open(config.text_data_path, 'w', encoding='utf-8') as f:
        for i in range(100): # 示例100张图片
            img_name = f"image_{i}.jpg"
            caption = f"这是一张关于物体 {i % 10} 的图片。"
            f.write(f"{img_name}\t{caption}\n")
            # 创建一个空白图片文件，模拟真实图片
            Image.new('RGB', (224, 224), color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))).save(os.path.join(config.image_dir, img_name))

    dataset = CustomCLIPDataset(
        image_dir=config.image_dir,
        text_data_path=config.text_data_path,
        processor=processor,
        max_seq_length=config.max_seq_length
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True # 锁页内存，提高数据传输效率
    )

    # 2. 初始化模型、损失函数和优化器
    model = CLIPContrastiveModel(config.model_name, config.projection_dim).to(device)
    criterion = ContrastiveLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    # 3. 训练循环
    print("Starting training...")
    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        for step, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)

            optimizer.zero_grad()

            text_features, vision_features, logit_scale = model(input_ids, attention_mask, pixel_values)

            loss = criterion(vision_features, text_features, logit_scale)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (step + 1) % config.log_steps == 0:
                pbar.set_postfix({'loss': f'{total_loss / (step + 1):.4f}'})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}")

    # 4. 保存模型
    os.makedirs(os.path.dirname(config.save_model_path), exist_ok=True)
    torch.save(model.state_dict(), config.save_model_path)
    print(f"Model saved to {config.save_model_path}")

# 6. 主函数
if __name__ == "__main__":
    config = TrainingConfig()
    train(config)