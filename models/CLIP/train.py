import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer, CLIPTextModel, CLIPVisionModel
from PIL import Image
import os
import random
from tqdm import tqdm
import argparse
from model.clip import CLIPContrastiveModel, ContrastiveLoss
from utils.dataset import CustomCLIPDataset



# 训练函数
def train(args):
    """
    训练 CLIP 对比学习模型。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 准备数据和处理器
    processor = CLIPProcessor.from_pretrained(args.model_name)
    # 不需要单独的 tokenizer 来获取 max_seq_length，processor 已经包含了

    # 创建颜色块数据集
    # 为了演示，我们在这里创建一个简单的数据集
    # os.makedirs(args.image_dir, exist_ok=True)
    # with open(args.text_data_path, 'w', encoding='utf-8') as f:
    #     for i in range(100): # 示例100张图片
    #         img_name = f"image_{i}.jpg"
    #         caption = f"红色" # 示例caption
    #         f.write(f"{img_name} {caption}\n")
    #         # 创建一个空白图片文件，模拟真实图片
    #         Image.new('RGB', (224, 224), color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))).save(os.path.join(args.image_dir, img_name))

    dataset = CustomCLIPDataset(
        image_dir=args.image_dir,
        text_data_path=args.text_data_path,
        processor=processor,
        max_seq_length=args.max_seq_length
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True # 锁页内存，提高数据传输效率
    )

    # 2. 初始化模型、损失函数和优化器
    model = CLIPContrastiveModel(args.model_name, args.projection_dim).to(device)
    criterion = ContrastiveLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # 3. 训练循环
    print("Starting training...")
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
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

            if (step + 1) % args.log_steps == 0:
                pbar.set_postfix({'loss': f'{total_loss / (step + 1):.4f}'})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}")

    # 4. 保存模型
    os.makedirs(os.path.dirname(args.save_model_path), exist_ok=True)
    torch.save(model.state_dict(), args.save_model_path)
    print(f"Model saved to {args.save_model_path}")

# 6. 主函数
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CLIP Contrastive Model")

    # 添加命令行参数，对应 TrainingConfig 中的参数
    parser.add_argument("--model_name", type=str, default="openai/clip-vit-base-patch32", help="CLIP model name")
    parser.add_argument("--image_dir", type=str, default="./data/clip_images", help="Directory containing image data")
    parser.add_argument("--text_data_path", type=str, default="./data/clip_captions.txt", help="Path to text data file")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--temperature", type=float, default=0.07, help="Temperature parameter for contrastive loss")
    parser.add_argument("--projection_dim", type=int, default=512, help="Projection layer dimension")
    parser.add_argument("--max_seq_length", type=int, default=77, help="Maximum text sequence length")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers")
    parser.add_argument("--save_model_path", type=str, default="./model/CLIP/ckpt/clip_contrastive_model.pth", help="Path to save the trained model")
    parser.add_argument("--log_steps", type=int, default=100, help="Log every n steps")

    args = parser.parse_args()

    # 运行训练函数，将解析后的参数传递进去
    train(args)