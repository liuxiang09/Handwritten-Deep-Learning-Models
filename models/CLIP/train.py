import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPProcessor
from PIL import Image
import os
import random
from tqdm import tqdm
import argparse
import sys

# 导入你的自定义模型
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.clip import CLIP
from model.vision_encoder import VisionEncoder
from model.text_encoder import TextEncoder
from utils.dataset import CustomCLIPDataset

# 对比损失函数
class ContrastiveLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image_features, text_features, logit_scale):
        # 归一化特征 (已经在 CLIP 模型的 forward 中完成)
        
        # 计算相似度矩阵
        logits = (image_features @ text_features.T) * logit_scale.exp()

        # 创建标签 (对角线为正样本)
        labels = torch.arange(len(logits)).to(logits.device)

        # 计算图像到文本的损失 (行是图像，列是文本)
        loss_i = F.cross_entropy(logits, labels)
        
        # 计算文本到图像的损失 (转置 logits，行是文本，列是图像)
        loss_t = F.cross_entropy(logits.T, labels)
        
        # 返回平均损失
        return (loss_i + loss_t) / 2


# 训练函数
def train(args):
    """
    训练自定义 CLIP 对比学习模型。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 准备数据和处理器
    # CLIPProcessor 仍然用于数据预处理，你可以根据需要替换为自定义的 Transforms
    processor = CLIPProcessor.from_pretrained(args.model_name)

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

    # 2. 初始化自定义模型、损失函数和优化器
    # 初始化 VisionEncoder
    image_encoder = VisionEncoder(
        image_size=args.image_size,
        patch_size=args.patch_size,
        embed_dim=args.vision_feature_dim, # VisionEncoder 输出维度
        n_head=args.image_n_head,
        n_layer=args.image_n_layer
    ).to(device)

    # 初始化 TextEncoder
    text_encoder = TextEncoder(
        vocab_size=args.vocab_size,
        embed_dim=args.text_feature_dim, # TextEncoder 输出维度
        max_length=args.max_seq_length,
        n_head=args.text_n_head,
        n_layer=args.text_n_layer
    ).to(device)

    # 初始化自定义 CLIP 模型
    model = CLIP(
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        vision_feature_dim=args.vision_feature_dim,
        text_feature_dim=args.text_feature_dim,
        embed_dim=args.projection_dim # CLIP 的共享嵌入维度
    ).to(device)

    criterion = ContrastiveLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # 3. 训练循环
    print("Starting training...")
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for step, batch in enumerate(pbar):
            # CustomCLIPDataset 返回的 batch 结构
            input_ids = batch['input_ids'].to(device)         # 文本 token IDs
            # attention_mask = batch['attention_mask'].to(device) # 文本注意力掩码 (TextEncoder 不使用)
            pixel_values = batch['pixel_values'].to(device)   # 图像像素值

            optimizer.zero_grad()

            # CLIP 模型的 forward 方法现在返回 image_features, text_features
            # logit_scale 是 CLIP 模型的一个 nn.Parameter
            image_features, text_features = model(pixel_values, input_ids)
            logit_scale = model.logit_scale

            loss = criterion(image_features, text_features, logit_scale)
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
    parser = argparse.ArgumentParser(description="Train Custom CLIP Contrastive Model")

    # 添加命令行参数，对应 TrainingConfig 中的参数
    parser.add_argument("--model_name", type=str, default="openai/clip-vit-base-patch32", help="CLIP model name (for processor)")
    parser.add_argument("--image_dir", type=str, default="./data/clip_images", help="Directory containing image data")
    parser.add_argument("--text_data_path", type=str, default="./data/clip_captions.txt", help="Path to text data file")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    # parser.add_argument("--temperature", type=float, default=0.07, help="Temperature parameter for contrastive loss") # logit_scale 已经包含
    parser.add_argument("--projection_dim", type=int, default=512, help="Projection layer dimension (shared embed_dim)")
    parser.add_argument("--max_seq_length", type=int, default=77, help="Maximum text sequence length")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers")
    parser.add_argument("--save_model_path", type=str, default="./model/CLIP/ckpt/custom_clip_model.pth", help="Path to save the trained model")
    parser.add_argument("--log_steps", type=int, default=100, help="Log every n steps")

    # 添加自定义编码器参数
    parser.add_argument("--image_size", type=int, default=224, help="Input image size for VisionEncoder")
    parser.add_argument("--patch_size", type=int, default=16, help="Patch size for VisionEncoder")
    parser.add_argument("--vision_feature_dim", type=int, default=768, help="Output feature dimension of VisionEncoder")
    parser.add_argument("--image_n_head", type=int, default=8, help="Number of attention heads for VisionEncoder")
    parser.add_argument("--image_n_layer", type=int, default=6, help="Number of transformer layers for VisionEncoder")

    parser.add_argument("--vocab_size", type=int, default=49408, help="Vocabulary size for TextEncoder")
    parser.add_argument("--text_feature_dim", type=int, default=512, help="Output feature dimension of TextEncoder")
    parser.add_argument("--text_n_head", type=int, default=8, help="Number of attention heads for TextEncoder")
    parser.add_argument("--text_n_layer", type=int, default=6, help="Number of transformer layers for TextEncoder")

    args = parser.parse_args()

    # 运行训练函数，将解析后的参数传递进去
    train(args)