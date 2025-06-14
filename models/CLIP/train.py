import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPProcessor
import os
import random
from tqdm import tqdm
import argparse
import sys
from utils.flickr30k import Flickr30kDataset, collate_fn

# 导入你的自定义模型
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval import evaluate
from model.vision_encoder import VisionEncoder, VisionEncoderPretrained
from model.text_encoder import TextEncoder
from model.modified_resnet import ModifiedResNet
from model.clip import CLIP
from utils.contrastive_loss import MultiTextContrastiveLoss


# 命令行参数解析函数
def parse_args():
    """
    解析命令行参数
    Returns:
        args: 解析后的参数
    """
    parser = argparse.ArgumentParser(description="Train Custom CLIP Contrastive Model")

    # 路径相关参数
    parser.add_argument("--image_dir", type=str, default="./data/flickr30k_images/flickr30k_images")
    parser.add_argument("--text_data_path", type=str, default="./data/flickr30k_images/results.csv")
    parser.add_argument("--log_dir", type=str, default="./models/CLIP/logs")

    # 训练相关参数
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_steps", type=int, default=10)
    parser.add_argument("--eval", action="store_true", help="Run evaluation only")
    parser.add_argument("--train", action="store_true", help="Run training only")
    parser.add_argument("--train_sample_rate", type=float, default=0.7, help="train data sample rate")
    parser.add_argument("--eval_sample_rate", type=float, default=0.3, help="eval data sample rate")

    # 模型相关参数
    parser.add_argument("--model_name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--pretrained_model_name", type=str, default="vit_base_patch16_clip_224.openai")
    parser.add_argument("--projection_dim", type=int, default=512)
    parser.add_argument("--max_seq_length", type=int, default=77)
    parser.add_argument("--image_encoder_type", type=str, default="resnet")
    # parser.add_argument("--temperature", type=float, default=0.07, help="Temperature parameter for contrastive loss") # logit_scale 已经包含

    # 视觉编码器参数
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--vision_feature_dim", type=int, default=768)
    parser.add_argument("--resnet_feature_dim", type=int, default=512)
    parser.add_argument("--image_n_head", type=int, default=8)
    parser.add_argument("--image_n_layer", type=int, default=6)
    
    # 文本编码器参数
    parser.add_argument("--vocab_size", type=int, default=49408)
    parser.add_argument("--text_feature_dim", type=int, default=512)
    parser.add_argument("--text_n_head", type=int, default=8)
    parser.add_argument("--text_n_layer", type=int, default=6)

    return parser.parse_args()


# 训练函数
def train(args, model, dataloader, device):
    """
    训练自定义 CLIP 对比学习模型。
    Args:
        args: 训练参数
        model: 初始化好的CLIP模型
        dataloader: 数据加载器
        device: 训练设备
    """
    criterion = MultiTextContrastiveLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    print("Starting training...")
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for idx, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            optimizer.zero_grad()

            image_features, text_features = model(pixel_values, input_ids, attention_mask)
            logit_scale = model.logit_scale

            loss = criterion(image_features, text_features, logit_scale)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (idx + 1) % args.log_steps == 0:
                pbar.set_postfix({'loss': f'{total_loss / (idx + 1):.4f}'})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}")

    # 保存模型
    save_model_path = f"./models/CLIP/checkpoints/my_clip_{args.image_encoder_type}_epoch_{args.num_epochs}.pth"
    os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
    # 如果文件已存在则删除
    if os.path.exists(save_model_path):
        os.remove(save_model_path)
    torch.save(model.state_dict(), save_model_path)
    print(f"Model saved to {save_model_path}")

    return model


# 6. 主函数
if __name__ == "__main__":
    args = parse_args()

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 准备数据和处理器
    processor = CLIPProcessor.from_pretrained(args.model_name)
    dataset = Flickr30kDataset(
        image_dir=args.image_dir,
        text_path=args.text_data_path,
        processor=processor,
        max_len=args.max_seq_length
    )
    print(dataset[0])
    print(dataset[0]['attention_mask'].shape)
    print(dataset[0]['input_ids'].shape)
    print(dataset[0]['pixel_values'].shape)
    
    # 创建训练集和评估集
    total_size = len(dataset)
    train_size = int(total_size * args.train_sample_rate)
    eval_size = int(total_size * args.eval_sample_rate)
    
    try:
        # 随机采样不重叠的索引
        all_indices = list(range(total_size))
        train_indices = random.sample(all_indices, train_size)
        remaining_indices = list(set(all_indices) - set(train_indices))
        eval_indices = random.sample(remaining_indices, eval_size)
    except ValueError as e:
        print(f"请求的样本数量大于数据集总数量，请调整采样率")
        print(f"ValueError: {e}")
        exit()
    
    # 创建训练集和评估集
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    eval_dataset = torch.utils.data.Subset(dataset, eval_indices)

    print(f"Original dataset length: {total_size}")
    print(f"Training dataset length: {len(train_dataset)}")
    print(f"Evaluation dataset length: {len(eval_dataset)}")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    for i, batch in enumerate(train_dataloader):
        if i == 0:  
            print(batch['input_ids'].shape)
            print(batch['attention_mask'].shape)
            print(batch['pixel_values'].shape)
            # input("Press Enter to continue...")

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    # 初始化模型
    image_encoder_vit = VisionEncoder(
        image_size=args.image_size,
        patch_size=args.patch_size,
        embed_dim=args.vision_feature_dim,
        n_head=args.image_n_head,
        n_layer=args.image_n_layer
    ).to(device)

    image_encoder_vit_pretrained = VisionEncoderPretrained(
        pretrained_model_name=args.pretrained_model_name,
        pretrained=True,
        embed_dim=args.vision_feature_dim
    ).to(device)

    image_encoder_resnet = ModifiedResNet(
        embed_dim=args.resnet_feature_dim,
        n_head=args.image_n_head
    ).to(device)

    text_encoder = TextEncoder(
        vocab_size=args.vocab_size,
        embed_dim=args.text_feature_dim,
        max_length=args.max_seq_length,
        n_head=args.text_n_head,
        n_layer=args.text_n_layer,
    ).to(device)

    # 根据选择的图像编码器类型设置特征维度
    if args.image_encoder_type == 'vit':
        image_encoder = image_encoder_vit
        vision_feature_dim = args.vision_feature_dim
    elif args.image_encoder_type == 'resnet':  # resnet
        image_encoder = image_encoder_resnet
        vision_feature_dim = args.resnet_feature_dim
    elif args.image_encoder_type == 'vit_pretrained':
        image_encoder = image_encoder_vit_pretrained
        vision_feature_dim = args.vision_feature_dim
    else:
        raise ValueError(f"Invalid image encoder type: {args.image_encoder_type}")

    model = CLIP(
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        vision_feature_dim=vision_feature_dim,
        text_feature_dim=args.text_feature_dim,
        embed_dim=args.projection_dim,
    ).to(device)

    # 统计模型参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"\nModel Parameters:")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")
    print(f"Total parameters: {trainable_params + non_trainable_params:,}\n")

    # 如果是评估模式，加载预训练模型
    save_model_path = f"./models/CLIP/checkpoints/my_clip_{args.image_encoder_type}_epoch_{args.num_epochs}.pth"
    if args.train:
        # 训练模式
        train(args, model, train_dataloader, device)
    if args.eval:
        print(f"Loading model from {save_model_path}")
        model.load_state_dict(torch.load(save_model_path))
        evaluate(model, eval_dataloader, device)
    