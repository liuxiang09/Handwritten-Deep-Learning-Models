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
from utils.flickr30k import Flickr30kDataset, collate_fn

# 导入你的自定义模型
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.clip import CLIP
from model.vision_encoder import VisionEncoder
from model.text_encoder import TextEncoder
from model.modified_resnet import ModifiedResNet

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
def train(args, model, dataloader, device):
    """
    训练自定义 CLIP 对比学习模型。
    Args:
        args: 训练参数
        model: 初始化好的CLIP模型
        dataloader: 数据加载器
        device: 训练设备
    """
    criterion = ContrastiveLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    print("Starting training...")
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for idx, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(device)
            pixel_values = batch['pixel_values'].to(device)

            optimizer.zero_grad()

            image_features, text_features = model(pixel_values, input_ids)
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

def evaluate(args, model, dataloader, criterion, device):
    """
    评估 CLIP 模型的性能
    """
    model.eval()
    total_loss = 0
    total_samples = 0
    correct_image_text = 0
    correct_text_image = 0

    print("Starting evaluation...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            batch_size = input_ids.size(0)

            # 获取特征
            image_features, text_features = model(pixel_values, input_ids)
            logit_scale = model.logit_scale

            # 计算相似度矩阵
            logits = (image_features @ text_features.T) * logit_scale.exp()
            
            # 计算损失
            labels = torch.arange(len(logits)).to(logits.device)
            loss_i = F.cross_entropy(logits, labels)
            loss_t = F.cross_entropy(logits.T, labels)
            loss = (loss_i + loss_t) / 2
            
            # 计算准确率
            image_pred = logits.argmax(dim=1)
            text_pred = logits.T.argmax(dim=1)
            correct_image_text += (image_pred == labels).sum().item()
            correct_text_image += (text_pred == labels).sum().item()
            
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    avg_loss = total_loss / total_samples
    image_text_accuracy = 100 * correct_image_text / total_samples
    text_image_accuracy = 100 * correct_text_image / total_samples
    
    print(f"Evaluation Results:")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Image->Text Accuracy: {image_text_accuracy:.2f}%")
    print(f"Text->Image Accuracy: {text_image_accuracy:.2f}%")
    print(f"Average Accuracy: {(image_text_accuracy + text_image_accuracy) / 2:.2f}%")

    return avg_loss

# 6. 主函数
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Custom CLIP Contrastive Model")

    # 路径相关参数
    parser.add_argument("--image_dir", type=str, default="./data/flickr30k_images/flickr30k_images")
    parser.add_argument("--text_data_path", type=str, default="./data/flickr30k_images/results.csv")
    # parser.add_argument("--save_model_path", type=str, default="./models/CLIP/checkpoints/my_clip_resnet_epoch_1.pth")

    # 训练相关参数
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_steps", type=int, default=10)
    parser.add_argument("--eval", action="store_true", help="Run evaluation only")
    parser.add_argument("--train", action="store_true", help="Run training only") 

    # 模型相关参数
    parser.add_argument("--model_name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--projection_dim", type=int, default=512)
    parser.add_argument("--max_seq_length", type=int, default=77)
    parser.add_argument("--image_encoder_type", type=str, default="vit")
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

    args = parser.parse_args()

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

    # 创建训练集和评估集
    total_size = len(dataset)
    train_size = total_size // 5  # 训练集取1/5
    eval_size = total_size // 10  # 评估集取1/10
    
    # 随机采样不重叠的索引
    all_indices = list(range(total_size))
    train_indices = random.sample(all_indices, train_size)
    remaining_indices = list(set(all_indices) - set(train_indices))
    eval_indices = random.sample(remaining_indices, eval_size)
    
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

    image_encoder_resnet = ModifiedResNet(
        embed_dim=args.resnet_feature_dim,
        n_head=args.image_n_head
    ).to(device)

    text_encoder = TextEncoder(
        vocab_size=args.vocab_size,
        embed_dim=args.text_feature_dim,
        max_length=args.max_seq_length,
        n_head=args.text_n_head,
        n_layer=args.text_n_layer
    ).to(device)

    # 根据选择的图像编码器类型设置特征维度
    if args.image_encoder_type == 'vit':
        image_encoder = image_encoder_vit
        vision_feature_dim = args.vision_feature_dim
    elif args.image_encoder_type == 'resnet':  # resnet
        image_encoder = image_encoder_resnet
        vision_feature_dim = args.resnet_feature_dim
    else:
        raise ValueError(f"Invalid image encoder type: {args.image_encoder_type}")

    model = CLIP(
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        vision_feature_dim=vision_feature_dim,
        text_feature_dim=args.text_feature_dim,
        embed_dim=args.projection_dim
    ).to(device)

    # 如果是评估模式，加载预训练模型
    save_model_path = f"./models/CLIP/checkpoints/my_clip_{args.image_encoder_type}_epoch_{args.num_epochs}.pth"
    if args.train:
        # 训练模式
        train(args, model, train_dataloader, device)
    if args.eval:
        print(f"Loading model from {save_model_path}")
        model.load_state_dict(torch.load(save_model_path))
        criterion = ContrastiveLoss()
        evaluate(args, model, eval_dataloader, criterion, device)
    