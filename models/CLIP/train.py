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

# 对比损失函数
# 由于一个图片有5个文本，所以无法采用常规的交叉熵损失函数
# class ContrastiveLoss(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, image_features, text_features, logit_scale):
#         # 归一化特征 (已经在 CLIP 模型的 forward 中完成)
        
#         # 计算相似度矩阵
#         logits = (image_features @ text_features.T) * logit_scale.exp()

#         # 创建标签 (对角线为正样本)
#         labels = torch.arange(len(logits)).to(logits.device)

#         # 计算图像到文本的损失 (行是图像，列是文本)
#         loss_i = F.cross_entropy(logits, labels)
        
#         # 计算文本到图像的损失 (转置 logits，行是文本，列是图像)
#         loss_t = F.cross_entropy(logits.T, labels)
        
#         # 返回平均损失
#         return (loss_i + loss_t) / 2

class ContrastiveLoss(nn.Module):
    """
    能够正确处理一个图像对应多个文本描述的对比损失函数。
    """
    def __init__(self):
        super().__init__()

    def forward(self, image_features, text_features, logit_scale):
        """
        Args:
            image_features: shape [N, D], N 是批次中的图片数量。
            text_features: shape [5*N, D], 对应 N 张图片的 5*N 个文本描述。
            logit_scale: 可学习的温度参数。
        """
        device = image_features.device
        num_images = image_features.shape[0]
        num_texts = text_features.shape[0]

        # 验证输入形状是否匹配
        if num_texts % num_images != 0 or num_texts // num_images != 5:
            raise ValueError("文本特征数量必须是图片特征数量的5倍。")

        # 计算相似度矩阵
        # logits_per_image shape: [N, 5*N]
        logits_per_image = (logit_scale.exp() * image_features @ text_features.T)
        # logits_per_text shape: [5*N, N]
        logits_per_text = logits_per_image.T

        # --- 正确的损失计算 ---

        # 1. 计算 loss_t (文找图): 这是一个标准的多类别分类问题
        # 每个文本都有一个正确的图片目标。
        # 创建标签 [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, ..., N-1, ...]
        text_labels = torch.arange(num_images, device=device).repeat_interleave(5)
        loss_t = F.cross_entropy(logits_per_text, text_labels)

        # 2. 计算 loss_i (图找文): 这是一个多标签分类问题
        # 每张图片有5个正确的文本目标。标准的 cross_entropy 不适用。
        # 我们需要创建一个 "多热" (multi-hot) 的标签矩阵。
        # ground_truth shape: [N, 5*N]
        ground_truth = torch.zeros(logits_per_image.shape, dtype=torch.float, device=device)
        for i in range(num_images):
            # 将图片i对应的5个文本位置标记为1
            start_idx = i * 5
            end_idx = start_idx + 5
            ground_truth[i, start_idx:end_idx] = 1.0
        
        # 使用二元交叉熵损失 (Binary Cross Entropy)
        # 它将每个输出logit视为一个独立的二元分类（是/不是 正确的匹配）
        loss_i = F.binary_cross_entropy_with_logits(logits_per_image, ground_truth)

        # 返回两个方向损失的平均值
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
    elif args.image_encoder_type == 'vit-pretrained':
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
    