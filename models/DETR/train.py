import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import argparse
from tqdm import tqdm
import time
from pathlib import Path

# 导入DETR模型相关模块
from models.DETR.model import DETR, Backbone, Transformer, HungarianMatcher, SetCriterion
from models.DETR.utils import PascalVOCDataset, collate_fn, train_one_epoch, evaluate


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Train DETR Object Detection Model")
    
    # 路径相关参数
    parser.add_argument("--data_dir", type=str, default="./data/Pascal_VOC/VOC2012_train_val/VOC2012_train_val")
    parser.add_argument("--save_dir", type=str, default="./models/DETR/checkpoints")
    parser.add_argument("--log_dir", type=str, default="./models/DETR/logs")
    parser.add_argument("--resume", type=str, default="", help="检查点路径，用于恢复训练")

    # 训练相关参数
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    
    # 模型相关参数
    parser.add_argument("--num_queries", type=int, default=100)
    parser.add_argument("--num_classes", type=int, default=20)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--nheads", type=int, default=8)
    parser.add_argument("--num_encoder_layers", type=int, default=6)
    parser.add_argument("--num_decoder_layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    
    # 损失相关参数
    parser.add_argument("--cost_class", type=float, default=1.0)
    parser.add_argument("--cost_bbox", type=float, default=5.0)
    parser.add_argument("--cost_giou", type=float, default=2.0)
    parser.add_argument("--loss_ce", type=float, default=1.0)
    parser.add_argument("--loss_bbox", type=float, default=5.0)
    parser.add_argument("--loss_giou", type=float, default=2.0)
    parser.add_argument("--eos_coef", type=float, default=0.1)
    
    # 其他参数
    parser.add_argument("--train", action="store_true", help="训练模式")
    parser.add_argument("--eval", action="store_true", help="训练时进行验证")
    parser.add_argument("--save_epochs", type=int, default=1)
    parser.add_argument("--print_steps", type=int, default=50)
    parser.add_argument("--max_size", type=int, default=800)
    
    return parser.parse_args()

def build_model(args):
    """构建DETR模型"""
    # 创建backbone
    backbone = Backbone(name="resnet50", train_backbone=True, return_interm_layers=False, dilation=False)
    
    # 创建transformer
    transformer = Transformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=2048,
        dropout=args.dropout
    )
    
    # 创建DETR模型
    model = DETR(
        backbone=backbone,
        transformer=transformer,
        num_classes=args.num_classes,
        num_queries=args.num_queries,
        return_intermediate_dec=False
    )
    
    return model

def build_criterion(args):
    """构建损失函数"""
    matcher = HungarianMatcher(
        cost_class=args.cost_class,
        cost_bbox=args.cost_bbox,
        cost_giou=args.cost_giou
    )
    
    criterion = SetCriterion(
        num_classes=20,  # Pascal VOC有20个类别（不包括背景）
        matcher=matcher,
        weight_dict={
            "loss_ce": args.loss_ce,
            "loss_bbox": args.loss_bbox,
            "loss_giou": args.loss_giou
        },
        eos_coef=args.eos_coef
    )
    
    return criterion

def save_checkpoint(model, optimizer, epoch, loss, save_path):
    """保存检查点"""
    print("=====> 正在保存检查点...")
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, save_path)


def load_model_for_eval(model, model_path):
    """加载模型用于评估"""
    print(f"=====> 正在加载模型: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # 处理不同的保存格式
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 0)
        loss = checkpoint.get('loss', 0.0)
        print(f"=====> 模型加载成功 (Epoch: {epoch}, Loss: {loss:.4f})")
    else:
        # 如果直接保存的是模型参数
        model.load_state_dict(checkpoint)
        print("=====> 模型参数加载成功")
    
    return model


def load_checkpoint(model, optimizer, checkpoint_path):
    """加载检查点"""
    print("=====> 正在加载检查点...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, loss


def main():
    args = parse_args()
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 训练数据集
    if args.train:
        train_dataset = PascalVOCDataset(
            data_dir=args.data_dir,
            split='train',
            transform=transform
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_fn
        )

    # 验证数据集
    if args.eval:
        val_dataset = PascalVOCDataset(
            data_dir=args.data_dir,
            split='val',
            transform=transform
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn
        )
    
    # 构建模型
    model = build_model(args).to(device)
    print("=====> 模型构建完成")
    criterion = build_criterion(args).to(device)
    print("=====> 损失函数构建完成")
    
    # 输出模型参数量
    print(f"\n模型总参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"模型可训练参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 设置优化器
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.learning_rate * 0.1,
        },
    ]
    optimizer = optim.AdamW(param_dicts, lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    start_epoch = 0
    best_loss = float('inf')
    
    # 加载预训练模型，恢复训练

    if args.resume and os.path.exists(args.resume):
        print(f"Resuming training from {args.resume}")
        start_epoch, _ = load_checkpoint(model, optimizer, args.resume)
        start_epoch += 1
        
    # 训练循环
    if args.train:
        print("Starting training...")
        for epoch in range(start_epoch, args.num_epochs):
            # 训练
            train_loss = train_one_epoch(args, model, criterion, train_loader, optimizer, device, epoch)

            # 更新学习率
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Learning rate: {current_lr:.6f}")
            
            # 验证
            if args.eval:
                val_loss = evaluate(model, criterion, val_loader, device)
                
                # 保存最佳模型
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_model_path = os.path.join(args.save_dir, f'best_model_epoch_{epoch+1}.pth')
                    save_checkpoint(model, optimizer, epoch, val_loss, best_model_path)
                    print(f"Best model saved to {best_model_path}")
            
            # 定期保存检查点
            if (epoch + 1) % args.save_epochs == 0:
                checkpoint_path = os.path.join(args.log_dir, f'checkpoint_epoch_{epoch+1}.pth')
                save_checkpoint(model, optimizer, epoch, train_loss, checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")
        
        print("Training completed!")


if __name__ == "__main__":
    main()
