import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import sys
import argparse
from tqdm import tqdm
import time
from pathlib import Path

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入DETR模型相关模块
from model.detr import DETR
from model.backbone import Backbone
from model.transformer import Transformer
from model.matcher import HungarianMatcher
from model.criterion import SetCriterion
from utils.dataset import PascalVOCDataset, collate_fn
from utils.utils import NestedTensor, build_model, build_criterion
from utils.train_utils import train_one_epoch
from utils.eval_utils import build_model, build_criterion

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Train DETR Object Detection Model")
    
    # 路径相关参数
    parser.add_argument("--data_dir", type=str, default="./data/Pascal_VOC")
    parser.add_argument("--save_dir", type=str, default="./models/DETR/checkpoints")
    parser.add_argument("--log_dir", type=str, default="./models/DETR/logs")

    # 训练相关参数
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    
    # 模型相关参数
    parser.add_argument("--num_queries", type=int, default=100)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--nheads", type=int, default=8)
    parser.add_argument("--num_encoder_layers", type=int, default=6)
    parser.add_argument("--num_decoder_layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    
    # 损失相关参数
    parser.add_argument("--set_cost_class", type=float, default=1.0)
    parser.add_argument("--set_cost_bbox", type=float, default=5.0)
    parser.add_argument("--set_cost_giou", type=float, default=2.0)
    parser.add_argument("--loss_ce", type=float, default=1.0)
    parser.add_argument("--loss_bbox", type=float, default=5.0)
    parser.add_argument("--loss_giou", type=float, default=2.0)
    parser.add_argument("--eos_coef", type=float, default=0.1)
    
    # 其他参数
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--save_steps", type=int, default=10)
    parser.add_argument("--print_steps", type=int, default=50)
    parser.add_argument("--max_size", type=int, default=800)
    
    return parser.parse_args()


def save_checkpoint(model, optimizer, epoch, loss, save_path):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, save_path)


def load_checkpoint(model, optimizer, checkpoint_path):
    """加载检查点"""
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
            transform=transform,
            max_size=args.max_size
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_fn
        )
    # 评估数据集
    if args.eval:
        val_dataset = PascalVOCDataset(
            data_dir=args.data_dir,
            split='val',
            transform=transform,
            max_size=args.max_size
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
    criterion = build_criterion(args).to(device)
    
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
    
    # 恢复训练
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming training from {args.resume}")
        start_epoch, _ = load_checkpoint(model, optimizer, args.resume)
        start_epoch += 1
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 训练循环
    if args.train:
        print("Starting training...")
        for epoch in range(start_epoch, args.num_epochs):
            # 训练
            train_loss = train_one_epoch(
                model, criterion, train_loader, optimizer, device, epoch, args
            )
            
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
                    best_model_path = os.path.join(args.save_dir, 'best_model.pth')
                    save_checkpoint(model, optimizer, epoch, val_loss, best_model_path)
                    print(f"Best model saved to {best_model_path}")
            
            # 定期保存检查点
            if (epoch + 1) % args.save_steps == 0:
                checkpoint_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pth')
                save_checkpoint(model, optimizer, epoch, train_loss, checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")
        
        # 保存最终模型
        final_model_path = os.path.join(args.save_dir, 'final_model.pth')
        save_checkpoint(model, optimizer, args.num_epochs-1, train_loss, final_model_path)
        print(f"Final model saved to {final_model_path}")
    
    # 仅评估
    elif args.eval:
        print("Starting evaluation...")
        evaluate(model, criterion, val_loader, device)


if __name__ == "__main__":
    main()
