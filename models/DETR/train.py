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
import logging
from pathlib import Path

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入DETR模型相关模块
from model.detr import DETR
from model.backbone import Backbone
from model.transformer import Transformer
from model.matcher import HungarianMatcher
from model.criterion import SetCriterion
from utils.dataset import PascalVOCDataset, get_transforms, collate_fn
from utils.utils import NestedTensor


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Train DETR Object Detection Model")
    
    # 路径相关参数
    parser.add_argument("--data_dir", type=str, default="/home/hpc/Desktop/Pytorch/data/Pascal_VOC",
                        help="Pascal VOC dataset directory")
    parser.add_argument("--save_dir", type=str, default="/home/hpc/Desktop/Pytorch/models/DETR/checkpoints",
                        help="Directory to save model checkpoints")
    parser.add_argument("--log_dir", type=str, default="/home/hpc/Desktop/Pytorch/models/DETR/logs",
                        help="Directory to save training logs")
    
    # 训练相关参数
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--device", type=str, default="auto", help="Training device (cuda/cpu/auto)")
    
    # 模型相关参数
    parser.add_argument("--num_queries", type=int, default=100, help="Number of object queries")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Transformer hidden dimension")
    parser.add_argument("--nheads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num_encoder_layers", type=int, default=6, help="Number of encoder layers")
    parser.add_argument("--num_decoder_layers", type=int, default=6, help="Number of decoder layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    
    # 损失相关参数
    parser.add_argument("--set_cost_class", type=float, default=1.0, help="Class coefficient in the matching cost")
    parser.add_argument("--set_cost_bbox", type=float, default=5.0, help="L1 box coefficient in the matching cost")
    parser.add_argument("--set_cost_giou", type=float, default=2.0, help="GIoU coefficient in the matching cost")
    parser.add_argument("--loss_ce", type=float, default=1.0, help="Classification loss coefficient")
    parser.add_argument("--loss_bbox", type=float, default=5.0, help="L1 box loss coefficient")
    parser.add_argument("--loss_giou", type=float, default=2.0, help="GIoU loss coefficient")
    parser.add_argument("--eos_coef", type=float, default=0.1, help="Relative classification weight of no-object class")
    
    # 其他参数
    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--eval", action="store_true", help="Run evaluation")
    parser.add_argument("--resume", type=str, default="", help="Resume training from checkpoint")
    parser.add_argument("--save_freq", type=int, default=10, help="Save checkpoint every N epochs")
    parser.add_argument("--print_freq", type=int, default=50, help="Print training stats every N iterations")
    parser.add_argument("--max_size", type=int, default=800, help="Maximum image size")
    
    return parser.parse_args()


def setup_logging(log_dir: str):
    """设置日志"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"training_{time.strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def build_model(args):
    """构建DETR模型"""
    # 构建backbone
    backbone = Backbone(
        name='resnet50',
        train_backbone=True,
        return_interm_layers=False,
        dilation=False
    )
    
    # 构建transformer
    transformer = Transformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=2048,
        dropout=args.dropout,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=True
    )
    
    # 构建DETR模型
    model = DETR(
        backbone=backbone,
        transformer=transformer,
        num_classes=20,  # Pascal VOC有20个类别
        num_queries=args.num_queries,
        return_intermediate_dec=True
    )
    
    return model


def build_criterion(args):
    """构建损失函数"""
    # 构建匈牙利匹配器
    matcher = HungarianMatcher(
        cost_class=args.set_cost_class,
        cost_bbox=args.set_cost_bbox,
        cost_giou=args.set_cost_giou
    )
    
    # 损失权重
    weight_dict = {
        'loss_ce': args.loss_ce,
        'loss_bbox': args.loss_bbox,
        'loss_giou': args.loss_giou
    }
    
    # 添加辅助损失权重
    aux_weight_dict = {}
    for i in range(args.num_decoder_layers - 1):
        aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
    weight_dict.update(aux_weight_dict)
    
    # 构建损失函数
    criterion = SetCriterion(
        num_classes=20,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=args.eos_coef
    )
    
    return criterion


def create_nested_tensor(images):
    """将图像转换为NestedTensor格式"""
    batch_size, channels, height, width = images.shape
    
    # 创建mask（对于固定尺寸的图像，mask全为False）
    mask = torch.zeros((batch_size, height, width), dtype=torch.bool, device=images.device)
    
    return NestedTensor(images, mask)


def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, logger, args):
    """训练一个epoch"""
    model.train()
    criterion.train()
    
    running_loss = 0.0
    num_batches = len(data_loader)
    
    pbar = tqdm(data_loader, desc=f"Epoch {epoch}")
    for batch_idx, (images, targets) in enumerate(pbar):
        # 移动到设备
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # 创建NestedTensor
        nested_images = create_nested_tensor(images)
        
        # 前向传播
        outputs = model(nested_images)
        
        # 计算损失
        loss_dict = criterion(outputs, targets)
        losses = sum(loss_dict[k] * criterion.weight_dict[k] for k in loss_dict.keys() if k in criterion.weight_dict)
        
        # 反向传播
        optimizer.zero_grad()
        losses.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        
        optimizer.step()
        
        # 统计
        running_loss += losses.item()
        
        # 更新进度条
        pbar.set_postfix({
            'Loss': f"{losses.item():.4f}",
            'Avg Loss': f"{running_loss/(batch_idx+1):.4f}"
        })
        
        # 打印详细信息
        if batch_idx % args.print_freq == 0:
            loss_str = " | ".join([f"{k}: {v.item():.4f}" for k, v in loss_dict.items()])
            logger.info(f"Epoch [{epoch}][{batch_idx}/{num_batches}] | Total Loss: {losses.item():.4f} | {loss_str}")
    
    avg_loss = running_loss / num_batches
    logger.info(f"Epoch {epoch} completed | Average Loss: {avg_loss:.4f}")
    
    return avg_loss


@torch.no_grad()
def evaluate(model, criterion, data_loader, device, logger):
    """评估模型"""
    model.eval()
    criterion.eval()
    
    running_loss = 0.0
    num_batches = len(data_loader)
    
    pbar = tqdm(data_loader, desc="Evaluating")
    for batch_idx, (images, targets) in enumerate(pbar):
        # 移动到设备
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # 创建NestedTensor
        nested_images = create_nested_tensor(images)
        
        # 前向传播
        outputs = model(nested_images)
        
        # 计算损失
        loss_dict = criterion(outputs, targets)
        losses = sum(loss_dict[k] * criterion.weight_dict[k] for k in loss_dict.keys() if k in criterion.weight_dict)
        
        running_loss += losses.item()
        
        pbar.set_postfix({'Loss': f"{losses.item():.4f}"})
    
    avg_loss = running_loss / num_batches
    logger.info(f"Evaluation completed | Average Loss: {avg_loss:.4f}")
    
    return avg_loss


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
    
    # 验证路径
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 设置日志
    logger = setup_logging(args.log_dir)
    logger.info(f"Training arguments: {args}")
    
    # 设置设备
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # 准备数据
    if args.train:
        train_transform = get_transforms(train=True, max_size=args.max_size)
        train_dataset = PascalVOCDataset(
            data_dir=args.data_dir,
            split='train',
            transform=train_transform,
            max_size=args.max_size
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )
        logger.info(f"Training dataset: {len(train_dataset)} images")
    
    if args.eval:
        val_transform = get_transforms(train=False, max_size=args.max_size)
        val_dataset = PascalVOCDataset(
            data_dir=args.data_dir,
            split='val',
            transform=val_transform,
            max_size=args.max_size
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )
        logger.info(f"Validation dataset: {len(val_dataset)} images")
    
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
        logger.info(f"Resuming training from {args.resume}")
        start_epoch, _ = load_checkpoint(model, optimizer, args.resume)
        start_epoch += 1
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 训练循环
    if args.train:
        logger.info("Starting training...")
        for epoch in range(start_epoch, args.num_epochs):
            # 训练
            train_loss = train_one_epoch(
                model, criterion, train_loader, optimizer, device, epoch, logger, args
            )
            
            # 更新学习率
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"Learning rate: {current_lr:.6f}")
            
            # 验证
            if args.eval:
                val_loss = evaluate(model, criterion, val_loader, device, logger)
                
                # 保存最佳模型
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_model_path = os.path.join(args.save_dir, 'best_model.pth')
                    save_checkpoint(model, optimizer, epoch, val_loss, best_model_path)
                    logger.info(f"Best model saved to {best_model_path}")
            
            # 定期保存检查点
            if (epoch + 1) % args.save_freq == 0:
                checkpoint_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pth')
                save_checkpoint(model, optimizer, epoch, train_loss, checkpoint_path)
                logger.info(f"Checkpoint saved to {checkpoint_path}")
        
        # 保存最终模型
        final_model_path = os.path.join(args.save_dir, 'final_model.pth')
        save_checkpoint(model, optimizer, args.num_epochs-1, train_loss, final_model_path)
        logger.info(f"Final model saved to {final_model_path}")
    
    # 仅评估
    elif args.eval:
        logger.info("Starting evaluation...")
        evaluate(model, criterion, val_loader, device, logger)


if __name__ == "__main__":
    main()