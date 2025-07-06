import torch
import argparse
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import datetime

# 导入DETR模型相关模块
from models.DETR.model import DETR, Backbone, Transformer
from models.DETR.utils import PascalVOCDataset, collate_fn, evaluate_ap_ar


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Evaluate DETR Object Detection Model")
    
    # 路径相关参数
    parser.add_argument("--data_dir", type=str, default="./data/Pascal_VOC/VOC2012_train_val/VOC2012_train_val")
    parser.add_argument("--model_path", type=str, default="models/DETR/checkpoints/best_model_epoch_135.pth", help="模型检查点路径")
    parser.add_argument("--result_dir", type=str, default="./models/DETR/results")
    
    # 评估相关参数
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--conf_threshold", type=float, default=0.5)
    
    # 模型相关参数
    parser.add_argument("--num_queries", type=int, default=25)
    parser.add_argument("--num_classes", type=int, default=20)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--nheads", type=int, default=8)
    parser.add_argument("--num_encoder_layers", type=int, default=6)
    parser.add_argument("--num_decoder_layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    
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


def load_model(model, model_path):
    """加载模型"""
    print(f"=====> 正在加载模型: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # 处理不同的保存格式
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 0)
        loss = checkpoint.get('loss', 'N/A')
        print(f"=====> 模型加载成功 (Epoch: {epoch + 1} | Train Loss: {loss})")
    else:
        model.load_state_dict(checkpoint)
        print("=====> 模型参数加载成功")
    
    return model, checkpoint


def main():
    args = parse_args()
    
    # 创建结果目录
    os.makedirs(args.result_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 验证数据集
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
    
    print(f"验证集大小: {len(val_dataset)}")
    
    # 构建和加载模型
    model = build_model(args).to(device)
    model, checkpoint = load_model(model, args.model_path)
    model.eval()
    
    print("=====> 开始评估...")
    
    # 评估AP和AR指标
    results = evaluate_ap_ar(args, model, val_loader, device, args.conf_threshold)

    # 确保结果目录存在
    os.makedirs(args.result_dir, exist_ok=True)
    
    # 写入文件，包含更多模型相关信息
    with open(os.path.join(args.result_dir, "eval_results.txt"), "w") as f:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"评估时间: {timestamp}\n")
        f.write(f"模型路径: {args.model_path}\n")
        f.write(f"训练轮次: Epoch {checkpoint.get('epoch', 'N/A') + 1}\n")
        f.write(f"训练损失: {checkpoint.get('loss', 'N/A')}\n")
        f.write("\n" + "="*50 + "\n")
        f.write("\n=== 评估指标 ===\n")
        f.write(f"mAP: {results['map'].item():.4f}\n")
        f.write(f"mAP_50: {results['map_50'].item():.4f}\n")
        f.write(f"mAP_75: {results['map_75'].item():.4f}\n")
        f.write(f"mAP_small: {results['map_small'].item():.4f}\n")
        f.write(f"mAP_medium: {results['map_medium'].item():.4f}\n")
        f.write(f"mAP_large: {results['map_large'].item():.4f}\n")
        f.write("="*50 + "\n")

if __name__ == "__main__":
    main()
