import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import argparse
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 导入DETR模型相关模块
from models.DETR.model import DETR, Backbone, Transformer
from models.DETR.utils import PascalVOCDataset, collate_fn, cxcywh_to_xyxy, box_iou, compute_ap
from models.DETR.utils.eval_utils import evaluate_ap_ar

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Test DETR Object Detection Model")
    
    # 路径相关参数
    parser.add_argument("--data_dir", type=str, default="./data/Pascal_VOC/VOC2012_test/VOC2012_test", help="数据集路径")
    parser.add_argument("--model_path", type=str, required=True, help="模型文件路径")
    parser.add_argument("--results_dir", type=str, default="./models/DETR/results", help="结果保存路径")

    # 测试相关参数
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载线程数")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"], help="测试数据集分割")
    
    # 模型相关参数
    parser.add_argument("--num_queries", type=int, default=100, help="查询数量")
    parser.add_argument("--num_classes", type=int, default=20, help="类别数量")
    parser.add_argument("--hidden_dim", type=int, default=256, help="隐藏层维度")
    parser.add_argument("--nheads", type=int, default=8, help="注意力头数")
    parser.add_argument("--num_encoder_layers", type=int, default=6, help="编码器层数")
    parser.add_argument("--num_decoder_layers", type=int, default=6, help="解码器层数")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout概率")
        
    # 评估参数
    parser.add_argument("--conf_threshold", type=float, default=0.5, help="置信度阈值")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="IoU阈值")
    parser.add_argument("--save_predictions", type=bool, default=True, help="保存预测结果")
    parser.add_argument("--save_visualizations", type=bool, default=True, help="保存可视化结果")
    
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
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
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


def main():
    args = parse_args()
    
    # 创建结果保存目录
    os.makedirs(args.results_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 测试数据集
    test_dataset = PascalVOCDataset(
        data_dir=args.data_dir,
        split=args.split,
        transform=transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    print(f"测试数据集大小: {len(test_dataset)}")
    print(f"测试批次数量: {len(test_loader)}")
    
    # 构建模型
    model = build_model(args).to(device)
    print("=====> 模型构建完成")
    
    # 输出模型参数量
    print(f"\n模型总参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"模型可训练参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 加载模型
    model = load_model(model, args.model_path)
    
    # 评估模型
    results = evaluate_ap_ar(model, test_loader, device, args.conf_threshold)
    
    # 输出结果
    print("\n" + "="*60)
    print("DETR检测评估结果 (COCO风格指标):")
    print("="*60)
    print(f"AP@0.5      : {results['AP@0.5']:.4f}")
    print(f"AP@0.75     : {results['AP@0.75']:.4f}")
    print(f"AP@[0.5:0.95]: {results['AP@[0.5:0.95]']:.4f}")
    print(f"置信度阈值: {args.conf_threshold}")
    
    print("\n各类别AP@0.5详情:")
    print("-" * 60)
    for i, (class_name, ap) in enumerate(zip(results['class_names'], results['class_aps@0.5'])):
        print(f"{class_name:12}: {ap:.4f}")
    
    print("="*60)
    
    # 保存结果
    if args.save_predictions:
        results_file = os.path.join(args.results_dir, f"Results_VOC_{args.split}.txt")
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write("DETR目标检测模型评估结果 (COCO风格指标)\n")
            f.write("="*60 + "\n")
            f.write(f"模型路径: {args.model_path}\n")
            f.write(f"数据集分割: {args.split}\n")
            f.write(f"置信度阈值: {args.conf_threshold}\n")
            f.write(f"批次大小: {args.batch_size}\n")
            f.write(f"\n总体指标:\n")
            f.write(f"AP@0.5      : {results['AP@0.5']:.4f}\n")
            f.write(f"AP@0.75     : {results['AP@0.75']:.4f}\n")
            f.write(f"AP@[0.5:0.95]: {results['AP@[0.5:0.95]']:.4f}\n")
            f.write(f"\n各类别AP@0.5详细指标:\n")
            f.write("-" * 60 + "\n")
            
            for class_name, ap in zip(results['class_names'], results['class_aps@0.5']):
                f.write(f"{class_name:12}: {ap:.4f}\n")
        
        print(f"结果已保存到: {results_file}")


if __name__ == "__main__":
    main()
