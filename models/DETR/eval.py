import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import sys
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from tqdm import tqdm

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入DETR模型相关模块
from model.detr import DETR
from model.backbone import Backbone
from model.transformer import Transformer
from utils.dataset import PascalVOCDataset, get_transforms, collate_fn
from utils.utils import NestedTensor, cxcywh_to_xyxy


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Evaluate DETR Object Detection Model")
    
    parser.add_argument("--data_dir", type=str, default="/home/hpc/Desktop/Pytorch/data/Pascal_VOC",
                        help="Pascal VOC dataset directory")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="/home/hpc/Desktop/Pytorch/models/DETR/eval_results",
                        help="Directory to save evaluation results")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda/cpu/auto)")
    parser.add_argument("--confidence_threshold", type=float, default=0.5, help="Confidence threshold for detection")
    parser.add_argument("--max_size", type=int, default=800, help="Maximum image size")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to visualize")
    
    return parser.parse_args()


def build_model():
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
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=True
    )
    
    # 构建DETR模型
    model = DETR(
        backbone=backbone,
        transformer=transformer,
        num_classes=20,  # Pascal VOC有20个类别
        num_queries=100,
        return_intermediate_dec=True
    )
    
    return model


def load_model(checkpoint_path, device):
    """加载训练好的模型"""
    model = build_model()
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded from {checkpoint_path}")
    if 'epoch' in checkpoint:
        print(f"Trained for {checkpoint['epoch']+1} epochs")
    if 'loss' in checkpoint:
        print(f"Final training loss: {checkpoint['loss']:.4f}")
    
    return model


def create_nested_tensor(images):
    """将图像转换为NestedTensor格式"""
    batch_size, channels, height, width = images.shape
    mask = torch.zeros((batch_size, height, width), dtype=torch.bool, device=images.device)
    return NestedTensor(images, mask)


@torch.no_grad()
def detect_objects(model, images, confidence_threshold=0.5):
    """
    使用模型检测目标
    
    Args:
        model: DETR模型
        images: 输入图像 [B, 3, H, W]
        confidence_threshold: 置信度阈值
    
    Returns:
        检测结果列表
    """
    nested_images = create_nested_tensor(images)
    outputs = model(nested_images)
    
    # 获取预测结果
    pred_logits = outputs['pred_logits']  # [B, num_queries, num_classes+1]
    pred_boxes = outputs['pred_boxes']    # [B, num_queries, 4]
    
    # 计算概率
    prob = pred_logits.softmax(-1)  # [B, num_queries, num_classes+1]
    scores, labels = prob[..., :-1].max(-1)  # 排除"无目标"类别
    
    results = []
    for i in range(len(images)):
        # 过滤低置信度检测
        keep = scores[i] > confidence_threshold
        
        result = {
            'scores': scores[i][keep],
            'labels': labels[i][keep],
            'boxes': pred_boxes[i][keep],
        }
        results.append(result)
    
    return results


def visualize_detections(image, detections, class_names, save_path=None):
    """
    可视化检测结果
    
    Args:
        image: PIL图像
        detections: 检测结果字典
        class_names: 类别名称列表
        save_path: 保存路径
    """
    draw = ImageDraw.Draw(image)
    
    # 尝试加载字体
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(class_names)))
    
    for i, (score, label, box) in enumerate(zip(
        detections['scores'], detections['labels'], detections['boxes']
    )):
        # 转换边界框格式 (cx, cy, w, h) -> (x1, y1, x2, y2)
        cx, cy, w, h = box
        x1 = (cx - w/2) * image.width
        y1 = (cy - h/2) * image.height
        x2 = (cx + w/2) * image.width
        y2 = (cy + h/2) * image.height
        
        # 绘制边界框
        color = tuple(int(c * 255) for c in colors[label][:3])
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        
        # 绘制标签
        class_name = class_names[label]
        text = f"{class_name}: {score:.2f}"
        draw.text((x1, y1-20), text, fill=color, font=font)
    
    if save_path:
        image.save(save_path)
        print(f"Visualization saved to {save_path}")
    
    return image


@torch.no_grad()
def evaluate_model(model, data_loader, device, args):
    """评估模型"""
    print("Starting evaluation...")
    
    model.eval()
    total_detections = 0
    total_images = 0
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Pascal VOC类别名称
    class_names = PascalVOCDataset.VOC_CLASSES
    
    sample_count = 0
    for batch_idx, (images, targets) in enumerate(tqdm(data_loader, desc="Evaluating")):
        images = images.to(device)
        
        # 检测目标
        detections = detect_objects(model, images, args.confidence_threshold)
        
        # 统计
        for i, detection in enumerate(detections):
            total_detections += len(detection['scores'])
            total_images += 1
            
            # 可视化前几个样本
            if sample_count < args.num_samples:
                # 反归一化图像
                image_tensor = images[i].cpu()
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                image_tensor = image_tensor * std + mean
                image_tensor = torch.clamp(image_tensor, 0, 1)
                
                # 转换为PIL图像
                image_pil = transforms.ToPILImage()(image_tensor)
                
                # 可视化检测结果
                vis_image = visualize_detections(
                    image_pil.copy(), 
                    detection, 
                    class_names,
                    save_path=os.path.join(args.output_dir, f"detection_{sample_count:03d}.jpg")
                )
                
                sample_count += 1
        
        # 打印进度
        if batch_idx % 10 == 0:
            avg_detections = total_detections / max(total_images, 1)
            print(f"Batch [{batch_idx}] - Avg detections per image: {avg_detections:.2f}")
    
    # 打印最终统计
    avg_detections = total_detections / max(total_images, 1)
    print(f"\nEvaluation completed!")
    print(f"Total images: {total_images}")
    print(f"Total detections: {total_detections}")
    print(f"Average detections per image: {avg_detections:.2f}")
    print(f"Confidence threshold: {args.confidence_threshold}")
    print(f"Results saved to: {args.output_dir}")


def main():
    args = parse_args()
    
    # 验证路径
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")
    
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    # 设置设备
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # 加载模型
    model = load_model(args.checkpoint, device)
    
    # 准备数据
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
    
    print(f"Validation dataset: {len(val_dataset)} images")
    
    # 评估模型
    evaluate_model(model, val_loader, device, args)


if __name__ == "__main__":
    main()
