import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
import os
from tqdm import tqdm
import argparse

from models.DETR.utils.utils import cxcywh_to_xyxy
from models.DETR.utils.dataset import PascalVOCDataset
from models.DETR.model import DETR, Backbone, Transformer
from models.DETR.utils.utils import create_nested_tensor

# 定义VOC类别列表
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

# 定义不同的颜色
COLORS = [
    (0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 0, 255), 
    (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
    (128, 0, 128), (0, 128, 128), (64, 0, 0), (0, 64, 0), (0, 0, 64), 
    (64, 64, 0), (64, 0, 64), (0, 64, 64), (192, 0, 0), (0, 192, 0)
]

def parse_args():
    parser = argparse.ArgumentParser(description="DETR 预测可视化工具")

    parser.add_argument("--model_path", type=str, default="models/DETR/checkpoints/best_model_epoch_135.pth", help="DETR模型检查点路径")
    parser.add_argument("--image_path", type=str, default="./data/Pascal_VOC/VOC2012_test/VOC2012_test/JPEGImages/2008_000014.jpg", help="输入图像或图像目录的路径")
    parser.add_argument("--output_dir", type=str, default="./models/DETR/results/visualize", help="输出目录")
    parser.add_argument("--conf_threshold", type=float, default=0.5, help="置信度阈值")
    
    # 模型相关参数
    parser.add_argument("--num_queries", type=int, default=25)
    parser.add_argument("--num_classes", type=int, default=20)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--nheads", type=int, default=8)
    parser.add_argument("--num_encoder_layers", type=int, default=6)
    parser.add_argument("--num_decoder_layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)

    args = parser.parse_args()
    return args

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
        print(f"=====> 模型加载成功 (Epoch: {epoch + 1} | Train Loss: {checkpoint.get('loss', 'N/A')})")
    else:
        model.load_state_dict(checkpoint)
        print("=====> 模型参数加载成功")
    
    return model

@torch.no_grad()
def visualize_predictions(model, image_path, output_path, device, conf_threshold=0.5):
    """可视化模型在单张图像上的预测结果"""
    # 加载图像
    image = Image.open(image_path).convert("RGB")
    

    # 预处理图像
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0).to(device) # [1, C, H, W]
    H, W = image_tensor.shape[2], image_tensor.shape[3]  # 获取图像的高度和宽度

    # 创建嵌套张量
    masks = torch.zeros((1, H, W), dtype=torch.bool, device=device)
    nested_tensor = create_nested_tensor(image_tensor, masks)
    # 模型推理
    model.eval()
    outputs = model(nested_tensor)
    
    # 获取预测结果
    pred_labels = outputs['pred_labels'][0].softmax(-1)  # [num_queries, num_classes+1]
    pred_boxes = outputs['pred_boxes'][0]                # [num_queries, 4]
    
    # 只考虑前20个类别的分数（忽略背景类）
    scores, labels = pred_labels[:, :-1].max(dim=1)

    # 筛选置信度高于阈值的预测
    keep = scores > conf_threshold
    pred_labels = pred_labels[keep]  # 只保留前20个类别的分数
    pred_boxes = pred_boxes[keep]
    
    # 转换坐标格式
    boxes_xyxy = cxcywh_to_xyxy(pred_boxes)

    # 将归一化坐标转换为原图像坐标
    x_min, y_min, x_max, y_max = boxes_xyxy.unbind(1)
    x_min *= W
    y_min *= H
    x_max *= W
    y_max *= H
    boxes_xyxy = torch.stack([x_min, y_min, x_max, y_max], dim=-1)

    # 可视化结果
    # 创建图像副本以便绘制
    image_draw = image.copy()
    draw = ImageDraw.Draw(image_draw)
    
    # 记录检测到的目标数量
    num_detections = 0
    
    # 对每个预测框进行绘制
    for i in range(len(boxes_xyxy)):
        box = boxes_xyxy[i].cpu().numpy()
        scores, label_idx = pred_labels[i, :-1].max(dim=0)
        score = scores.item()
        label_idx = label_idx.item()
        
        # 跳过置信度低于阈值的预测
        if score < conf_threshold:
            continue
        
        num_detections += 1
        
        # 获取边界框坐标
        x_min, y_min, x_max, y_max = box
        
        # 获取该类别对应的颜色
        color = COLORS[label_idx % len(COLORS)]
        
        # 绘制边界框
        draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=3)
        
        # 添加类别标签和置信度
        label_text = f"{VOC_CLASSES[label_idx]}: {score:.2f}"
        
        # 设置字体
        try:
            # 尝试加载更好的字体
            font = ImageFont.truetype("arial.ttf", 18)
        except IOError:
            # 使用默认字体
            font = ImageFont.load_default()
        
        # 计算文本大小并设置背景
        text_bbox = draw.textbbox((0, 0), label_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # 绘制文本背景
        draw.rectangle([x_min, y_min - text_height - 4, x_min + text_width, y_min], fill=color)
        
        # 绘制文本
        draw.text((x_min, y_min - text_height - 4), label_text, fill=(255, 255, 255), font=font)
    
    # 保存结果图像
    image_draw.save(output_path)
    print(f"结果已保存至: {output_path}")
    
    return num_detections
        
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型
    model = build_model(args).to(device)
    model = load_model(model, args.model_path)
    # 如果提供了图像目录，处理目录中的所有图像
    if os.path.isdir(args.image_path):
        image_files = [f for f in os.listdir(args.image_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        total_detections = 0
        
        for img_file in tqdm(image_files, desc="处理图像"):
            img_path = os.path.join(args.image_path, img_file)
            output_path = os.path.join(args.output_dir, f"pred_{img_file}")
            detections = visualize_predictions(
                model, img_path, output_path, device, args.conf_threshold
            )
            total_detections += detections
        
        print(f"总共检测到 {total_detections} 个目标，平均每张图像 {total_detections / len(image_files):.2f} 个目标")
    else:
        # 处理单张图像
        output_path = os.path.join(args.output_dir, f"pred_{os.path.basename(args.image_path)}")
        visualize_predictions(
            model, args.image_path, output_path, device, args.conf_threshold
        )

if __name__ == "__main__":
    args = parse_args()
    main(args)
