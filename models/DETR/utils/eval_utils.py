import torch
from tqdm import tqdm
from .utils import create_nested_tensor, cxcywh_to_xyxy
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torch.nn.functional as F
import pdb
import os
import datetime

@torch.no_grad()
def evaluate(model, criterion, data_loader, device):
    """评估模型"""
    model.eval()
    criterion.eval()
    
    running_loss = 0.0
    running_loss_ce = 0.0
    running_loss_bbox = 0.0
    running_loss_giou = 0.0
    num_batches = len(data_loader)
    
    pbar = tqdm(data_loader, desc="Evaluating")
    for batch_idx, data_dict in enumerate(pbar):
        # 移动到设备
        images = data_dict['images'].to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in data_dict['targets']]
        masks = data_dict['masks'].to(device)

        # 创建NestedTensor
        nested_images = create_nested_tensor(images, masks)
        
        # 前向传播
        outputs = model(nested_images)
        
        # 计算损失
        loss_dict = criterion(outputs, targets)
        losses = sum(loss_dict.values())

        # 统计
        running_loss += losses.item()
        running_loss_ce += loss_dict['loss_ce'].item()
        running_loss_bbox += loss_dict['loss_bbox'].item()
        running_loss_giou += loss_dict['loss_giou'].item()

        # 更新进度条
        if batch_idx % 10 == 0:
            pbar.set_postfix({
                'Loss': f"{running_loss/(batch_idx + 1):.4f}",
                'loss_ce': f"{running_loss_ce/(batch_idx + 1):.4f}",
                'loss_bbox': f"{running_loss_bbox/(batch_idx + 1):.4f}",
                'loss_giou': f"{running_loss_giou/(batch_idx + 1):.4f}",
            })
    avg_loss = running_loss / num_batches
    print(f"平均评估损失: {avg_loss:.4f}")
    return avg_loss


@torch.no_grad()
def evaluate_ap_ar(args, model, data_loader, device, conf_threshold=0.5):
    """
    评估模型的mAP和mAR  
    Args:
        model (torch.nn.Module): DETR模型
        data_loader (torch.utils.data.DataLoader): 评估用的数据加载器
        device (torch.device): 'cuda' or 'cpu'
        conf_threshold (float): 用于过滤预测结果的置信度阈值
    Returns:
        dict: 包含mAP和mAR等评估指标
    """
    model.eval()
    
    # 初始化评估指标 - 必须在CPU上计算
    metric = MeanAveragePrecision(
        box_format='xyxy',  # 重要：MeanAveragePrecision期望xyxy格式的边界框
        iou_type='bbox',
        iou_thresholds=[0.5, 0.75],
        rec_thresholds=None,
        max_detection_thresholds=[1, 10, 100]
    )
    
    pbar = tqdm(data_loader, desc="Evaluating AP/AR")
    
    for data_dict in pbar:
        images = data_dict['images'].to(device)
        masks = data_dict['masks'].to(device)
        
        # 创建NestedTensor
        nested_images = create_nested_tensor(images, masks)
        
        # 前向传播
        outputs = model(nested_images)

        # 后处理预测结果 - 所有计算都在GPU上进行
        preds = []
        for i in range(len(outputs['pred_labels'])):
            # 获取预测
            pred_labels = outputs['pred_labels'][i]  # [num_queries, num_classes+1]
            pred_boxes = outputs['pred_boxes'][i]    # [num_queries, 4] - 这是cxcywh格式，归一化到[0,1]
            
            # 计算每个类别的概率
            prob = F.softmax(pred_labels, dim=-1)
            
            # 排除背景类（最后一个类别），获取最高概率的类别及其概率
            scores, labels = prob[:, :-1].max(dim=1)  # 注意这里是dim=1而不是dim=-1
            
            # 过滤低置信度预测
            keep = scores > conf_threshold
            
            # 转换为xyxy格式 - 保持在[0,1]范围内
            pred_boxes_xyxy = cxcywh_to_xyxy(pred_boxes[keep])
            
            # 重要：所有数据必须移到CPU上
            preds.append({
                'boxes': pred_boxes_xyxy.cpu(),   # 转换为CPU tensor
                'scores': scores[keep].cpu(),     # 转换为CPU tensor
                'labels': labels[keep].cpu() + 1  # 标签从0开始，但torchmetrics期望从1开始
            })
        
        # 准备真实标签 - 从data_dict['targets']获取
        gt_targets = []
        for target in data_dict['targets']:
            # 将cxcywh格式的边界框转换为xyxy格式
            boxes_xyxy = cxcywh_to_xyxy(target['boxes'])
            
            # 所有数据必须移到CPU上
            gt_targets.append({
                'boxes': boxes_xyxy.cpu(),        # 转换为CPU tensor
                'labels': target['labels'].cpu() + 1  # 标签从0开始，但torchmetrics期望从1开始
            })
        
        # 更新指标
        metric.update(preds, gt_targets)

    # 计算最终指标
    results = metric.compute()
    
    # 打印主要指标
    print(f"\n===== 评估指标 =====")
    print(f"mAP: {results['map'].item():.4f}")
    print(f"mAP_50: {results['map_50'].item():.4f}")
    print(f"mAP_75: {results['map_75'].item():.4f}")
    print(f"mAP_small: {results['map_small'].item():.4f}")
    print(f"mAP_medium: {results['map_medium'].item():.4f}")
    print(f"mAP_large: {results['map_large'].item():.4f}")
    
    return results


