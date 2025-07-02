import torch
from tqdm import tqdm
from .utils import create_nested_tensor, rescale_bboxes
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torch.nn.functional as F


@torch.no_grad()
def evaluate(model, criterion, data_loader, device):
    """评估模型"""
    model.eval()
    criterion.eval()
    
    running_loss = 0.0
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
        losses = sum(loss_dict[k] * criterion.weight_dict[k] for k in loss_dict.keys())
        
        # 统计
        running_loss += losses.item()
        
        # 更新进度条
        if batch_idx % 10 == 0:
            pbar.set_postfix({
                'Loss': f"{losses.item():.4f}",
                'CE': f"{loss_dict['loss_ce'].item():.4f}",
                'Bbox': f"{loss_dict['loss_bbox'].item():.4f}",
                'GIoU': f"{loss_dict['loss_giou'].item():.4f}",
            })
    avg_loss = running_loss / num_batches
    print(f"平均评估损失: {avg_loss:.4f}")
    return avg_loss


@torch.no_grad()
def evaluate_ap_ar(model, data_loader, device, conf_threshold=0.5): # 【修改点】移除了 postprocessor 参数
    """
    评估模型的mAP和mAR，无需外部后处理类。
    
    Args:
        model (torch.nn.Module): 您自己实现的DETR模型。
        data_loader (torch.utils.data.DataLoader): 评估用的数据加载器。
        device (torch.device): 'cuda' or 'cpu'。
        conf_threshold (float): 用于过滤预测结果的置信度阈值。
    """
    model.eval()

    metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox')

    print("🚀 开始进行AP/AR评估...")
    for data_dict in tqdm(data_loader, desc="Calculating AP/AR"):
        images = data_dict['images'].to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in data_dict['targets']]
        masks = data_dict['masks'].to(device)
        nested_images = create_nested_tensor(images, masks)

        # 获取模型预测结果
        outputs = model(nested_images)
        
        # 后处理步骤
        # outputs['pred_logits'] 的形状: [batch_size, num_queries, num_classes + 1]
        # outputs['pred_boxes'] 的形状: [batch_size, num_queries, 4]
        
        # 使用softmax将logits转换为概率
        probs = F.softmax(outputs['pred_logits'], -1)
        
        # 获取每个预测框的分数和类别标签
        # 我们忽略最后一个类别，因为它是"no object"背景类
        scores, labels = probs[..., :-1].max(-1)
        
        # 将预测框从归一化的 [center_x, center_y, width, height] 格式
        # 转换为绝对像素值的 [xmin, ymin, xmax, ymax] 格式
        image_sizes = torch.tensor([images.shape[-2], images.shape[-1]], device=images.device).repeat(images.shape[0], 1)  # [B, 2]
        scaled_boxes = rescale_bboxes(outputs['pred_boxes'], image_sizes)

        # 格式化 `preds` 和 `targets` 以符合 torchmetrics 的要求
        preds = []
        for i in range(len(targets)): # 遍历batch中的每张图片
            img_scores = scores[i]
            img_labels = labels[i]
            img_boxes = scaled_boxes[i]

            # 根据置信度阈值进行过滤
            keep = img_scores > conf_threshold
            
            preds.append({
                'scores': img_scores[keep],
                'labels': img_labels[keep],
                'boxes': img_boxes[keep],
            })

        # `targets` 的处理方式保持不变，同样需要转换坐标格式
        targets_for_metric = []
        for t in targets:
            # 同样使用rescale_bboxes来转换真实框
            # 假设真实框也是cxcywh归一化格式
            targets_for_metric.append({
                'boxes': rescale_bboxes(t['boxes'], t['orig_size'].unsqueeze(0)).squeeze(0),
                'labels': t['labels'],
            })

        # 5. 使用当前批次的数据更新评估器状态
        metric.update(preds, targets_for_metric)

    # 6. 在所有数据都处理完毕后，计算最终的评估结果
    print("✅ 评估完成，正在计算最终指标...")
    final_metrics = metric.compute()
    return final_metrics

