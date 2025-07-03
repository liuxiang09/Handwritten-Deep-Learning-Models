import torch
from tqdm import tqdm
from .utils import create_nested_tensor
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
    评估模型的mAP和mAR  
    Args:
        model (torch.nn.Module): 您自己实现的DETR模型。
        data_loader (torch.utils.data.DataLoader): 评估用的数据加载器。
        device (torch.device): 'cuda' or 'cpu'。
        conf_threshold (float): 用于过滤预测结果的置信度阈值。
    """


