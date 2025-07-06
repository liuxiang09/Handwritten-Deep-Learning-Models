import torch
from tqdm import tqdm
from .utils import create_nested_tensor
import pdb


def train_one_epoch(args, model, criterion, data_loader, optimizer, scheduler, device, epoch):
    """训练一个epoch"""
    model.train()
    criterion.train()
    
    running_loss = 0.0
    running_loss_ce = 0.0
    running_loss_bbox = 0.0
    running_loss_giou = 0.0
    num_batches = len(data_loader)
    
    pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
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
        
        # 反向传播
        optimizer.zero_grad()
        losses.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        
        optimizer.step()
        scheduler.step()  # 更新学习率
        # 统计
        running_loss += losses.item()
        running_loss_ce += loss_dict['loss_ce'].item()
        running_loss_bbox += loss_dict['loss_bbox'].item()
        running_loss_giou += loss_dict['loss_giou'].item()
        
        # 更新进度条
        if (batch_idx + 1) % 10 == 0:
            pbar.set_postfix({
            'Loss': f"{running_loss/(batch_idx + 1):.4f}",
            'loss_ce': f"{running_loss_ce/(batch_idx + 1):.4f}",
            'loss_bbox': f"{running_loss_bbox/(batch_idx + 1):.4f}",
            'loss_giou': f"{running_loss_giou/(batch_idx + 1):.4f}",
        })
        
    
    avg_loss = running_loss / num_batches
    print(f"Epoch {epoch + 1} completed | Average Loss: {avg_loss:.4f} | Learning rate: {optimizer.param_groups[0]['lr']:.4f}")
    
    return avg_loss