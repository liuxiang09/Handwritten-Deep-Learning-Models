import torch
from tqdm import tqdm
from .utils import create_nested_tensor

def train_one_epoch(args, model, criterion, data_loader, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()
    criterion.train()
    
    running_loss = 0.0
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
        losses = sum(loss_dict[k] * criterion.weight_dict[k] for k in loss_dict.keys())
        
        # 反向传播
        optimizer.zero_grad()
        losses.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        
        optimizer.step()
        
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
    print(f"Epoch {epoch + 1} completed | Average Loss: {avg_loss:.4f}")
    
    return avg_loss