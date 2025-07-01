import torch
from tqdm import tqdm


def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, args):
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
        if batch_idx % args.print_steps == 0:
            loss_str = " | ".join([f"{k}: {v.item():.4f}" for k, v in loss_dict.items()])
            print(f"Epoch [{epoch}][{batch_idx}/{num_batches}] | Total Loss: {losses.item():.4f} | {loss_str}")
    
    avg_loss = running_loss / num_batches
    print(f"Epoch {epoch} completed | Average Loss: {avg_loss:.4f}")
    
    return avg_loss