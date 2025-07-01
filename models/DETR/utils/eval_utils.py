import torch
from tqdm import tqdm


@torch.no_grad()
def evaluate(model, criterion, data_loader, device):
    """评估模型"""
    model.eval()
    criterion.eval()
    
    running_loss = 0.0
    num_batches = len(data_loader)
    
    pbar = tqdm(data_loader, desc="Evaluating")
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
        
        running_loss += losses.item()
        
        pbar.set_postfix({'Loss': f"{losses.item():.4f}"})
    
    avg_loss = running_loss / num_batches
    print(f"Evaluation completed | Average Loss: {avg_loss:.4f}")
    
    return avg_loss
