from model.vgg import VGG
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Train and evaluate VGG model on CIFAR-10')
    
    # 路径相关参数
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--save_dir', type=str, default='./models/VGG/checkpoints', help='Directory to save model')
    
    # 模型相关参数
    parser.add_argument('--model', type=str, default='VGG11',choices=['VGG11', 'VGG13', 'VGG16', 'VGG19'], help='VGG model variant (VGG11, VGG13, VGG16, VGG19)')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes')
    
    # 训练和评估相关参数
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train')
    parser.add_argument('--sgd_lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--sgd_momentum', type=float, default=0.9, help='Momentum for SGD optimizer')
    parser.add_argument('--sgd_weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--adam_lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--adam_weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['sgd', 'adam'], help='Optimizer type')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--eval', action='store_true', help='Evaluate the model')
    parser.add_argument('--log_steps', type=int, default=50, help='Log every N steps')

    # 数据集相关参数
    parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['CIFAR10', 'CIFAR100'], help='Dataset name')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--pin_memory', action='store_true', help='Pin memory for faster GPU transfer')

    return parser.parse_args()

def _get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])

def train(args, model, train_loader, device):
    model.train()
    
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 定义优化器
    if args.optimizer.lower() == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=args.sgd_lr, 
            momentum=args.sgd_momentum,
            weight_decay=args.sgd_weight_decay
        )
    elif args.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.adam_lr,
            weight_decay=args.adam_weight_decay
        )
    
    print(f"Starting training for {args.epochs} epochs")
    
    for epoch in range(args.epochs):
        
        # 训练一个epoch
        running_loss = 0.0
        
        # 创建tqdm进度条
        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{args.epochs}")
        
        for batch_idx, (img, label) in enumerate(pbar):
            img, label = img.to(device), label.to(device)
            # img: [B, 3, 224, 224]
            # label: [B]
            optimizer.zero_grad()
            outputs = model(img) # [B, C]
            loss = criterion(outputs, label)
            loss.backward()
            # 在反向传播和优化器更新之前，使用梯度裁剪
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            running_loss += loss.item()
            
            # 更新tqdm进度条中的信息
            if (batch_idx + 1) % args.log_steps == 0:
                current_loss = loss.item()
                avg_loss = running_loss / (batch_idx + 1)
                pbar.set_postfix({'batch_loss': f'{current_loss:.4f}', 'avg_loss': f'{avg_loss:.4f}'})
        
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{args.epochs} completed, avg loss: {epoch_loss:.4f}")
    
    # 保存模型
    save_name = f"{args.model.lower()}_{args.dataset.lower()}_epoch_{args.epochs}.pth"
    save_path = os.path.join(args.save_dir, save_name)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    
    return model

def evaluate(args, model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (img, label) in enumerate(tqdm(test_loader, desc="Evaluating")):
            img, label = img.to(device), label.to(device)
            outputs = model(img)
            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    
    accuracy = 100 * correct / total
    print(f"✅ Accuracy on test set: {accuracy:.2f}%")
    return accuracy

def main():
    args = parse_args()
    
    # 确保保存目录存在
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 准备数据
    transform = _get_transforms()
    train_dataset = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform)
    print(f"total_dataset: {len(train_dataset) + len(test_dataset)}")
    print(f"train_dataset: {len(train_dataset)}")
    print(f"test_dataset: {len(test_dataset)}\n")
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers, 
        pin_memory=args.pin_memory, 
    )
    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers, 
        pin_memory=args.pin_memory, 
    )

    for idx, (img, label) in enumerate(train_loader):
        print(f"img shape: {img.shape}")
        print(f"label shape: {label.shape}\n")
        break

    # 创建模型
    model = VGG(args.model, num_classes=args.num_classes)
    print(f"Created {args.model} model with {args.num_classes} classes\n")
    model.to(device)
    
    # 统计模型参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Total parameters: {trainable_params + non_trainable_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}\n")
    

    # 训练模型
    if args.train:
        model = train(args, model, train_loader, device)
    
    # 评估模型
    if args.eval:
        # 自动生成 checkpoint 路径
        checkpoint_name = f"{args.model.lower()}_{args.dataset.lower()}_epoch_{args.epochs}.pth"
        checkpoint_path = os.path.join(args.save_dir, checkpoint_name)
        if os.path.exists(checkpoint_path):
            print(f"Loading model from {checkpoint_path}")
            model.load_state_dict(torch.load(checkpoint_path))
            print("Evaluating model")
            evaluate(args, model, test_loader, device)
        else:
            raise ValueError(f"Checkpoint not found: {checkpoint_path}")

if __name__ == "__main__":
    main()
