from model.vgg import VGG
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm # 导入 tqdm 库
import yaml

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
config = load_config('./models/VGG/configs/train_config.yaml')


transforms = transforms.Compose([
    transforms.Resize((224, 224)), # 调整大小
    transforms.ToTensor(), # 转换为张量
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]) # 标准化
])
train_cifar10 = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms) # 50000
test_cifar10 = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms) # 10000
train_loader = DataLoader(dataset=train_cifar10, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_cifar10, batch_size=64, shuffle=False)
# 创建VGG16模型实例
model = VGG('VGG16', num_classes=10)    # CIFAR-10有10个类别
print("VGG16模型创建完成")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("训练将在{}上进行".format(device))
model.to(device)
# 定义损失函数
criterion = nn.CrossEntropyLoss() # 交叉熵损失函数
# 定义优化器
# optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001) # AdamW优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9) # 随机梯度下降优化器
num_epochs = 5
print("开始训练,一共{}个epoch".format(num_epochs))
for epoch in range(num_epochs):
    print("第{}个epoch".format(epoch + 1))
    model.train() # 设置模型为训练模式
    running_loss = 0.0
    # 遍历训练数据
    for batch_idx, (img, label) in enumerate(tqdm(train_loader)):
        # img.shape: [B, C, H, W]
        img, label = img.to(device), label.to(device)
        # 清除梯度
        optimizer.zero_grad()
        # 前向传播
        # outputs:[B, C] labels:[B,]
        outputs = model(img)
        # 计算损失
        loss = criterion(outputs, label)
        # 反向传播
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if (batch_idx + 1) % 100 == 0: # 例如，每处理 100 个批次打印一次信息
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                    f"Batch [{batch_idx+1}/{len(train_loader)}], "
                    f"Loss: {loss.item():.4f}")
            
    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}] 训练完成，平均损失: {epoch_loss:.4f}")

# 保存模型
save_path = f'vgg16_cifar10_epoch_{num_epochs}.pth'
torch.save(model.state_dict(), save_path)
print(f"模型已保存到 {save_path}")
