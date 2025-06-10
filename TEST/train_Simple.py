import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 数据准备
transforms = transforms.Compose([
    transforms.ToTensor(), # 转换为张量
    transforms.Normalize((0.5,), (0.5,)) # 标准化
])

# 使用MNIST数据集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# 模型初始化
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),   # 展平输入
            nn.Linear(28*28, 128),  # 输入28*28的图像
            nn.ReLU(),  # 激活函数
            nn.Linear(128, 10)  # 输出10个类别
        )

    def forward(self, x):
        return self.fc(x)
    
model = SimpleNN()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss() # 交叉熵损失函数
optimizer = optim.SGD(model.parameters(), lr=0.1) # 随机梯度下降优化器

# 训练流程
epochs = 5
for epoch in range(epochs):
    model.train() # 设置模型为训练模式
    running_loss = 0.0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        # 清除梯度
        optimizer.zero_grad()
        # 前向传播
        # outputs:[B, C] labels:[B,]
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")


# 模型评估
model.eval() # 设置模型为评估模式
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1) # 取最大值的索引
        correct += (predicted == labels).sum().item() # 统计正确预测的数量
        total += labels.size(0)

print(f"Accuracy: {100 * correct / total:.2f}%")