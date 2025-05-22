import torch
import torchvision
from torchvision import datasets, transforms
from VGG import VGG
from tqdm import tqdm # 导入 tqdm 库



transforms = transforms.Compose([
        transforms.Resize((224, 224)), # 调整大小
        transforms.ToTensor(), # 转换为张量
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]) # 标准化
    ])

test_cifar10 = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms) # 10000
test_cifar10_loader = torch.utils.data.DataLoader(test_cifar10, batch_size=64, shuffle=False)
print(len(test_cifar10))

# 加载模型
model = VGG('VGG16', num_classes=10)  # 使用VGG16模型
model.load_state_dict(torch.load("VGG/vgg16_cifar10_epoch_5.pth"))  # 加载模型参数

# 使用torchvision.models中的VGG16模型
# model = torchvision.models.vgg16(pretrained=True)  # CIFAR-10有10个类别

model.eval()  # 设置模型为评估模式
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
correct = 0 # 初始化正确预测的数量
total = 0   # 初始化总样本数量
# 评估模型
with torch.no_grad():
    for batch_idx, (img, label) in enumerate(tqdm(test_cifar10_loader, desc="评估进度")):
        img, label = img.to(device), label.to(device)
        outputs = model(img)
        _, predicted = torch.max(outputs.data, 1)   # 获取预测结果
        total += label.size(0)
        correct += (predicted == label).sum().item()

print(f"模型在测试集上的准确率: {100 * correct / total:.2f}%")

