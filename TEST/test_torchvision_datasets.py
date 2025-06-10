import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

cifar10_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor()) # 50000
cifar10_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor()) # 10000

cifar10_train_loader = DataLoader(cifar10_train, batch_size=64, shuffle=True)
cifar10_test_loader = DataLoader(cifar10_test, batch_size=64, shuffle=False)

print(len(cifar10_train))
print(len(cifar10_test))

for batch_idx, (img, label) in enumerate(cifar10_train_loader):
    print(img.shape)  # 输出图像的形状
    print(label.shape)  # 输出标签的形状
    input()

