import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import xml.etree.ElementTree as ET
import os
import numpy as np
from typing import Dict, List, Tuple, Optional

class PascalVOCDataset(Dataset):
    """
    Pascal VOC数据集加载器，用于DETR目标检测训练
    """
    
    # Pascal VOC 2012 类别
    VOC_CLASSES = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    
    def __init__(self, 
                 data_dir: str,
                 split: str = 'train',
                 transform: Optional[transforms.Compose] = None,
                 max_size: int = 800):
        """
        Args:
            data_dir: Pascal VOC数据根目录
            split: 'train', 'val', 'trainval'
            transform: 图像变换
            max_size: 图像最大尺寸
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.max_size = max_size
        
        # 类别到索引的映射
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.VOC_CLASSES)}
        self.num_classes = len(self.VOC_CLASSES)
        
        # 获取图像列表
        split_file = os.path.join(data_dir, 'VOC2012_train_val', 'VOC2012_train_val', 'ImageSets', 'Main', f'{split}.txt')
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")
            
        with open(split_file, 'r') as f:
            self.image_ids = [line.strip() for line in f.readlines()]
        
        # 设置路径
        self.images_dir = os.path.join(data_dir, 'VOC2012_train_val', 'VOC2012_train_val', 'JPEGImages')
        self.annotations_dir = os.path.join(data_dir, 'VOC2012_train_val', 'VOC2012_train_val', 'Annotations')
        
        print(f"Loaded {len(self.image_ids)} images for {split} split")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Dict:
        image_id = self.image_ids[idx]
        
        # 加载图像
        image_path = os.path.join(self.images_dir, f'{image_id}.jpg')
        image = Image.open(image_path).convert('RGB')
        
        # 加载标注
        annotation_path = os.path.join(self.annotations_dir, f'{image_id}.xml')
        boxes, labels = self._parse_annotation(annotation_path)
        
        # 获取原始图像尺寸
        orig_w, orig_h = image.size
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        # 获取变换后的尺寸
        if isinstance(image, torch.Tensor):
            _, new_h, new_w = image.shape
        else:
            new_w, new_h = image.size
        
        # 缩放边界框坐标到变换后的图像尺寸，并归一化到[0,1]
        if len(boxes) > 0:
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * (new_w / orig_w) / new_w  # x坐标
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * (new_h / orig_h) / new_h  # y坐标
            
            # 转换为中心点格式 (x1,y1,x2,y2) -> (cx,cy,w,h)
            cx = (boxes[:, 0] + boxes[:, 2]) / 2
            cy = (boxes[:, 1] + boxes[:, 3]) / 2
            w = boxes[:, 2] - boxes[:, 0]
            h = boxes[:, 3] - boxes[:, 1]
            boxes = torch.stack([cx, cy, w, h], dim=1)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'area': (boxes[:, 2] * boxes[:, 3]) if len(boxes) > 0 else torch.tensor([]),
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64) if len(boxes) > 0 else torch.tensor([]),
            'orig_size': torch.tensor([orig_h, orig_w]),
            'size': torch.tensor([new_h, new_w])
        }
        
        return image, target

    def _parse_annotation(self, annotation_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """解析XML标注文件"""
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        
        boxes = []
        labels = []
        
        for obj in root.findall('object'):
            # 获取类别
            class_name = obj.find('name').text
            if class_name not in self.class_to_idx:
                continue
            
            label = self.class_to_idx[class_name]
            labels.append(label)
            
            # 获取边界框
            bbox = obj.find('bndbox')
            x1 = float(bbox.find('xmin').text)
            y1 = float(bbox.find('ymin').text)
            x2 = float(bbox.find('xmax').text)
            y2 = float(bbox.find('ymax').text)
            
            boxes.append([x1, y1, x2, y2])
        
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        
        return boxes, labels


def get_transforms(train: bool = True, max_size: int = 800):
    """获取数据变换"""
    if train:
        return transforms.Compose([
            transforms.Resize((max_size, max_size)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((max_size, max_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def collate_fn(batch):
    """
    自定义collate函数，处理不同数量的目标框
    """
    images, targets = zip(*batch)
    
    # 将图像堆叠
    images = torch.stack(images, dim=0)
    
    # 保持targets为列表，因为每个图像可能有不同数量的目标
    return images, list(targets)


# 测试代码
if __name__ == "__main__":
    # 测试数据集加载
    data_dir = "/home/hpc/Desktop/Pytorch/data/Pascal_VOC"
    
    if os.path.exists(data_dir):
        transform = get_transforms(train=True)
        dataset = PascalVOCDataset(data_dir, split='train', transform=transform)
        
        print(f"Dataset size: {len(dataset)}")
        print(f"Number of classes: {dataset.num_classes}")
        print(f"Classes: {dataset.VOC_CLASSES}")
        
        # 测试第一个样本
        if len(dataset) > 0:
            image, target = dataset[0]
            print(f"Image shape: {image.shape}")
            print(f"Number of objects: {len(target['labels'])}")
            print(f"Labels: {target['labels']}")
            print(f"Boxes shape: {target['boxes'].shape}")
            
            # 测试DataLoader
            dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
            images, targets = next(iter(dataloader))
            print(f"Batch images shape: {images.shape}")
            print(f"Batch targets length: {len(targets)}")
    else:
        print(f"Data directory not found: {data_dir}")
