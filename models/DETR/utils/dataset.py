import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict
import torchvision.transforms as transforms
from PIL import Image
from torch import Tensor
import os
import xml.etree.ElementTree as ET


def collate_fn(batch):
    pass


class PascalVOCDataset(Dataset):
    """Pascal VOC数据集加载器"""
    
    CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    
    def __init__(self, data_dir: str, split: str = 'train', transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.CLASSES)}
        
        # 读取split文件
        split_file = os.path.join(data_dir, 'ImageSets', 'Main', f'{split}.txt')
        with open(split_file, 'r') as f:
            self.image_ids = [line.strip() for line in f]
    
    def __len__(self):
        return len(self.image_ids)
    
    def _parse_xml(self, xml_path: str):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        boxes = []
        labels = []
        
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name in self.class_to_idx:
                labels.append(self.class_to_idx[class_name])
                
                bbox = obj.find('bndbox')
                x1 = float(bbox.find('xmin').text)
                y1 = float(bbox.find('ymin').text)
                x2 = float(bbox.find('xmax').text)
                y2 = float(bbox.find('ymax').text)
                boxes.append([x1, y1, x2, y2])
        
        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)
    
    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        
        # 加载图片
        image_path = os.path.join(self.data_dir, 'JPEGImages', f'{image_id}.jpg')
        image = Image.open(image_path).convert("RGB")
        
        # 加载XML标注
        xml_path = os.path.join(self.data_dir, 'Annotations', f'{image_id}.xml')
        boxes, labels = self._parse_xml(xml_path)
        
        if self.transform:
            image = self.transform(image)
        
        return {'image': image, 'boxes': boxes, 'labels': labels}
    
    


if __name__ == "__main__":
    # 测试数据集加载器
    data_dir = "e:/Handwritten-Deep-Learning-Models/data/Pascal_VOC/VOC2012_train_val/VOC2012_train_val"
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = PascalVOCDataset(data_dir=data_dir, split='train', transform=transform)
    print(f"数据集大小: {len(dataset)}")
    
    # 测试第一个样本
    sample = dataset[0]
    print(f"图片形状: {sample['image'].shape}")
    print(f"标注框: {sample['boxes']}")
    print(f"类别: {sample['labels']}")