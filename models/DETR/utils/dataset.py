import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict
import torchvision.transforms as transforms
from PIL import Image
from torch import Tensor
import os
import xml.etree.ElementTree as ET
import torch.nn.functional as F


def collate_fn(batch):
    images = [item['image'] for item in batch]
    boxes = [item['boxes'] for item in batch]
    labels = [item['labels'] for item in batch]

    # 获取该批次中最大的图像尺寸
    max_height = max(image.shape[1] for image in images)
    max_width = max(image.shape[2] for image in images)

    # 对图像进行padding
    padded_images = []
    masks = []
    targets = []

    for image in images:
        c, h, w = image.shape

        padding = (0, max_width - image.shape[2], 0, max_height - image.shape[1])
        padded_image = F.pad(image, padding, value=0)
        padded_images.append(padded_image)

        # 创建mask，True代表填充区域，False代表有效区域
        mask = torch.zeros((max_height, max_width), dtype=torch.bool)
        mask[h:, :] = True  # 填充区域
        mask[:, w:] = True  # 填充区域
        masks.append(mask)

    images_tensor = torch.stack(padded_images)
    masks_tensor = torch.stack(masks)
    
    # 处理目标框和标签
    for i, (box, label) in enumerate(zip(boxes, labels)):
        # 获取对应图像的原始尺寸
        orig_h, orig_w = images[i].shape[1], images[i].shape[2] # [C, H, W]
        
        # 归一化box坐标 (x1, y1, x2, y2) -> (0, 1)
        normalized_boxes = box.clone()
        normalized_boxes[:, [0, 2]] /= orig_w  # 归一化x坐标
        normalized_boxes[:, [1, 3]] /= orig_h  # 归一化y坐标
        
        target = {
            'boxes': normalized_boxes,
            'labels': label
        }
        targets.append(target)

    return {
        'images': images_tensor, # [B, C, H, W]
        'masks': masks_tensor, # [B, H, W]
        'targets': targets # List -> 每个元素是一个字典，包含 'boxes' 和 'labels'
    }

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
                
                # 转换为cx, cy, w, h格式
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                w = x2 - x1
                h = y2 - y1
                boxes.append([cx, cy, w, h])
        
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
    data_dir = "./data/Pascal_VOC/VOC2012_train_val/VOC2012_train_val"
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = PascalVOCDataset(data_dir=data_dir, split='train', transform=transform)
    print(f"数据集大小: {len(dataset)}")
    
    for i in range(4):  # 打印前5个样本的信息
        sample = dataset[i]
        print(f"\n第{i+1}个样本:")
        print(f"图片形状: {sample['image'].shape}")
        print(f"标注框: {sample['boxes']}")
        print(f"类别: {sample['labels']}")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2
    )

    for idx, batch in enumerate(dataloader):
        if idx < 5:
            print(f"\n第{idx+1}个批次:")
            print(f"图片形状: {batch['images'].shape}")
            print(f"标注框: {batch['targets']}")
            print(f"掩码形状: {batch['masks'].shape}")
        else:
            break

    print("数据加载器测试完成。")