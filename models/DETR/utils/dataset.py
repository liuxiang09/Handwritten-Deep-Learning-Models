import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict
import torchvision.transforms as transforms
from PIL import Image
from torch import Tensor
import os
import xml.etree.ElementTree as ET
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches


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

    for i, image in enumerate(images):
        c, h, w = image.shape

        padding = (0, max_width - image.shape[2], 0, max_height - image.shape[1])
        padded_image = F.pad(image, padding, value=0)
        padded_images.append(padded_image)

        # 创建mask，True代表填充区域，False代表有效区域
        mask = torch.zeros((max_height, max_width), dtype=torch.bool)
        mask[h:, :] = True  # 填充区域
        mask[:, w:] = True  # 填充区域
        masks.append(mask)

        if boxes[i].shape[0] > 0:
            cx = (boxes[i][:, 0] + boxes[i][:, 2]) / 2
            cy = (boxes[i][:, 1] + boxes[i][:, 3]) / 2
            box_w = boxes[i][:, 2] - boxes[i][:, 0]
            box_h = boxes[i][:, 3] - boxes[i][:, 1]
            cx = cx / w  # 归一化到[0, 1]
            cy = cy / h  # 归一化到[0, 1]
            box_w = box_w / w  # 归一化到[0, 1]
            box_h = box_h / h  # 归一化到[0, 1]
            boxes[i][:, 0] = cx
            boxes[i][:, 1] = cy
            boxes[i][:, 2] = box_w
            boxes[i][:, 3] = box_h

    padded_images = torch.stack(padded_images)
    masks = torch.stack(masks)
    
    # 处理目标框和标签
    for i, (box, label) in enumerate(zip(boxes, labels)):
        # 直接使用原始尺寸的边界框，不进行归一化
        target = {
            'boxes': box,
            'labels': label
        }
        targets.append(target)

    return {
        'images': padded_images, # [B, C, H, W]
        'masks': masks, # [B, H, W]
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
    data_dir = "./data/Pascal_VOC/VOC2012_train_val/VOC2012_train_val"
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.RandomHorizontalFlip(0.5),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
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
        shuffle=True,
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
    
    # 可视化部分：检查padding和掩码
    def visualize_padding_and_mask(batch, sample_idx=0):
        """
        可视化一个批次中的图像、padding和掩码
        
        Args:
            batch: 数据批次
            sample_idx: 要可视化的样本索引
        """
        # 获取图像、掩码和目标框
        image = batch['images'][sample_idx].cpu()
        mask = batch['masks'][sample_idx].cpu()
        targets = batch['targets'][sample_idx]
        boxes = targets['boxes'].cpu()
        
        # 获取原始图像尺寸（非padding部分）
        # 找到mask中为False的最大行和列索引
        valid_mask = ~mask  # 取反，现在False是padding，True是有效区域
        valid_rows = torch.any(valid_mask, dim=1)
        valid_cols = torch.any(valid_mask, dim=0)
        orig_height = valid_rows.sum().item()
        orig_width = valid_cols.sum().item()
        
        # 反归一化图像（根据之前使用的normalize参数）
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = image * std + mean
        
        # 转换为numpy进行可视化
        img_np = image.permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)  # 裁剪到[0,1]范围
        
        # 创建图形
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 1. 显示原始图像（带填充）
        axes[0].imshow(img_np)
        axes[0].set_title('Padded Image')
        axes[0].axis('off')
        
        # 2. 显示掩码（True=黑色(padding)，False=灰色(有效区域)）
        mask_display = np.zeros_like(mask.numpy(), dtype=np.float32)
        mask_display[mask.numpy()] = 0.0  # 黑色表示padding区域(True)
        mask_display[~mask.numpy()] = 0.7  # 灰色表示有效区域(False)
        axes[1].imshow(mask_display, cmap='gray')
        axes[1].set_title('Mask (Black=Ignore/Padding)')
        axes[1].axis('off')
        
        # 3. 显示图像并叠加边界框
        axes[2].imshow(img_np)
        
        # 绘制边界框，处理归一化的(cx,cy,w,h)格式
        for box in boxes:
            # 获取归一化的中心点和宽高
            cx, cy, w, h = box
            
            # 将归一化坐标转换回像素坐标，使用原始图像尺寸
            cx_pixel = cx * orig_width
            cy_pixel = cy * orig_height
            w_pixel = w * orig_width
            h_pixel = h * orig_height
            
            # 计算左上角坐标
            x1 = cx_pixel - w_pixel/2
            y1 = cy_pixel - h_pixel/2
            
            # 创建Rectangle patch
            rect = patches.Rectangle((x1, y1), w_pixel, h_pixel, linewidth=2, edgecolor='r', facecolor='none')
            axes[2].add_patch(rect)
        
        axes[2].set_title('Image with Bounding Boxes')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    # 获取一个批次
    batch_iter = iter(dataloader)
    batch = next(batch_iter)
    
    # 可视化第一个批次中的前两个样本
    for i in range(len(batch['images'])):
        visualize_padding_and_mask(batch, sample_idx=i)
    