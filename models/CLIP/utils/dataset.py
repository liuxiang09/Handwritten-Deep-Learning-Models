from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
from tqdm import tqdm


# def collate_fn(batch):
#     input_ids = [item['input_ids'] for item in batch]
#     attention_mask = [item['attention_mask'] for item in batch]
#     pixel_values = [item['pixel_values'] for item in batch]
#     images = [item['image'] for item in batch] # 收集原始图像
#     return 



class CustomCLIPDataset(Dataset):
    """
    自定义 CLIP 数据集。
    假设我们有一个图片文件夹，和一个文本文件，每行对应一个图片文件名和其描述，用空格分隔。
    例如：
    image1.jpg 红色
    image2.png 白色
    """
    def __init__(self, image_dir, text_data_path, processor, max_seq_length):
        self.image_dir = image_dir
        self.processor = processor
        self.max_seq_length = max_seq_length
        self.data = self._load_data(text_data_path)

    def _load_data(self, text_data_path):
        data = []
        with open(text_data_path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f, 1):
                parts = line.strip().split(' ')
                try:
                    assert len(parts) == 2
                    image_filename, text_caption = parts
                    image_path = os.path.join(self.image_dir, image_filename)
                    if os.path.exists(image_path): # 检查图片是否存在
                        data.append({'image_path': image_path, 'text_caption': text_caption})
                except AssertionError:
                    print(f"警告：文件 '{text_data_path}' 中第 {line_idx} 行的数据格式不正确。")
                except Exception as e: # 捕获其他可能的异常
                    print(f"处理文件 '{text_data_path}' 中第 {line_idx} 行时发生未知错误：{e}")
        return data # 一个字典列表

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item['image_path']
        text_caption = item['text_caption']

        # 加载图片
        image = Image.open(image_path).convert("RGB")

        # 处理图片和文本
        # processor 会处理图像大小调整、归一化等，并对文本进行分词和编码
        inputs = self.processor(
            text=text_caption,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_length
        )
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'pixel_values': inputs['pixel_values'].squeeze(),
        }