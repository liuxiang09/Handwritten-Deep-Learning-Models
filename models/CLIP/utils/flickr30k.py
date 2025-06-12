from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
from tqdm import tqdm
from transformers import CLIPProcessor
import torch
from torch.nn.utils.rnn import pad_sequence 

def collate_fn(batch):
    pixel_values = [item['pixel_values'] for item in batch] # N * [3, H, W]
    input_ids = [item['input_ids'] for item in batch] # N * [5, max_len]
    attention_mask = [item['attention_mask'] for item in batch] # N * [5, max_len]
    
    # 将列表转换为张量
    pixel_values = torch.stack(pixel_values)

    try:
        input_ids = torch.stack(input_ids)
        attention_mask = torch.stack(attention_mask)
    except Exception as e:
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    # 对input_ids和attention_mask进行展平处理
    input_ids = input_ids.view(input_ids.shape[0] * 5, -1)
    attention_mask = attention_mask.view(attention_mask.shape[0] * 5, -1)
    
    return {
        'pixel_values': pixel_values,     # [N, 3, H, W]
        'input_ids': input_ids,          # [N * 5, max_len]
        'attention_mask': attention_mask, # [N * 5, max_len]
    }



from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import csv
from tqdm import tqdm
from transformers import CLIPProcessor

# ... existing code ...
class Flickr30kDataset(Dataset):
    """
    数据集类，用于处理Flickr30k数据集
    """
    def __init__(self, image_dir: str, text_path: str, processor: CLIPProcessor, max_len: int):
        self.image_dir = image_dir
        self.text_path = text_path
        self.processor = processor
        self.max_len = max_len
        self.image_list, self.text_list = self._load_data()

    def _load_data(self):
        image_list = []
        text_list = []  # 每个元素是包含5个文本描述的列表
        image_paths_exist = set(os.listdir(self.image_dir))

        with open(self.text_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='|')
            # Skip header
            try:
                next(reader)
            except StopIteration:
                return [], [] # empty file

            # 创建一个字典来存储每个图片的所有评论
            image_comments = {}
            
            for i, row in enumerate(reader):
                # <3 说明该行不完整
                if len(row) < 3:
                    continue
                
                image_name = row[0].strip()
                comment = row[2].strip()
                
                # 如果是新图片,初始化评论列表
                if image_name not in image_comments:
                    image_comments[image_name] = []
                
                # 添加评论到对应图片
                image_comments[image_name].append(comment)
            
            # 处理收集到的所有图片和评论
            for image_name, comments in image_comments.items():
                # 只添加有5条评论且图片文件存在的数据
                if len(comments) == 5 and image_name in image_paths_exist:
                    image_list.append(image_name)
                    text_list.append(comments)
        
        return image_list, text_list

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.image_list[index])
        image = Image.open(image_path).convert("RGB") # [H, W, C]
        text = self.text_list[index]

        # 使用transformers库的processor处理图像和文本
        inputs = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_len,
            truncation=True
        )

        # processor 的输出已经包含了模型所需的全部输入,
        # __getitem__ 只需返回这些预处理好的张量即可。
        # .squeeze(0) 是因为 processor 会为单个样本添加一个批次维度,
        # 而 DataLoader 会自动处理批次, 所以这里需要移除。
        return {
            "pixel_values": inputs.pixel_values.squeeze(0),      # [3, H, W]
            "input_ids": inputs.input_ids.squeeze(0),            # [5, max_len]
            "attention_mask": inputs.attention_mask.squeeze(0),  # [5, max_len]
        }