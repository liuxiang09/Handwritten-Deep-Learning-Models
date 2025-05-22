from torch.utils.data import Dataset
from PIL import Image
import os

class MyData(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(root_dir, label_dir)
        self.img_path = os.listdir(self.path)


        super().__init__()



    def __getitem__(self, index):
        image_name = self.img_path[index]
        img_item_path = os.path.join(self.path, image_name)
        image = Image.open(img_item_path).convert('RGB')
        label = self.label_dir
        return image, label
    
    def __len__(self):
        return len(self.img_path)


    
data = MyData("data/ants_bees/train", 'ants_image')
print(len(data.img_path))
print(data[0][1])
data[123][0].show()