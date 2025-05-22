from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter


image_path = "data/ants_bees/train/ants_image/swiss-army-ant.jpg"
image_PIL = Image.open(image_path)
trans = transforms.Compose([
    transforms.RandomCrop(512, pad_if_needed=True),  # 调整大小
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize((0.5,), (0.5,)),  # 标准化
    transforms.ToPILImage()  # 转换回PIL图像
])
image = trans(image_PIL)
image.show()
# 使用TensorBoard可视化
# writer = SummaryWriter('logs')
# writer.add_image("test", image)
# writer.close()