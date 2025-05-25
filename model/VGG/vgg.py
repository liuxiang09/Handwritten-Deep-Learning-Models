import torch
import torch.nn as nn

# 定义不同VGG变体的配置
# M 表示 Max Pooling Layer
# 数字表示该层卷积的输出通道数
_MODLE = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    """
    基于PyTorch nn.Module实现的VGG网络
    """
    def __init__(self, vgg_name: str, num_classes: int = 1000, init_weights: bool = True):
        """
        Args:
            vgg_name (str): VGG变体的名称，如 'VGG16'
            num_classes (int): 分类任务的类别数量
            init_weights (bool): 是否初始化网络权重
        """
        super(VGG, self).__init__()
        self.cfg = _MODLE[vgg_name] # 获取指定VGG变体的配置

        # 构建卷积层部分 (features)
        # 输入通道默认为3 (RGB图像)
        self.features = self._make_features(self.cfg, in_channels=3)

        # 构建分类器部分 (classifier)
        # VGG的原始FC层是固定的，但第一个FC层的输入大小取决于最后一个卷积层的输出尺寸
        # 假设输入图片是224x224，经过5次2x2的MaxPooling后，空间尺寸变为 224 / (2^5) = 7x7
        # 最后一个卷积层的输出通道数是512
        # 所以展平后的特征数量是 512 * 7 * 7 = 25088
        # 如果你的输入图片尺寸不同或Pooling配置不同，这里的输入维度需要相应调整
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096), # 第一个全连接层
            nn.ReLU(True), # inplace=True 可以节省一点内存
            nn.Dropout(),  # 训练时使用Dropout
            nn.Linear(4096, 4096), # 第二个全连接层
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes), # 最后一个分类层
        )

        # 初始化权重
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            x (torch.Tensor): 输入张量 (图片数据)，形状通常为 (batch_size, channels, height, width)
        Returns:
            torch.Tensor: 输出张量 (分类得分)，形状通常为 (batch_size, num_classes)
        """
        # 通过卷积层部分
        x = self.features(x)

        # 展平特征图，准备输入全连接层
        # x.size(0) 是批次大小
        # -1 表示自动计算展平后的维度
        x = torch.flatten(x, 1) # 或者使用 x = x.view(x.size(0), -1)

        # 通过分类器部分
        x = self.classifier(x)

        return x

    def _make_features(self, cfg: list, in_channels: int = 3) -> nn.Sequential:
        """
        根据配置列表构建卷积层序列
        Args:
            cfg (list): VGG配置列表
            in_channels (int): 输入通道数
        Returns:
            nn.Sequential: 包含卷积、ReLU和Pooling层的Sequential模块
        """
        layers = []
        # current_in_channels = in_channels # 记录当前的输入通道数
        for v in cfg:
            if v == 'M':
                # Max Pooling 层，通常使用 2x2 的窗口和 2 的步长
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                # 卷积层 + ReLU 激活
                out_channels = v # v 是配置中的输出通道数
                conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1) # 3x3卷积，padding=1保持空间尺寸
                layers += [conv2d, nn.ReLU(True)]
                in_channels = out_channels # 更新下一层的输入通道数

        return nn.Sequential(*layers) # 使用 * 将列表解包作为Sequential的参数

    def _initialize_weights(self) -> None:
        """
        初始化网络权重
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 使用Kaiming正态分布初始化卷积层权重 (适用于ReLU激活)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0) # 偏置初始化为0
            elif isinstance(m, nn.Linear):
                # 使用正态分布初始化全连接层权重
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0) # 偏置初始化为0

if __name__ == "__main__":
    # 测试模型的创建
    model = VGG('VGG16', num_classes=10)  # 创建VGG16模型实例
    # 测试模型的前向传播
    x = torch.randn(1, 3, 224, 224)
    print("模型输入形状:", x.shape)
    output = model(x)
    print("模型输出形状:", output.shape)  # 应该是 (1, 10)
    # 测试模型的参数数量
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("模型参数数量:", num_params)  # 打印模型的参数数量


