import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        """
        初始化前馈网络
        Args:
            d_model (int): 嵌入维度
            d_ff (int): 前馈网络的隐藏层维度
            dropout (float): Dropout概率
        """
        super(FeedForward, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff),  # 线性变换
            nn.ReLU(),  # 激活函数
            nn.Dropout(dropout),  # Dropout层
            nn.Linear(d_ff, d_model)  # 线性变换
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            x (torch.Tensor): 输入张量，形状为 [batch_size, seq_len, d_model]
        Returns:
            torch.Tensor: 前馈网络的输出，形状为 [batch_size, seq_len, d_model]
        """
        x = self.fc(x)  # 前馈网络
        return x