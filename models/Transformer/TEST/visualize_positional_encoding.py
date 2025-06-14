import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import math
from torch import nn
import matplotlib as mpl

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 添加父目录到路径中，以便导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 从Transformer模型中导入位置编码类
from model.positional_encoding import PositionalEncoding

def visualize_positional_encoding():
    """
    可视化位置编码
    """
    # 参数设置
    d_model = 128  # 嵌入维度
    max_len = 50  # 最大序列长度
    dropout = 0.1  # Dropout概率
    
    # 创建位置编码实例
    pos_encoder = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=max_len)
    
    # 创建一个全零输入张量进行测试
    x = torch.zeros(max_len, 1, d_model)
    
    # 获取位置编码 (不应用dropout，仅获取编码)
    with torch.no_grad():
        # 直接访问pe缓冲区，避免应用dropout
        pos_encoding = pos_encoder.pe[:max_len, 0, :].numpy()
    
    # 设置红蓝配色方案
    cmap_rb = mpl.colors.LinearSegmentedColormap.from_list('RdBu', ['#B2182B', '#F4A582', '#F7F7F7', '#92C5DE', '#2166AC'])
    
    # 创建一个大图，包含最重要的三个可视化
    plt.figure(figsize=(18, 6))
    
    # 1. 热力图展示完整的位置编码
    plt.subplot(1, 3, 1)
    im = plt.imshow(pos_encoding, cmap=cmap_rb, aspect='auto')
    plt.xlabel('维度', fontsize=12)
    plt.ylabel('位置', fontsize=12)
    plt.title('位置编码热力图', fontsize=14)
    plt.colorbar(im)
    
    # 2. 选择几个不同维度，展示其随位置变化的值
    plt.subplot(1, 3, 2)
    dimensions = [0, 16, 32, 64]
    colors = ['#B2182B', '#D6604D', '#4393C3', '#2166AC']  # 红蓝色系
    for i, dim in enumerate(dimensions):
        plt.plot(pos_encoding[:, dim], label=f'维度 {dim}', color=colors[i], linewidth=2)
    plt.legend(fontsize=10)
    plt.title('不同维度的位置编码变化', fontsize=14)
    plt.xlabel('位置', fontsize=12)
    plt.ylabel('编码值', fontsize=12)
    plt.grid(alpha=0.3)
    
    # 3. 展示相对位置关系 (余弦相似度矩阵)
    plt.subplot(1, 3, 3)
    similarity = np.zeros((max_len, max_len))
    for i in range(max_len):
        for j in range(max_len):
            # 计算余弦相似度
            similarity[i, j] = np.dot(pos_encoding[i], pos_encoding[j]) / (np.linalg.norm(pos_encoding[i]) * np.linalg.norm(pos_encoding[j]))
    im = plt.imshow(similarity, cmap=cmap_rb)
    plt.colorbar(im)
    plt.title('位置编码的相似度矩阵', fontsize=14)
    plt.xlabel('位置 j', fontsize=12)
    plt.ylabel('位置 i', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('models/Transformer/TEST/positional_encoding_visualization.png', dpi=300, bbox_inches='tight')
    print("位置编码可视化已保存为 'positional_encoding_visualization.png'")

if __name__ == "__main__":
    visualize_positional_encoding()
    print("可视化完成!") 