import sys
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

# 添加项目根目录到路径，以便导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from DETR.model.position_encoding import PositionEmbeddingSine, PositionEmbeddingLearned
from DETR.utils.utils import NestedTensor

# 设置中文字体
def set_chinese_font():
    # 尝试设置中文字体
    try:
        # 尝试使用系统中的中文字体
        font_list = font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
        chinese_fonts = [f for f in font_list if 'simhei' in f.lower() or 
                                               'simsun' in f.lower() or 
                                               'microsoftyahei' in f.lower() or
                                               'wqy' in f.lower() or
                                               'noto' in f.lower() and 'cjk' in f.lower()]
        if chinese_fonts:
            plt.rcParams['font.family'] = font_manager.FontProperties(fname=chinese_fonts[0]).get_name()
        else:
            # 如果没有找到中文字体，使用默认字体并打印警告
            print("警告：未找到中文字体，中文可能显示为方块")
    except:
        print("设置中文字体失败，中文可能显示为方块")


def test_position_encoding():
    # 设置中文字体
    set_chinese_font()
    
    print("开始测试位置编码模型...")
    
    # 创建正弦位置编码实例
    pos_enc_sine = PositionEmbeddingSine(num_pos_feats=128, normalize=True)
    print("创建正弦位置编码实例成功")
    
    # 创建可学习位置编码实例
    pos_enc_learned = PositionEmbeddingLearned(num_pos_feats=128)
    print("创建可学习位置编码实例成功")
    
    # 准备输入数据 - 使用NestedTensor
    batch_size = 32
    height, width = 30, 40
    tensors = torch.rand(batch_size, 3, height, width)
    # 创建掩码，假设部分区域为填充区域
    masks = torch.zeros(batch_size, height, width, dtype=torch.bool)
    # 设置一些区域为填充区域（True）
    masks[:, :5, :10] = True
    nested_tensor = NestedTensor(tensors=tensors, mask=masks)
    print(f"输入NestedTensor - tensors形状: {nested_tensor.tensors.shape}, mask形状: {nested_tensor.mask.shape}")
    
    # 测试正弦位置编码
    pos_sine = pos_enc_sine(nested_tensor)
    print(f"正弦位置编码输出形状: {pos_sine.shape}")
    
    # 测试可学习位置编码
    pos_learned = pos_enc_learned(nested_tensor.mask)
    print(f"可学习位置编码输出形状: {pos_learned.shape}")
    
    # 可视化位置编码
    visualize_position_encoding(pos_sine, "sine")
    visualize_position_encoding(pos_learned, "learned")
    
    print("\n位置编码测试完成")


def visualize_position_encoding(pos_encoding, encoding_type):
    """可视化位置编码"""
    # 取第一个批次的位置编码
    pos = pos_encoding[0].detach().cpu().numpy()
    
    # 创建保存目录
    save_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 可视化前16个通道
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        if i < min(16, pos.shape[0]):
            im = ax.imshow(pos[i], cmap='viridis')
            ax.set_title(f"通道 {i}")
            ax.axis('off')
            fig.colorbar(im, ax=ax)
    
    plt.suptitle(f"{encoding_type.capitalize()} 位置编码可视化", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{encoding_type}_position_encoding.png"))
    print(f"位置编码可视化已保存到: {os.path.join(save_dir, f'{encoding_type}_position_encoding.png')}")
    
    # 可视化位置编码的空间分布
    # 选择两个通道进行可视化
    channel1, channel2 = 0, 1
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(pos[channel1], cmap='viridis')
    plt.title(f"通道 {channel1} 的空间分布")
    plt.colorbar()
    
    plt.subplot(1, 2, 2)
    plt.imshow(pos[channel2], cmap='viridis')
    plt.title(f"通道 {channel2} 的空间分布")
    plt.colorbar()
    
    plt.suptitle(f"{encoding_type.capitalize()} 位置编码空间分布", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{encoding_type}_position_encoding_spatial.png"))
    print(f"位置编码空间分布可视化已保存到: {os.path.join(save_dir, f'{encoding_type}_position_encoding_spatial.png')}")


if __name__ == "__main__":
    test_position_encoding()
