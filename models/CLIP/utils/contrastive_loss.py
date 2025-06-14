import torch
import torch.nn as nn
import torch.nn.functional as F

# 对比损失函数
# 由于一个图片有5个文本，所以无法采用常规的交叉熵损失函数
class StandardContrastiveLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image_features, text_features, logit_scale):
        # 归一化特征 (已经在 CLIP 模型的 forward 中完成)
        
        # 计算相似度矩阵
        logits = (image_features @ text_features.T) * logit_scale.exp()

        # 创建标签 (对角线为正样本)
        labels = torch.arange(len(logits)).to(logits.device)

        # 计算图像到文本的损失 (行是图像，列是文本)
        loss_i = F.cross_entropy(logits, labels)
        
        # 计算文本到图像的损失 (转置 logits，行是文本，列是图像)
        loss_t = F.cross_entropy(logits.T, labels)
        
        # 返回平均损失
        return (loss_i + loss_t) / 2

class MultiTextContrastiveLoss(nn.Module):
    """
    能够正确处理一个图像对应多个文本描述的对比损失函数。
    """
    def __init__(self):
        super().__init__()

    def forward(self, image_features, text_features, logit_scale):
        """
        Args:
            image_features: shape [N, D], N 是批次中的图片数量。
            text_features: shape [5*N, D], 对应 N 张图片的 5*N 个文本描述。
            logit_scale: 可学习的温度参数。
        """
        device = image_features.device
        num_images = image_features.shape[0]
        num_texts = text_features.shape[0]

        # 验证输入形状是否匹配
        if num_texts % num_images != 0 or num_texts // num_images != 5:
            raise ValueError("文本特征数量必须是图片特征数量的5倍。")

        # 计算相似度矩阵
        # logits_per_image shape: [N, 5*N]
        logits_per_image = (logit_scale.exp() * image_features @ text_features.T)
        # logits_per_text shape: [5*N, N]
        logits_per_text = logits_per_image.T

        # --- 正确的损失计算 ---

        # 1. 计算 loss_t (文找图): 这是一个标准的多类别分类问题
        # 每个文本都有一个正确的图片目标。
        # 创建标签 [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, ..., N-1, ...]
        text_labels = torch.arange(num_images, device=device).repeat_interleave(5)
        loss_t = F.cross_entropy(logits_per_text, text_labels)

        # 2. 计算 loss_i (图找文): 这是一个多标签分类问题
        # 每张图片有5个正确的文本目标。标准的 cross_entropy 不适用。
        # 我们需要创建一个 "多热" (multi-hot) 的标签矩阵。
        # ground_truth shape: [N, 5*N]
        ground_truth = torch.zeros(logits_per_image.shape, dtype=torch.float, device=device)
        for i in range(num_images):
            # 将图片i对应的5个文本位置标记为1
            start_idx = i * 5
            end_idx = start_idx + 5
            ground_truth[i, start_idx:end_idx] = 1.0
        
        # 使用二元交叉熵损失 (Binary Cross Entropy)
        # 它将每个输出logit视为一个独立的二元分类（是/不是 正确的匹配）
        loss_i = F.binary_cross_entropy_with_logits(logits_per_image, ground_truth)

        # 返回两个方向损失的平均值
        return (loss_i + loss_t) / 2