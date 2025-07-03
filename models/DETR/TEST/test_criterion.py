import torch
import random
from models.DETR.model.criterion import SetCriterion
from models.DETR.model.matcher import HungarianMatcher

# 全局参数配置

# 损失函数参数
NUM_CLASSES = 90
EOS_COEF = 0.1 # "无目标"类别的权重

# 分类损失一般为1，边界框损失通常为5-10，GIoU损失为1-2
WEIGHT_DICT = {
    'loss_ce': 1.0,
    'loss_bbox': 5.0,
    'loss_giou': 2.0
}

# 测试数据参数
BATCH_SIZE = 8
NUM_QUERIES = 100


def test_set_criterion():
    """测试DETR损失函数"""
    print("开始测试 SetCriterion...")
    
    # 创建匹配器和损失函数实例
    matcher = HungarianMatcher()
    criterion = SetCriterion(
        num_classes=NUM_CLASSES,
        matcher=matcher,
        weight_dict=WEIGHT_DICT,
        eos_coef=EOS_COEF
    )
    print("创建 SetCriterion 实例成功")
    
    # 准备模型输出
    outputs = {
        'pred_labels': torch.rand(BATCH_SIZE, NUM_QUERIES, NUM_CLASSES + 1),
        'pred_boxes': torch.rand(BATCH_SIZE, NUM_QUERIES, 4)
    }

    print(f"预测类别logits形状: {outputs['pred_labels'].shape}") # [B, 100, 91]
    print(f"预测边界框形状: {outputs['pred_boxes'].shape}") # [B, 100, 4]
    
    # 准备目标
    targets = []
    for i in range(BATCH_SIZE):
        num_objects = random.randint(1, 10) # 随机生成目标数量
        target = {
            'labels': torch.randint(0, NUM_CLASSES, (num_objects,)),
            'boxes': torch.rand(num_objects, 4)
        }
        targets.append(target)
    
    print(f"Target检测框数量列表: {[len(t['labels']) for t in targets]}")
    
    # 计算损失
    losses = criterion(outputs, targets)
    
    print("损失计算结果:")
    # 先加权损失并赋值回 losses 字典
    total_loss = sum(losses.values()) # 计算总损失

    print(f"{losses['loss_ce'].item():.4f} (加权分类损失)")
    print(f"{losses['loss_bbox'].item():.4f} (加权边界框损失)")
    print(f"{losses['loss_giou'].item():.4f} (加权GIoU损失)")
    print(f"{total_loss.item():.4f} (总损失)")
    
    print("\nSetCriterion 测试成功！")

if __name__ == "__main__":
    test_set_criterion()
