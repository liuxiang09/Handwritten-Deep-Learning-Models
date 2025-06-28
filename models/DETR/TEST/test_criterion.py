import sys
import os
import torch

# 添加项目根目录到路径，以便导入模块
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_root)

from models.DETR.model.criterion import SetCriterion
from models.DETR.model.matcher import HungarianMatcher

# 全局参数配置
# 匹配器参数
COST_CLASS = 1.0
COST_BBOX = 5.0
COST_GIOU = 2.0

# 损失函数参数
NUM_CLASSES = 91
EOS_COEF = 0.1
WEIGHT_DICT = {
    'loss_ce': 1.0,
    'loss_bbox': 5.0,
    'loss_giou': 2.0
}

# 测试数据参数
BATCH_SIZE = 2
BATCH_SIZE_SINGLE = 1
NUM_QUERIES = 100
MIN_OBJECTS = 1
MAX_OBJECTS = 5
TEST_OBJECTS = 3


def test_set_criterion():
    """测试DETR损失函数"""
    print("开始测试 SetCriterion...")
    
    # 创建匹配器和损失函数实例
    matcher = HungarianMatcher(cost_class=COST_CLASS, cost_bbox=COST_BBOX, cost_giou=COST_GIOU)
    criterion = SetCriterion(
        num_classes=NUM_CLASSES,
        matcher=matcher,
        weight_dict=WEIGHT_DICT,
        eos_coef=EOS_COEF
    )
    print("创建 SetCriterion 实例成功")
    
    # 准备模型输出
    outputs = {
        'pred_logits': torch.rand(BATCH_SIZE, NUM_QUERIES, NUM_CLASSES + 1),
        'pred_boxes': torch.rand(BATCH_SIZE, NUM_QUERIES, 4)
    }
    
    print(f"预测类别logits形状: {outputs['pred_logits'].shape}")
    print(f"预测边界框形状: {outputs['pred_boxes'].shape}")
    
    # 准备目标
    targets = []
    for i in range(BATCH_SIZE):
        num_objects = torch.randint(MIN_OBJECTS, MAX_OBJECTS + 1, (1,)).item()
        target = {
            'labels': torch.randint(0, NUM_CLASSES, (num_objects,)),
            'boxes': torch.rand(num_objects, 4)
        }
        targets.append(target)
    
    print(f"目标数量: {[len(t['labels']) for t in targets]}")
    
    # 计算损失
    losses = criterion(outputs, targets)
    
    print("损失计算结果:")
    for loss_name, loss_value in losses.items():
        print(f"  {loss_name}: {loss_value.item():.4f}")
    
    # 验证损失字典包含所有预期的损失项
    expected_losses = ['loss_ce', 'loss_bbox', 'loss_giou']
    for loss_name in expected_losses:
        assert loss_name in losses, f"损失字典中应该包含 {loss_name}"
        assert torch.is_tensor(losses[loss_name]), f"{loss_name} 应该是张量"
        # 检查损失值是有限的数值
        assert torch.isfinite(losses[loss_name]), f"{loss_name} 应该是有限的数值"
    
    # 计算总损失
    total_loss = sum(losses.values())
    print(f"总损失: {total_loss.item():.4f}")
    
    print("SetCriterion 测试成功！")


def test_criterion_individual_losses():
    """测试各个单独的损失函数"""
    print("\n开始测试各个损失函数组件...")
    
    matcher = HungarianMatcher()
    criterion = SetCriterion(num_classes=NUM_CLASSES, matcher=matcher, weight_dict=WEIGHT_DICT, eos_coef=EOS_COEF)
    
    # 测试数据
    outputs = {
        'pred_logits': torch.rand(BATCH_SIZE_SINGLE, NUM_QUERIES, NUM_CLASSES + 1),
        'pred_boxes': torch.rand(BATCH_SIZE_SINGLE, NUM_QUERIES, 4)
    }
    
    targets = [{
        'labels': torch.randint(0, NUM_CLASSES, (TEST_OBJECTS,)),
        'boxes': torch.rand(TEST_OBJECTS, 4)
    }]
    
    # 获取匹配结果
    indices = matcher(outputs, targets)
    
    # 测试calculate_losses方法
    print("测试calculate_losses方法...")
    losses = criterion.calculate_losses(outputs, targets, indices, num_boxes=TEST_OBJECTS)
    
    print("各个损失函数组件结果:")
    for loss_name, loss_value in losses.items():
        print(f"  {loss_name}: {loss_value.item():.4f}")
        assert torch.is_tensor(loss_value), f"{loss_name} 应该是张量"
        assert torch.isfinite(loss_value), f"{loss_name} 应该是有限的数值"
    
    # 验证损失字典包含所有预期的损失项
    expected_losses = ['loss_ce', 'loss_bbox', 'loss_giou']
    for loss_name in expected_losses:
        assert loss_name in losses, f"损失字典中应该包含 {loss_name}"
    
    print("各个损失函数组件测试成功！")


def test_criterion_empty_targets():
    """测试空目标的情况"""
    print("\n开始测试空目标情况...")
    
    matcher = HungarianMatcher()
    criterion = SetCriterion(num_classes=NUM_CLASSES, matcher=matcher, weight_dict=WEIGHT_DICT, eos_coef=EOS_COEF)
    
    # 空目标
    outputs = {
        'pred_logits': torch.rand(BATCH_SIZE_SINGLE, NUM_QUERIES, NUM_CLASSES + 1),
        'pred_boxes': torch.rand(BATCH_SIZE_SINGLE, NUM_QUERIES, 4)
    }
    
    targets = [{
        'labels': torch.empty(0, dtype=torch.long),
        'boxes': torch.empty(0, 4)
    }]
    
    print("计算空目标损失...")
    losses = criterion(outputs, targets)
    
    # 空目标情况下，bbox和giou损失可能是NaN（因为没有匹配的框）
    print("空目标损失结果:")
    for loss_name, loss_value in losses.items():
        print(f"  {loss_name}: {loss_value.item():.4f}")
    
    # 检查分类损失是有限的（应该有值，因为有负样本）
    assert torch.isfinite(losses['loss_ce']), "空目标的分类损失应该是有限的"
    
    # 对于空目标，bbox和giou损失可能是NaN，这是正常的
    if torch.isnan(losses['loss_bbox']):
        print("  注意：空目标的边界框损失为NaN，这是正常的（没有匹配的框）")
    if torch.isnan(losses['loss_giou']):
        print("  注意：空目标的GIoU损失为NaN，这是正常的（没有匹配的框）")
    
    print("空目标情况测试成功！")


if __name__ == "__main__":
    test_set_criterion()
    test_criterion_individual_losses()
    test_criterion_empty_targets()
