import sys
import os
import torch

# 添加项目根目录到路径，以便导入模块
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_root)

from models.DETR.model.matcher import HungarianMatcher

# 全局参数配置
# 匹配器参数
COST_CLASS = 1.0
COST_BBOX = 5.0
COST_GIOU = 2.0

# 测试数据参数
BATCH_SIZE = 2
NUM_QUERIES = 100
NUM_CLASSES = 91
MIN_OBJECTS = 1
MAX_OBJECTS = 5


def test_hungarian_matcher():
    """测试匈牙利匹配器"""
    print("开始测试 HungarianMatcher...")
    
    # 创建匹配器实例
    matcher = HungarianMatcher(cost_class=COST_CLASS, cost_bbox=COST_BBOX, cost_giou=COST_GIOU)
    print("创建 HungarianMatcher 实例成功")
    
    # 模拟预测输出
    outputs = {
        'pred_logits': torch.rand(BATCH_SIZE, NUM_QUERIES, NUM_CLASSES + 1),  # +1 for background
        'pred_boxes': torch.rand(BATCH_SIZE, NUM_QUERIES, 4)  # [cx, cy, w, h] normalized
    }
    
    print(f"预测类别logits形状: {outputs['pred_logits'].shape}")
    print(f"预测边界框形状: {outputs['pred_boxes'].shape}")
    
    # 准备目标（真实标签）
    targets = []
    for i in range(BATCH_SIZE):
        # 每个图像有不同数量的目标对象
        num_objects = torch.randint(MIN_OBJECTS, MAX_OBJECTS + 1, (1,)).item()
        target = {
            'labels': torch.randint(0, NUM_CLASSES, (num_objects,)),
            'boxes': torch.rand(num_objects, 4)  # [cx, cy, w, h] normalized
        }
        targets.append(target)
    
    print(f"目标数量: {[len(t['labels']) for t in targets]}")
    print(f"目标标签形状: {[t['labels'].shape for t in targets]}")
    print(f"目标边界框形状: {[t['boxes'].shape for t in targets]}")
    
    # 执行匹配
    indices = matcher(outputs, targets)
    
    print("匹配结果:")
    for i, (pred_indices, tgt_indices) in enumerate(indices):
        print(f"  批次 {i}: 预测索引 {pred_indices.shape}, 目标索引 {tgt_indices.shape}")
        print(f"    匹配的预测: {pred_indices[:5]}...")  # 显示前5个
        print(f"    匹配的目标: {tgt_indices[:5]}...")
        
        # 验证匹配结果的合理性
        assert len(pred_indices) == len(tgt_indices), "预测和目标索引数量应该相等"
        assert len(pred_indices) == len(targets[i]['labels']), "匹配数量应该等于目标数量"
        assert torch.all(pred_indices < NUM_QUERIES), "预测索引应该小于查询数量"
        assert torch.all(tgt_indices < len(targets[i]['labels'])), "目标索引应该小于目标数量"
    
    print("HungarianMatcher 测试成功！")


def test_matcher_edge_cases():
    """测试匹配器的边界情况"""
    print("\n开始测试匹配器边界情况...")
    
    matcher = HungarianMatcher(cost_class=COST_CLASS, cost_bbox=COST_BBOX, cost_giou=COST_GIOU)
    
    # 测试空目标的情况
    outputs = {
        'pred_logits': torch.rand(1, NUM_QUERIES, NUM_CLASSES + 1),
        'pred_boxes': torch.rand(1, NUM_QUERIES, 4)
    }
    
    # 空目标列表
    targets = [{
        'labels': torch.empty(0, dtype=torch.long),
        'boxes': torch.empty(0, 4)
    }]
    
    print("测试空目标情况...")
    indices = matcher(outputs, targets)
    
    for i, (pred_indices, tgt_indices) in enumerate(indices):
        print(f"  空目标批次 {i}: 预测索引数量 {len(pred_indices)}, 目标索引数量 {len(tgt_indices)}")
        assert len(pred_indices) == 0, "空目标应该没有匹配"
        assert len(tgt_indices) == 0, "空目标应该没有匹配"
    
    print("边界情况测试成功！")


if __name__ == "__main__":
    test_hungarian_matcher()
    test_matcher_edge_cases()
