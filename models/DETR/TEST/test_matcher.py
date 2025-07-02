import torch
from models.DETR.model.matcher import HungarianMatcher

# 全局参数配置
# 匹配器参数
COST_CLASS = 1.0
COST_BBOX = 5.0
COST_GIOU = 2.0

# 测试数据参数
BATCH_SIZE = 4
NUM_QUERIES = 100
NUM_CLASSES = 20
MIN_OBJECTS = 1
MAX_OBJECTS = 5


def test_hungarian_matcher():
    """测试匈牙利匹配器"""
    print("🚀 开始测试 HungarianMatcher...")
    
    # 创建匹配器实例
    matcher = HungarianMatcher(cost_class=COST_CLASS, cost_bbox=COST_BBOX, cost_giou=COST_GIOU)
    print("✅ 创建 HungarianMatcher 实例成功")
    
    # 模拟预测输出
    outputs = {
        'pred_logits': torch.rand(BATCH_SIZE, NUM_QUERIES, NUM_CLASSES + 1), # [B, num_queries, num_classes + 1]
        'pred_boxes': torch.rand(BATCH_SIZE, NUM_QUERIES, 4)  # [B, num_queries, 4] (cx, cy, w, h normalized)
    }
    
    print(f"预测类别形状: {outputs['pred_logits'].shape}")
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

    print("\n🏷️ 真实标签:")
    for i, target in enumerate(targets):
        print(f"Batch {i}:")
        print(f"  标签: {target['labels']}")
        print(f"  边界框: {target['boxes']}")

    # 执行匹配
    indices = matcher(outputs, targets)
    
    print("\n🔗 匹配结果:")
    for i, (pred_indices, tgt_indices) in enumerate(indices):
        print(f"Batch {i}:")
        print(f"  预测索引: {pred_indices}")
        print(f"  目标索引: {tgt_indices}")
    
    print("🎉 HungarianMatcher 测试成功！")


if __name__ == "__main__":
    test_hungarian_matcher()
