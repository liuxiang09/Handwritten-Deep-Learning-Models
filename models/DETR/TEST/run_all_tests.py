import sys
import os

# 添加项目根目录到路径，以便导入模块
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_root)

from test_backbone import test_backbone
from test_position_encoding import test_position_encoding  
from test_transformer import test_transformer
from test_matcher import test_hungarian_matcher, test_matcher_edge_cases
from test_criterion import test_set_criterion, test_criterion_individual_losses, test_criterion_empty_targets
from test_detr import test_mlp, test_detr


def run_all_tests():
    """运行所有DETR模块测试"""
    print("=" * 60)
    print("开始运行 DETR 所有模块测试")
    print("=" * 60)
    
    try:
        # 测试Backbone
        print("\n" + "=" * 40)
        print("1. 测试 Backbone 模块")
        print("=" * 40)
        test_backbone()
        
        # 测试位置编码
        print("\n" + "=" * 40)
        print("2. 测试位置编码模块")
        print("=" * 40)
        test_position_encoding()
        
        # 测试Transformer组件
        print("\n" + "=" * 40)
        print("3. 测试 Transformer 模块")
        print("=" * 40)
        test_transformer()
        
        # 测试匹配器
        print("\n" + "=" * 40)
        print("4. 测试匈牙利匹配器")
        print("=" * 40)
        test_hungarian_matcher()
        test_matcher_edge_cases()
        
        # 测试损失函数
        print("\n" + "=" * 40)
        print("5. 测试损失函数")
        print("=" * 40)
        test_set_criterion()
        test_criterion_individual_losses()
        test_criterion_empty_targets()
        
        # 测试DETR主模型
        print("\n" + "=" * 40)
        print("6. 测试 DETR 主模型")
        print("=" * 40)
        test_mlp()
        test_detr()
        
        print("\n" + "=" * 60)
        print("所有 DETR 模块测试成功完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n测试过程中出现错误: {str(e)}")
        print("请检查相关模块和依赖项")
        raise


if __name__ == "__main__":
    run_all_tests()
