import torch
import unittest
import sys
import os
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.CLIP.model.vision_encoder import VisionEncoder

console = Console()

class TestVisionEncoder(unittest.TestCase):
    def setUp(self):
        """初始化测试环境"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        console.print(f"[bold green]测试设备: {self.device}[/bold green]")
        
        # 设置模型参数
        self.image_size = 224
        self.patch_size = 16
        self.embed_dim = 512
        self.n_head = 8
        self.n_layer = 6
        
        # 初始化模型
        self.model = VisionEncoder(
            image_size=self.image_size,
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            n_head=self.n_head,
            n_layer=self.n_layer
        ).to(self.device)
        
        # 创建测试输入
        self.batch_size = 4
        self.test_input = torch.randn(self.batch_size, 3, self.image_size, self.image_size).to(self.device)

    def test_output_shape(self):
        """测试输出形状是否正确"""
        with torch.no_grad():
            output = self.model(self.test_input)
        
        expected_shape = torch.Size([self.batch_size, self.embed_dim])
        self.assertEqual(output.shape, expected_shape)
        
        # 打印详细信息
        shape_table = Table(title="形状测试结果")
        shape_table.add_column("测试项", style="cyan")
        shape_table.add_column("预期形状", style="green")
        shape_table.add_column("实际形状", style="green")
        shape_table.add_column("状态", style="bold")
        
        status = "[bold green]通过[/bold green]" if output.shape == expected_shape else "[bold red]失败[/bold red]"
        shape_table.add_row("输出张量", str(expected_shape), str(output.shape), status)
        console.print(shape_table)

    def test_cls_token(self):
        """测试CLS令牌是否正确添加和提取"""
        # 保存中间输出的钩子
        cls_token_output = None
        
        def hook_fn(module, input, output):
            nonlocal cls_token_output
            # 提取CLS令牌
            cls_token_output = output[:, 0, :]
            
        # 注册钩子
        hook = self.model.transformer_encoder.register_forward_hook(hook_fn)
        
        with torch.no_grad():
            output = self.model(self.test_input)
            
        # 移除钩子
        hook.remove()
        
        # 验证CLS令牌是否被正确处理
        self.assertTrue(torch.allclose(output, self.model.ln_final(cls_token_output)))
        
        console.print(Panel("[bold green]CLS令牌测试通过: 确认模型正确提取并处理了CLS令牌[/bold green]"))

    def test_normalization(self):
        """测试输出特征的范数"""
        with torch.no_grad():
            output = self.model(self.test_input)
            
        # 计算每个特征向量的L2范数
        norms = torch.norm(output, dim=1)
        
        # 打印范数统计
        norm_table = Table(title="特征向量范数统计")
        norm_table.add_column("统计量", style="cyan")
        norm_table.add_column("值", style="green")
        
        norm_table.add_row("最小值", f"{norms.min().item():.4f}")
        norm_table.add_row("最大值", f"{norms.max().item():.4f}")
        norm_table.add_row("平均值", f"{norms.mean().item():.4f}")
        norm_table.add_row("标准差", f"{norms.std().item():.4f}")
        
        console.print(norm_table)

    def test_model_parameters(self):
        """测试模型参数统计"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        param_table = Table(title="模型参数统计")
        param_table.add_column("参数类型", style="cyan")
        param_table.add_column("数量", style="green")
        
        param_table.add_row("总参数量", f"{total_params:,}")
        param_table.add_row("可训练参数量", f"{trainable_params:,}")
        param_table.add_row("不可训练参数量", f"{total_params - trainable_params:,}")
        
        console.print(param_table)
        
        # 打印模型结构
        console.print("[bold]模型结构:[/bold]")
        console.print(self.model)

if __name__ == "__main__":
    unittest.main() 