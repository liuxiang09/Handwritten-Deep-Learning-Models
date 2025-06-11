import torch
import unittest
import sys
import os
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.vision_encoder import VisionEncoder

console = Console()

class TestVisionEncoder(unittest.TestCase):
    def setUp(self):
        """初始化测试环境"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        console.print(f"[bold green]测试设备: {self.device}[/bold green]")
        
        # 设置模型参数
        self.image_size = 224
        self.patch_size = 16
        self.embed_dim = 768
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
        self.batch_size = 64
        self.test_input = torch.randn(self.batch_size, 3, self.image_size, self.image_size).to(self.device)

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

    def test_tensor_shapes(self):
        """测试各部分输入输出tensor的shape（含中间层，更清晰）"""
        intermediate_shapes = {}

        # Hook for patch_embedding
        def patch_embedding_hook(module, input, output):
            intermediate_shapes["输入 (VisionEncoder)"] = tuple(input[0].shape)
            intermediate_shapes["Patch Embedding 输出 (N, D, H/P, W/P)"] = tuple(output.shape)

        # Hook for transformer_encoder
        def transformer_encoder_hook(module, input, output):
            # The input to transformer_encoder is already after CLS token and positional embedding
            intermediate_shapes["Transformer Encoder 输入 (N, 1+num_patches, D)"] = tuple(input[0].shape)
            intermediate_shapes["Transformer Encoder 输出 (N, 1+num_patches, D)"] = tuple(output.shape)

        # Hook for ln_final
        def ln_final_hook(module, input, output):
            # Input to ln_final is the CLS token output from transformer
            intermediate_shapes["最终 LayerNorm 输入 (CLS Token, N, D)"] = tuple(input[0].shape)
            intermediate_shapes["最终 LayerNorm 输出 (N, D)"] = tuple(output.shape)

        # Register hooks
        hooks = [
            self.model.patch_embedding.register_forward_hook(patch_embedding_hook),
            self.model.transformer_encoder.register_forward_hook(transformer_encoder_hook),
            self.model.ln_final.register_forward_hook(ln_final_hook),
        ]

        with torch.no_grad():
            final_output = self.model(self.test_input)

        # Print shape information
        shape_table = Table(title="模型主要模块输入输出Shape统计")
        shape_table.add_column("模块/阶段", style="cyan")
        shape_table.add_column("Shape", style="green")

        # Order them logically for better readability
        shape_table.add_row("输入 (VisionEncoder)", str(intermediate_shapes["输入 (VisionEncoder)"]))
        shape_table.add_row("Patch Embedding 输出", str(intermediate_shapes["Patch Embedding 输出 (N, D, H/P, W/P)"]))
        shape_table.add_row("Transformer Encoder 输入", str(intermediate_shapes["Transformer Encoder 输入 (N, 1+num_patches, D)"]))
        shape_table.add_row("Transformer Encoder 输出", str(intermediate_shapes["Transformer Encoder 输出 (N, 1+num_patches, D)"]))
        shape_table.add_row("最终 LayerNorm 输入 (CLS Token)", str(intermediate_shapes["最终 LayerNorm 输入 (CLS Token, N, D)"]))
        shape_table.add_row("最终 LayerNorm 输出", str(intermediate_shapes["最终 LayerNorm 输出 (N, D)"]))

        console.print(shape_table)

        # Assert final output shape
        expected_shape = torch.Size([self.batch_size, self.embed_dim])
        self.assertEqual(final_output.shape, expected_shape)

        # Remove all hooks
        for h in hooks:
            h.remove()

if __name__ == "__main__":
    unittest.main()