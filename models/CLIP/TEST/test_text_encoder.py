import torch
import unittest
import sys
import os
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.text_encoder import TextEncoder

console = Console()

class TestTextEncoder(unittest.TestCase):
    def setUp(self):
        """初始化测试环境"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        console.print(f"[bold green]测试设备: {self.device}[/bold green]")
        
        # 设置模型参数 (示例值，请根据实际情况调整)
        self.vocab_size = 49408  # CLIP 词汇表大小 (示例)
        self.embed_dim = 512
        self.max_length = 77  # CLIP 文本最大序列长度 (示例)
        self.n_head = 8
        self.n_layer = 6
        
        # 初始化模型
        self.model = TextEncoder(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            max_length=self.max_length,
            n_head=self.n_head,
            n_layer=self.n_layer
        ).to(self.device)
        
        # 创建测试输入 (模拟 token id)
        self.batch_size = 64
        # 文本输入通常是整数 token ID
        # 确保 token ID 在 vocab_size 范围内，并包含一个 EOS token
        # 这里模拟一个简单的输入，所有 token ID 都小于 vocab_size
        # 并且最后一个 token 是 EOS (假设 EOS ID 为 vocab_size - 1)
        self.test_input = torch.randint(0, self.vocab_size - 1, (self.batch_size, self.max_length - 1)).to(self.device)
        # 假设 EOS token 的ID是最大的，添加 EOS token
        eos_token_id = self.vocab_size - 1
        eos_tokens = torch.full((self.batch_size, 1), eos_token_id, dtype=torch.long).to(self.device)
        self.test_input = torch.cat([self.test_input, eos_tokens], dim=1)
        

    def test_model_parameters(self):
        """测试模型参数统计"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        param_table = Table(title="文本编码器模型参数统计")
        param_table.add_column("参数类型", style="cyan")
        param_table.add_column("数量", style="green")
        
        param_table.add_row("总参数量", f"{total_params:,}")
        param_table.add_row("可训练参数量", f"{trainable_params:,}")
        param_table.add_row("不可训练参数量", f"{total_params - trainable_params:,}")
        
        console.print(param_table)

    def test_tensor_shapes(self):
        """测试文本编码器各部分输入输出tensor的shape（含中间层，更清晰）"""
        intermediate_shapes = {}

        # Hook for token_embedding (captures input to TextEncoder and output of embedding)
        def token_embedding_hook(module, input, output):
            intermediate_shapes["输入 (TextEncoder)"] = tuple(input[0].shape)
            intermediate_shapes["Token Embedding 输出 (N, L, D)"] = tuple(output.shape)

        # Hook for transformer_encoder
        def transformer_encoder_hook(module, input, output):
            intermediate_shapes["Transformer Encoder 输入 (N, L, D)"] = tuple(input[0].shape)
            intermediate_shapes["Transformer Encoder 输出 (N, L, D)"] = tuple(output.shape)

        # Hook for ln_final
        def ln_final_hook(module, input, output):
            # Input to ln_final is the EOT token output from transformer
            intermediate_shapes["最终 LayerNorm 输入 (EOT Token, N, D)"] = tuple(input[0].shape)
            intermediate_shapes["最终 LayerNorm 输出 (N, D)"] = tuple(output.shape)

        # Register hooks
        hooks = [
            self.model.token_embedding.register_forward_hook(token_embedding_hook),
            self.model.transformer_encoder.register_forward_hook(transformer_encoder_hook),
            self.model.ln_final.register_forward_hook(ln_final_hook),
        ]

        with torch.no_grad():
            final_output = self.model(self.test_input)

        # Print shape information
        shape_table = Table(title="文本编码器主要模块输入输出Shape统计")
        shape_table.add_column("模块/阶段", style="cyan")
        shape_table.add_column("Shape", style="green")

        # Order them logically for better readability
        shape_table.add_row("输入 (TextEncoder)", str(intermediate_shapes["输入 (TextEncoder)"]))
        shape_table.add_row("Token Embedding 输出", str(intermediate_shapes["Token Embedding 输出 (N, L, D)"]))
        shape_table.add_row("Transformer Encoder 输入", str(intermediate_shapes["Transformer Encoder 输入 (N, L, D)"]))
        shape_table.add_row("Transformer Encoder 输出", str(intermediate_shapes["Transformer Encoder 输出 (N, L, D)"]))
        shape_table.add_row("最终 LayerNorm 输入 (EOT Token)", str(intermediate_shapes["最终 LayerNorm 输入 (EOT Token, N, D)"]))
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