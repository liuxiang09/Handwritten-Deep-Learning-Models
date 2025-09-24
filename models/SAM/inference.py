import argparse
import os
from typing import List, Optional, Tuple

import io
import urllib.request
import numpy as np
import torch
from PIL import Image

from model.image_encoder import ImageEncoderViT
from model.mask_decoder import MaskDecoder
from model.prompt_encoder import PromptEncoder
from model.sam import Sam
from model.transformer import TwoWayTransformer


# -----------------------------
# 模型搭建辅助函数
# -----------------------------
def build_sam(model_type: str = "sam_h",
			  image_size: int = 1024,
			  device: str = "cuda" if torch.cuda.is_available() else "cpu") -> Sam:
	"""
	使用本地实现构建 SAM 模型（不使用外部库一键加载）。

	model_type: {sam_b, sam_l, sam_h}
	image_size: 输入方形尺寸（模型内部会 pad 到该尺寸）
	"""
	model_type = model_type.lower()

	# 按官方 SAM-H 配置构建编码器（严格对齐权重键）：
	# img_size=1024, patch_size=16, embed_dim=1280, depth=32, heads=16,
	# use_abs_pos=True, use_rel_pos=True, window_size=14, global_attn_indexes=(7,15,23,31)
	image_encoder = ImageEncoderViT(
		img_size=image_size,
		patch_size=16,
		in_chans=3,
		embed_dim=1280,
		depth=32,
		num_heads=16,
		mlp_ratio=4.0,
		out_chans=256,
		qkv_bias=True,
		use_abs_pos=True,
		use_rel_pos=True,
		rel_pos_zero_init=True,
		window_size=14,
		global_attn_indexes=(7, 15, 23, 31),
	)

	# 计算嵌入尺寸（与 patch_size 保持一致，此处为 16）
	embed_hw = image_size // 16
	image_embedding_size: Tuple[int, int] = (embed_hw, embed_hw)

	prompt_encoder = PromptEncoder(
		embed_dim=256,
		image_embedding_size=image_embedding_size,
		input_image_size=(image_size, image_size),
		# 与官方权重对齐：mask_downscaling 中间通道较小（4/16/->256）
		mask_in_channels=16,
	)

	transformer = TwoWayTransformer(
		depth=2,
		embedding_dim=256,
		num_heads=8,
		mlp_dim=2048,
		attention_downsample_rate=2,
	)

	mask_decoder = MaskDecoder(
		transformer_dim=256,
		transformer=transformer,
		num_multimask_outputs=3,
		iou_head_depth=3,
		iou_head_hidden_dim=256,
	)

	model = Sam(
		image_encoder=image_encoder,
		prompt_encoder=prompt_encoder,
		mask_decoder=mask_decoder,
	)
	return model.to(device)


def load_pretrained(model: torch.nn.Module, ckpt_path: str, device: str) -> None:
	"""
	直接加载官方权重（严格匹配）。
	"""
	ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
	state = ckpt.get("model", ckpt.get("state_dict", ckpt))
	missing, unexpected = model.load_state_dict(state, strict=True)
	# 严格模式下一般不会返回 missing/unexpected，保留打印以便调试
	if missing:
		print(f"[load] 缺失键：{len(missing)}")
	if unexpected:
		print(f"[load] 额外键：{len(unexpected)}")


# -----------------------------
# I/O 辅助函数
# -----------------------------
def load_image_as_tensor(path_or_url: str) -> Tuple[torch.Tensor, Tuple[int, int]]:
	"""
	读取 RGB 图像，返回 float32 张量 [0,255]，形状 (3,H,W) 以及原始尺寸 (H,W)。
	支持本地路径或 http(s) URL。
	"""
	if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
		with urllib.request.urlopen(path_or_url) as r:
			data = r.read()
		img = Image.open(io.BytesIO(data)).convert("RGB")
	else:
		img = Image.open(path_or_url).convert("RGB")
	np_img = np.array(img).astype(np.float32)  # (H, W, 3), 0..255
	np_img = np.transpose(np_img, (2, 0, 1))  # (3, H, W)
	tensor = torch.from_numpy(np_img)
	h, w = img.height, img.width
	return tensor, (h, w)


def parse_points(points: List[str]) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
	"""
	解析 --point x,y,label（可重复）为张量。
	返回 (coords[B,N,2], labels[B,N])，B=1；若无点返回 None。
	"""
	if not points:
		return None
	coords: List[List[float]] = []
	labels: List[int] = []
	for p in points:
		xs = p.split(",")
		if len(xs) != 3:
			continue
		x_str, y_str, l_str = xs
		coords.append([float(x_str), float(y_str)])
		labels.append(int(l_str))
	coords_t = torch.tensor([coords], dtype=torch.float32)  # (1, N, 2)
	labels_t = torch.tensor([labels], dtype=torch.int64)    # (1, N)
	return coords_t, labels_t


def parse_box(box: Optional[str]) -> Optional[torch.Tensor]:
	"""
	解析 --box x1,y1,x2,y2 为形状 (1,1,4) 的张量。
	"""
	if not box:
		return None
	x1, y1, x2, y2 = [float(v) for v in box.split(",")]
	return torch.tensor([[[x1, y1, x2, y2]]], dtype=torch.float32)


def save_mask(mask: torch.Tensor, path: str) -> None:
	"""
	保存单通道掩码 (H,W) 为图像（0/255）。
	"""
	m = mask.detach().cpu().numpy()
	if m.dtype != np.bool_:
		m = m > 0.0
	m = (m.astype(np.uint8) * 255)
	Image.fromarray(m).save(path)


# -----------------------------
# Main
# -----------------------------
def main():
	parser = argparse.ArgumentParser(description="SAM 推理（本地实现版本）")
	parser.add_argument("--image", type=str, required=True, help="输入图像路径或 http(s) URL")
	parser.add_argument(
		"--checkpoint", type=str,
		default="models/SAM/checkpoints/sam_vit_h_4b8939.pth",
		help="预训练权重路径 (.pth)，默认使用本仓库提供路径"
	)
	parser.add_argument("--model-type", type=str, default="sam_h", choices=["sam_b", "sam_l", "sam_h"], help="模型规模")
	parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="计算设备：cpu 或 cuda")
	parser.add_argument("--multimask", action="store_true", help="是否输出多掩码（消歧）")
	parser.add_argument("--point", action="append", default=[], help="点提示：x,y,label，可重复传入")
	parser.add_argument("--box", type=str, default=None, help="框提示：x1,y1,x2,y2")
	parser.add_argument("--out", type=str, default="mask.png", help="第一张掩码的保存路径")
	args = parser.parse_args()

	device = args.device

	# 构建模型并加载权重
	model = build_sam(model_type=args.model_type, image_size=1024, device=device)
	print(f"[模型] 已构建 {args.model_type}，image_size=1024")
	if os.path.isfile(args.checkpoint):
		load_pretrained(model, args.checkpoint, device)
		print(f"[模型] 已从 {args.checkpoint} 加载权重")
	else:
		print(f"[警告] 未找到权重：{args.checkpoint}（将使用随机初始化权重进行推理）")

	model.eval()

	# 准备图像
	image_tensor, original_size = load_image_as_tensor(args.image)
	batched: dict = {
		"image": image_tensor.to(device),
		"original_size": original_size,
	}

	# 可选提示
	pt = parse_points(args.point)
	if pt is not None:
		coords, labels = pt
		batched["point_coords"] = coords.to(device)
		batched["point_labels"] = labels.to(device)
	bx = parse_box(args.box)
	if bx is not None:
		batched["boxes"] = bx.to(device)

	# 推理
	with torch.no_grad():
		outputs = model([batched], multimask_output=args.multimask)

	out = outputs[0]
	masks: torch.Tensor = out["masks"]  # (B, C, H, W)
	ious: torch.Tensor = out["iou_predictions"]  # (B, C)
	low_res: torch.Tensor = out["low_res_logits"]  # (B, C, 256, 256)

	# 保存第一个提示的第一张掩码
	first_mask = masks[0, 0]
	save_mask(first_mask, args.out)
	print(f"[完成] 已保存掩码到 {args.out}；masks={tuple(masks.shape)}, ious={tuple(ious.shape)}")


if __name__ == "__main__":
	main()

