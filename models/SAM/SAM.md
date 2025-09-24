## 简介

本目录提供 Segment Anything Model（SAM）的一个轻量本地实现，包含图像编码器（ViT-based）、提示编码器（点/框/掩码）、以及掩码解码器（Two-Way Transformer + 上采样与 IoU 头）。目标：

- 以清晰的结构复现 SAM 的核心推理路径；
- 使用本地实现类进行推理，验证与官方预训练权重的适配程度；
- 不依赖外部“一键加载官方模型”的封装，便于理解与二次开发。


## 模型结构与实现说明

- ImageEncoderViT（`model/image_encoder.py`）
	- 使用 `timm` 的 ViT 作为主干，去掉分类头，输出 patch 特征；
	- 通过一个简洁的 Neck（1x1/3x3 Conv + LayerNorm2d）调整通道到 256，并保持空间布局；
	- 约定输入图像在模型内部 pad 到 1024x1024，patch size 通常为 16（保证整除）。

- PromptEncoder（`model/prompt_encoder.py`）
	- 支持点、框、以及掩码三类提示；
	- 点/框通过随机位置编码 + 类型嵌入（正/负点、框角）得到稀疏提示嵌入；
	- 掩码通过下采样卷积得到与图像嵌入同尺度的密集提示嵌入；
	- 提供 `get_dense_pe()` 返回密集位置编码给解码器使用。

- MaskDecoder（`model/mask_decoder.py`）
	- 使用 TwoWayTransformer（`model/transformer.py`）在提示与图像嵌入之间进行交互；
	- 通过反卷积与超网络（HyperNetworks MLP）从 Transformer 特征生成掩码；
	- IoU 头预测每个掩码的质量分数；
	- 支持单掩码或多掩码（消歧）输出。

- Sam（`model/sam.py`）
	- 封装预处理（归一化 + pad 到 1024）、后处理（去 pad + 上采样回原图尺寸）、与端到端前向；
	- 输入为字典列表，包含图像、原始尺寸与可选提示（点/框/掩码）。


## 预训练权重与兼容性

- 本实现默认使用 `timm` 提供的 ViT 权重初始化图像编码器，这与官方 SAM-H 的骨干结构/命名并非完全一致；
- `inference.py` 加载 checkpoint 时使用 `strict=False`，可部分加载匹配权重，并打印缺失/多余键，便于评估适配程度；
- 若需要“严格对齐官方 SAM-H 权重”，需要：
	1) 复刻官方骨干结构与命名；或 2) 编写权重映射/转换脚本，将官方权重键名映射到本地实现；
- 当前默认在 `sam_h` 下使用 `vit_large_patch16_224` 作为安全回退（patch=16），可通过 `--vit` 参数显式指定 timm 的 ViT 名称（如你的环境具备 `vit_huge_patch16_224`）。


## 推理脚本

文件：`models/SAM/inference.py`

特点：
- 仅用于推理，不进行训练；
- 通过命令行参数传参；
- 支持本地路径或 URL 加载图像；
- 支持点/框提示与多掩码输出；
- 使用本地实现类构建模型，不使用第三方一键加载。

常用参数：
- `--image`：输入图像路径或 http(s) URL（必填）。
- `--checkpoint`：预训练权重路径，默认 `models/SAM/checkpoints/sam_vit_h_4b8939.pth`。
- `--model-type`：`sam_b` / `sam_l` / `sam_h`，默认 `sam_h`。
- `--vit`：可选，覆盖 timm 的 ViT 名称（如 `vit_huge_patch16_224`）。
- `--device`：`cuda` 或 `cpu`，默认自动侦测。
- `--multimask`：是否输出多掩码（消歧）。
- `--point`：点提示，格式 `x,y,label`，可重复传入多次（label：1 正样本，0 负样本）。
- `--box`：框提示，格式 `x1,y1,x2,y2`。
- `--out`：第一张掩码的保存路径，默认 `mask.png`。


### 运行示例

1) 本地图片、默认权重路径：

```bash
python models/SAM/inference.py \
	--image /path/to/your.jpg \
	--checkpoint models/SAM/checkpoints/sam_vit_h_4b8939.pth \
	--model-type sam_h \
	--out mask.png
```

2) 网络图片：

```bash
python models/SAM/inference.py \
	--image https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRdAOK50OIh-j4AOpeQ7XbiOm5JaKJ8q4eyMA&s \
	--checkpoint models/SAM/checkpoints/sam_vit_h_4b8939.pth \
	--model-type sam_h \
	--out mask.png
```

3) 点/框提示与多掩码：

```bash
python models/SAM/inference.py \
	--image /path/to/your.jpg \
	--checkpoint models/SAM/checkpoints/sam_vit_h_4b8939.pth \
	--model-type sam_h \
	--point 320,240,1 --point 100,120,0 \
	--box 50,60,420,360 \
	--multimask \
	--out mask.png
```

4) 指定 timm 的 ViT 名称（如果你的环境包含对应模型）：

```bash
python models/SAM/inference.py \
	--image /path/to/your.jpg \
	--checkpoint models/SAM/checkpoints/sam_vit_h_4b8939.pth \
	--model-type sam_h \
	--vit vit_huge_patch16_224 \
	--out mask.png
```


## 坐标与尺寸约定

- 输入图像在 `Sam.preprocess` 中进行归一化并 pad 到 `1024x1024`；
- 点/框提示应使用原图像素坐标（左上角为原点），`PromptEncoder` 会基于 `input_image_size=(1024,1024)` 做归一化与位置编码；
- 输出掩码由 `Sam.postprocess_masks` 去 pad 并上采样回原图尺寸。


## 依赖环境

- 关键依赖：`torch`, `timm`, `Pillow`, `numpy`（已在项目 `requirements.txt` 中列出）。


## 常见问题（FAQ）

1) 为什么不直接一键加载官方 SAM-H？
	 - 目的是用本地实现验证与理解模型结构；同时评估与官方权重的兼容度。

2) timm 的 ViT 与官方骨干完全兼容吗？
	 - 不是完全兼容。当前通过 `strict=False` 部分加载，打印缺失/多余键用于诊断；若需完全一致，需要结构与键名严格对齐或进行权重转换。

3) 输出多掩码有什么意义？
	 - 官方设计用于消歧：在提示不明确时给出多个可能的掩码，结合 IoU 评分选择最优。


## 目录结构

- `model/`
	- `sam.py`：端到端封装；
	- `image_encoder.py`：ViT 主干 + Neck；
	- `prompt_encoder.py`：点/框/掩码提示编码；
	- `mask_decoder.py`：Two-Way Transformer + 上采样 + IoU 头；
	- `transformer.py`：注意力与解码器模块；
	- `common.py`：通用层（MLPBlock、LayerNorm2d）。
- `inference.py`：命令行推理脚本；
- `checkpoints/`：放置权重（默认 `sam_vit_h_4b8939.pth`）。


## 致谢

本实现参考了开源社区对 SAM 的讨论与结构描述，简化与本地化重构仅用于学习与研究目的。

