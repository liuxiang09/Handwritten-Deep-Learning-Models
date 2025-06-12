import timm

# 使用通配符查找所有和 vit_b_16 或 vit_base_patch16 相关的、并且带预训练权重的模型
# 我们直接查找 'vit_b_16' 因为它更通用
matching_models = timm.list_models('*vit*', pretrained=True)

print("您当前环境中可用的相关ViT模型有：")
for model_name in matching_models:
    print(model_name)