import torch
from tqdm import tqdm

def evaluate(model, dataloader, device):
    """
    更标准、更全面地评估 CLIP 模型在“一对多”检索任务上的性能。
    分别计算 Image-to-Text 和 Text-to-Image 的 Recall@1 和 Recall@5。
    """
    model.eval()
    
    # 初始化各种指标的计数器
    total_samples = 0
    i2t_r1_correct = 0
    i2t_r5_correct = 0
    t2i_r1_correct = 0
    t2i_r5_correct = 0

    print("🚀 Starting comprehensive evaluation for retrieval...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # pixel_values: [N, C, H, W]
            # input_ids: [5*N, max_len]
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            num_images = pixel_values.shape[0]
            num_texts = input_ids.shape[0]

            # 1. 获取特征
            image_features, text_features = model(pixel_values, input_ids, attention_mask)
            logit_scale = model.logit_scale.exp()

            # 2. 计算相似度矩阵
            # logits_per_image (I2T): [N, 5*N]
            logits_per_image = image_features @ text_features.T * logit_scale
            # logits_per_text (T2I): [5*N, N]
            logits_per_text = logits_per_image.T
            
            # --- 3. Image-to-Text (I2T) Recall 计算 ---
            # 对于第 i 张图片，正确的文本索引是 [i*5, i*5+1, ..., i*5+4]
            
            # I2T Recall@1
            # 找到每张图片最匹配的文本索引
            i2t_preds_r1 = logits_per_image.argmax(dim=1)
            # 检查预测是否在正确范围内
            for i in range(num_images):
                if (i * 5) <= i2t_preds_r1[i] < ((i + 1) * 5):
                    i2t_r1_correct += 1

            # I2T Recall@5
            # 找到每张图片最匹配的前5个文本索引
            _, i2t_preds_r5_indices = logits_per_image.topk(5, dim=1)
            # 检查这top-5的预测中，是否有任何一个落在正确的5个答案里
            for i in range(num_images):
                pred_indices = set(i2t_preds_r5_indices[i].tolist())
                true_indices = set(range(i * 5, (i + 1) * 5))
                if len(pred_indices & true_indices) > 0:
                    i2t_r5_correct += 1

            # --- 4. Text-to-Image (T2I) Recall 计算 ---
            # 对于第 j 个文本，正确的图片索引是 floor(j / 5)
            ground_truth = torch.arange(num_texts, device=device) // 5 # [0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,...]

            # T2I Recall@1
            t2i_preds_r1 = logits_per_text.argmax(dim=1)
            t2i_r1_correct += (t2i_preds_r1 == ground_truth).sum().item()

            # T2I Recall@5
            _, t2i_preds_r5_indices = logits_per_text.topk(5, dim=1) # [N*5, 5]
            # 检查正确答案是否出现在 top-5 预测中
            t2i_r5_correct += (t2i_preds_r5_indices == ground_truth.unsqueeze(1)).any(dim=1).sum().item()
            
            total_samples += num_images

    # --- 5. 计算并打印最终结果 ---
    i2t_r1 = 100 * i2t_r1_correct / total_samples
    i2t_r5 = 100 * i2t_r5_correct / total_samples
    # 对于T2I，样本总数是 5 * total_samples
    t2i_r1 = 100 * t2i_r1_correct / (total_samples * 5)
    t2i_r5 = 100 * t2i_r5_correct / (total_samples * 5)

    print("\n✅ Evaluation Results:")
    print(f"  Image-to-Text Recall@1: {i2t_r1:.2f}%")
    print(f"  Image-to-Text Recall@5: {i2t_r5:.2f}%")
    print("-" * 30)
    print(f"  Text-to-Image Recall@1: {t2i_r1:.2f}%")
    print(f"  Text-to-Image Recall@5: {t2i_r5:.2f}%")
    
    return {
        "i2t_r1": i2t_r1, "i2t_r5": i2t_r5,
        "t2i_r1": t2i_r1, "t2i_r5": t2i_r5
    }