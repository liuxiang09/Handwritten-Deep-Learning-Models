import torch
from transformers import pipeline
cifar10_labels = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'] 

pipeline = pipeline(
    task="image-classification",
    # model="google/vit-base-patch16-224",
    model="./model/ViT/fine_tuned_vit_cifar10",
    torch_dtype=torch.float16,
    device=0,
    use_fast=True,
)
predictions = pipeline(images="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg")
for item in predictions:
    label = item['label'].replace("LABEL_", '')
    label = int(label)

    if 0 <= label < len(cifar10_labels):
        item['label'] = cifar10_labels[label]
    else:
        print("ERROR LABEL!")
print(predictions)


