# 数据集下载链接

## Flickr30k 数据集

- [Flickr30k 数据集下载链接](https://www.kaggle.com/api/v1/datasets/download/hsankesara/flickr-image-dataset)

下载文件并解压到 `data/flickr30k_images/` 目录下。
期望的目录结构：

```
flickr30k_images/
├── flickr30k_images/...     # 包含所有图片
└── results.csv              # 包含图片-文本对
```

## Pascal VOC 数据集

- [Pascal VOC 数据集下载链接](https://www.kaggle.com/api/v1/datasets/download/gopalbhattrai/pascal-voc-2012-dataset)

下载并解压到 `data/Pascal_VOC/` 目录下。
期望的目录结构：

```
Pascal_VOC/
├── VOC2012_train_val/
│   └── VOC2012_train_val/
│       ├── Annotations/         # 标注文件
│       ├── ImageSets/           # 图像集划分
│       ├── JPEGImages/          # 图片文件
│       ├── SegmentationClass/   # 语义分割掩码
│       └── SegmentationObject   # 实例分割掩码
│
└── VOC2012_test/
    └── VOC2012_test/
        ├── Annotations/         # 标注文件
        ├── ImageSets/           # 图像集划分
        └── JPEGImages/          # 图片文件
        
```

## 注意

这些数据集仅供科研学习用途，使用者的一切其他用途与本项目无关。
