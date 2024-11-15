# 农业病害虫识别

## 环境

Ubuntu 22.04 LTS

CUDA 12.4

cuDNN 8.9.7

python 3.10.15

pytorch 2.5.1

conda 24.1.2

## 数据集

https://www.kaggle.com/datasets/simranvolunesia/pest-dataset

#### 注意
这是作者第一次尝试写机器学习模型，所以构建的模型可能会很烂，而且因为作者的显卡太烂（NVIDIA GeForce MX250），显存太小(1.95 GiB)，所以dataLoader每一次只加载了16张图片，如果有条件可以修改。