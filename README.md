# H2Net： Enhanced Object Detection in Water Disaster Scenarios via Orthogonal Channel Attention and Wavelet Fusion
[![Pytorch](https://img.shields.io/badge/PyTorch-2.2.1%2Bcu121-red)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-8.0.201-blue)](https://github.com/ultralytics/ultralytics)


> **投稿信息**：《The Visual Computer》期刊在投中 

---
## 核心创新
针对水域灾害检测的关键挑战：
- 🛡️ **OCA正交通道注意机制**：抑制水面反光/波浪噪声，降低特征冗余
- 🌊 **Haar小波多尺度融合（WTConv）**：对数级扩展感受野，提升极小目标检测能力
- ⚡ **83 FPS实时性能**：RTX 4060Ti实测速度，较baseline参数量↓22.8% (15.3M)，计算量↓6.67% (53.2G)
- 🔁 **跨域泛化能力**：SeaDronesSee-V2 + VisDrone2019验证有效性
![H2Net在mAP50与计算效率的平衡性](docs\results\image1.png)

---
## 环境配置

### 硬件要求

- GPU: NVIDIA GeForce RTX 4060Ti 16G (或同等级别)
- RAM: ≥32GB

### 软件环境
```bash
# 创建虚拟环境
conda create -n h2net python=3.10
conda activate h2net
# 安装核心依赖
pip install torch==2.2.1+cu121 torchvision==0.17.1+cu121
pip install pywavelets==1.5.0 opencv-python==4.9.0.80 scikit-learn==1.4.0
```

---
## 快速开始

### 数据集准备
实验使用的数据集为：[SeaDronesSee Object Detection v2](https://cloud.cs.uni-tuebingen.de/index.php/s/ZZxX65FGnQ8zjBP)的Compressed Version版本
官方数据集为COCO标注，本实验的YOLO标注下载链接如下：百度网盘链接: https://pan.baidu.com/s/1lbVx9UCtVvn2qLAMR1iZoQ?pwd=bshe
下载数据集并组织为以下结构：

```TEXT
datasets/
├── drowning_person_yolo/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│   │   ├── train/
│   │   └── val/
│   │   └── classes.txt
└── VisDrone2019_yolo/
│   └── images/
│   └── labels/
```
使用datasets\A_drowning_person.yaml的配置文件进行训练与验证


### 预训练权重下载

RTDETR-r18 预训练权重下载地址：
百度网盘链接：https://pan.baidu.com/s/18o2uId1X8z8U66lgZyP8Vw?pwd=ulwk

### 训练H2Net模型 (train.py)

```bash
python train.py
```

### 模型验证 (val.py)

```bash
python val.py
```

