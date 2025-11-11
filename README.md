# CRRE-YOLO

This repository provides the official implementation and training scripts for our paper:

> **CRRE-YOLO: An Enhanced YOLOv11 Model with Efficient Local Attention and Multi-Scale Convolution for Rice Pest Detection**  
> *(Submitted to Applied Sciences (MDPI), 2025)*

---

## 1. Overview

This project implements the CRRE-YOLO model described in the paper.  
The repository contains the training script, configuration, and environment settings necessary to reproduce the experimental results reported in **Section 2.3 (Experimental Platform and Parameter Settings)** of the manuscript.

---

## 2. Environment

All experiments were conducted on a local computer with the following configuration:

- **Operating System:** Windows 11  
- **CPU:** Intel Core i5-14600KF  
- **RAM:** 32 GB  
- **GPU:** NVIDIA GeForce RTX 5060 Ti (16 GB VRAM)  
- **Python:** 3.10.18  
- **PyTorch:** 2.7.1  
- **CUDA:** 12.8  

All models were trained and validated on this local machine.

### 2.1 Python Dependencies

Install the required Python packages with:

```bash
pip install -r requirements.txt
您可以根据本地设置调整 requirements.txt，但其中列出的版本与本文中描述的环境一致。

3. 可重复性设置
为确保可重复性，所有实验都使用以下设置：

使用固定的随机种子：

python
复制
编辑
seed = 0
种子应用于：

Python 随机

数字

PyTorch

启用 PyTorch 确定性模式，例如：

python
复制
编辑
import torch
import numpy as np
import random

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
关键训练设置包括：

输入大小：640 × 640

批量大小（训练）：4

纪元数：200

这些细节对应于论文的第 2.3 节（实验平台和参数设置）。

4. 培训
运行主训练脚本：

bash
复制
编辑
python train.py
或者使用显式参数：

bash
复制
编辑
python train.py --img 640 --batch 4 --epochs 200
确保在 train.py 或 config.yaml 中正确设置数据集路径。

5. 推理
训练后，使用以下命令运行推理：

bash
复制
编辑
python detect.py --weights runs/train/exp/weights/best.pt --source ./test_images --img 640
（根据您自己的实现调整脚本名称和参数。

6. 推理速度
推理性能（单张图片）：

批量大小：1

平均延迟： 每张图像 3.0 毫秒

0.2 毫秒预处理

2.0 毫秒推理

0.8 毫秒后处理

使用 Ultralytics YOLO 框架的内置分析器进行测量。

7. 存储库结构
text
复制
编辑
CRRE-YOLO/
├── train.py
├── detect.py
├── config.yaml
├── requirements.txt
├── README.md
└── weights/
8. 数据
本研究中使用的水稻害虫数据集可根据通讯作者的合理要求提供。

9. 引用
如果您使用此存储库，请引用：

bibtex
复制
编辑
@article{crre-yolo-2025,
  title   = {CRRE-YOLO: An Enhanced YOLOv11 Model with Efficient Local Attention and Multi-Scale Convolution for Rice Pest Detection},
  author  = {[Your Name] and [Co-authors]},
  journal = {Applied Sciences},
  year    = {2025},
  note    = {under review}
}
10. 许可
该项目根据 MIT 许可证发布。

text
复制
编辑
MIT License

Copyright (c) 2025 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
