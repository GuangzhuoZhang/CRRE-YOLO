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

---

### 2.1 Python Dependencies

Install the required Python packages with:

```bash
pip install -r requirements.txt
```

You may adjust `requirements.txt` according to your local setup, but the versions listed there are consistent with the environment described in the paper.

---

## 3. Reproducibility Settings

To ensure reproducibility, the following settings are used throughout all experiments:

```python
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
```

**Training parameters:**
- Input size: 640 × 640  
- Batch size: 4  
- Epochs: 200  

These settings correspond to Section 2.3 of the paper.

---

## 4. Training

Run the main training script:

```bash
python train.py
```

Or with explicit arguments:

```bash
python train.py --img 640 --batch 4 --epochs 200
```

Ensure dataset paths are correctly set in `train.py` or `config.yaml`.

---

## 5. Inference

After training, perform inference using:

```bash
python detect.py --weights runs/train/exp/weights/best.pt --source ./test_images --img 640
```

---

## 6. Inference Speed

Single-image inference (batch = 1):

- **Average Latency:** 3.0 ms per image  
  - 0.2 ms preprocessing  
  - 2.0 ms inference  
  - 0.8 ms post-processing  

Measured using the built-in profiler of the Ultralytics YOLO framework.

---

## 7. Repository Structure

```text
CRRE-YOLO/
├── train.py
├── detect.py
├── config.yaml
├── requirements.txt
├── README.md
└── weights/
```

---

## 8. Data

The rice pest dataset used in this study is available upon reasonable request from the corresponding author.

---

## 9. Citation

```bibtex
@article{crre-yolo-2025,
  title   = {CRRE-YOLO: An Enhanced YOLOv11 Model with Efficient Local Attention and Multi-Scale Convolution for Rice Pest Detection},
  author  = {Guangzhuo Zhang and Co-authors},
  journal = {Applied Sciences},
  year    = {2025},
  note    = {under review}
}
```

---

## 10. License

```text
MIT License

Copyright (c) 2025 Guangzhuo Zhang

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
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```
