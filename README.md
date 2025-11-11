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
