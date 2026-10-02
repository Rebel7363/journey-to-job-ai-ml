# Module 08: Computer Vision & Convolutional Neural Networks (CNNs)

This module explores computer vision foundations, ranging from pure spatial convolution arithmetic in NumPy to modern modular CNN architectures and training pipelines in PyTorch.

---

## Architecture & Implementation Overview

| Script | Domain | Key Concepts |
| :--- | :--- | :--- |
| `01_convolution_from_scratch.py` | Math / Spatial Ops | 2D discrete convolution from scratch, padding arithmetic, stride mechanics, Sobel edge filtering |
| `02_pytorch_cnn_architecture.py` | PyTorch Architecture | `nn.Conv2d`, `nn.BatchNorm2d`, `nn.MaxPool2d`, spatial dimension tracking, fully connected head |
| `03_data_augmentation_transforms.py` | Vision Ingestion | Pipeline construction using `torchvision.transforms`, random flips, crops, and channel normalization |
| `04_cnn_training_pipeline.py` | End-to-End Pipeline | Mini-batch vision data ingestion, AdamW optimization, MPS GPU acceleration, validation metrics |

---

## Mathematical Foundations

### 1. Spatial Dimension Formula
Given an input feature map of size $W \times H$, kernel size $K$, padding $P$, and stride $S$, the output dimension is:

$$O = \left\lfloor \frac{W - K + 2P}{S} \right\rfloor + 1$$

### 2. Why Convolution Works
- **Parameter Sharing:** The same filter weights slide across the entire image, drastically reducing parameter counts compared to dense layers.
- **Translation Equivariance:** Shifting a feature in the input produces a corresponding shifted activation in the output feature map.

---

## Verification & Execution

To run the complete CNN pipeline with hardware acceleration:

```bash
python 08_Computer_Vision_CNNs/04_cnn_training_pipeline.py