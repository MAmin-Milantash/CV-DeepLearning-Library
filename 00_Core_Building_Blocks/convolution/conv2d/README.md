# Convolution 2D (Conv2D)

## 1. What is Conv2D?
Conv2D is a fundamental operation in computer vision used to extract spatial features such as edges, textures, and shapes from images.

It applies a learnable kernel (filter) that slides over the input feature map and computes dot products.

---

## 2. Historical Background
Convolutional Neural Networks gained popularity with:
- **LeNet-5 (1998)** – digit recognition
- **AlexNet (2012)** – ImageNet breakthrough
- Later used extensively in VGG, ResNet, EfficientNet, etc.

---

## 3. Mathematical Formulation
Given:
- Input: (N, C_in, H, W)
- Kernel: (C_out, C_in, K_h, K_w)

Output spatial size:
    H_out = (H + 2P − K_h) / S + 1
    W_out = (W + 2P − K_w) / S + 1

Where:
- P = padding
- S = stride

---

## 4. Intuition
Conv2D learns local patterns.
Early layers capture edges and textures.
Deeper layers capture semantic structures.

---

## 5. Implementation Strategy
This folder includes:
- A **from-scratch implementation** using PyTorch tensors
- A **PyTorch official implementation** using `nn.Conv2d`

---

## 6. When to Use Conv2D
- Image classification
- Object detection
- Segmentation
- Medical imaging :)


## Folder Detail:
- conv2d_from_scratch.py
    Just PyTorch tensor
    without nn.Conv2d ❌ 

- conv2d_pytorch.py
    Industrial Version

- test_conv2d.py
    For sanity check + Examples