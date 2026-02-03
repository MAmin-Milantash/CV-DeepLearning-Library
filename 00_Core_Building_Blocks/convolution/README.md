# Convolution Building Blocks

This folder provides a **complete and structured understanding of convolution operations**
from both **mathematical (from scratch)** and **framework-based (PyTorch)** perspectives.

The goal is to deeply understand:
- What convolution really is
- How Conv1D, Conv2D, and Conv3D differ
- How deep learning frameworks implement them
- Shape transformations and parameter counts

---

## 📁 Folder Structure

Convolution/
├── conv1d_from_scratch.py
├── conv2d_from_scratch.py
├── conv3d_from_scratch.py
│
├── conv1d_torch.py
├── conv2d_torch.py
├── conv3d_torch.py
│
├── utils.py
└── README.md


---

## 🧠 Conceptual Overview

### What is Convolution?
Convolution is a **sliding weighted sum operation** used to extract local patterns from data.

| Type   | Input Example              | Common Use Case           |
|------|---------------------------|---------------------------|
| Conv1D | Time series, audio        | NLP, speech, signals     |
| Conv2D | Images (H × W × C)        | Computer Vision          |
| Conv3D | Videos / Volumetric data | Medical imaging, video   |

---

## 📌 Two Implementation Styles

### 1️⃣ From Scratch (Pure Python / NumPy)
Used to:
- Understand math
- Understand sliding windows
- Understand padding, stride, kernel behavior

### 2️⃣ PyTorch Implementations
Used to:
- Match real-world deep learning usage
- Verify correctness
- Understand parameters & shapes

---

## 📄 File Descriptions

### `conv1d_from_scratch.py`
Manual implementation of 1D convolution using NumPy loops.

### `conv2d_from_scratch.py`
Manual implementation of 2D convolution (image-like data).

### `conv3d_from_scratch.py`
Manual implementation of 3D convolution (volume/video).

---

### `conv1d_torch.py`
Using `torch.nn.Conv1d` with shape analysis.

### `conv2d_torch.py`
Using `torch.nn.Conv2d` and channel-based convolution.

### `conv3d_torch.py`
Using `torch.nn.Conv3d` for volumetric data.

---

### `utils.py`
Shared helper functions:
- Padding
- Output shape calculation
- Visualization helpers


---
