# Data Preprocessing for Computer Vision

In computer vision, the quality of data preprocessing
often matters as much as the model architecture itself.

Even the most advanced neural networks can fail
if the input data is not properly prepared.

This directory focuses on **how raw visual data is transformed**
before entering a deep learning model.

---

## Learning Order (Conceptual Roadmap)

The learning path in this folder follows the **real-world data flow**:

Raw Data  
→ Normalization & Scaling  
→ Data Augmentation  
→ Batch Normalization (inside the network)

Each step builds intuition and practical skills
for training stable and well-generalizing models.

---

## Step 1 — Normalization & Scaling (First and Most Important)

📁 `normalization/`

Normalization is the very first operation applied to raw input data.

### Why is normalization done first?

- It affects **all deep learning models**, not just CNNs
- Poor normalization can **completely prevent learning**
- It significantly improves:
  - Training stability
  - Convergence speed
  - Numerical robustness
- It is a **very common interview topic**

---

### What problems does normalization solve?

Raw image pixel values usually lie in ranges like:
- `[0, 255]` for images
- Different channels may have different distributions

Neural networks work best when:
- Inputs have similar scales
- Distributions are centered and well-behaved

---

### Key Concepts Covered in This Section

#### 1. Scale vs Distribution

- **Scaling** controls the numeric range of values
- **Distribution** controls how values are spread (mean & variance)

Two datasets can have the same scale but very different distributions,
which affects how gradients flow during training.

---

#### 2. Why CNNs Are Sensitive to Normalization

- Convolution layers amplify scale differences
- Unnormalized inputs cause:
  - Exploding or vanishing activations
  - Unstable gradients
- Normalization makes optimization landscapes smoother

---

#### 3. ImageNet Mean/Std vs Custom Dataset Statistics

- Pretrained models expect **ImageNet-normalized inputs**
- Using wrong statistics can degrade transfer learning performance
- When and why to compute dataset-specific mean/std

---

## Folder Structure

normalization/
├── min_max_normalization.py # From-scratch Min-Max scaling
├── standardization.py # Z-score normalization
├── normalization_torch.py # PyTorch-based normalization
└── README.md # Detailed explanations & comparisons


Each technique is implemented:
- From scratch (to understand the math)
- Using PyTorch (industry standard practice)

---

## What You Should Gain From This Module

By completing this section, you should be able to:

- Explain normalization intuitively and mathematically
- Choose the correct normalization strategy for a task
- Debug training issues caused by poor preprocessing
- Confidently answer normalization-related interview questions

---
Raw Data → Normalize → Augment → BatchNorm 