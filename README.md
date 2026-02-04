# 🏗️ Deep Learning Core & Advanced Modules Repository

This repository contains a **comprehensive, modular deep learning codebase**, designed for:

* Research
* Learning & experimentation
* Production-ready pipelines
* Interview preparation

It covers the full workflow from **core building blocks** to **advanced architectural patterns**, **optimizers**, **hyperparameter tuning**, **transfer learning**, and **visualizations**.

---

## 📁 Project Folder Structure Overview

```
00_Core_Building_Blocks/
01_Data_Preprocessing/
02_Weight_Initialization/
03_Regularization/
04_Optimizers_and_LR_Schedulers/
05_Hyperparameter_Tuning/
06_Activation_Functions/
07_Architectural_Blocks/
08_Transfer_Learning/
09_Dimension_Analysis/
10_Experiments_and_Visualizations/
```

---

## 00_Core_Building_Blocks

**Purpose:** Fundamental components of deep learning models, especially CNNs.

### Modules:

1. **Convolution**

   * 1D, 2D, 3D convolution
   * From-scratch & PyTorch implementations
   * Mathematical intuition for kernels, stride, padding

2. **Pooling & Downsampling**

   * Max, Average, Adaptive pooling
   * Spatial resolution reduction & translation invariance

3. **Receptive Field & Dimension Tracking**

   * Layer-by-layer dimension analysis
   * Receptive field computation
   * Essential for CNN architecture design & interviews

---

## 01_Data_Preprocessing

**Purpose:** Prepare and augment datasets for robust training.

* **Augmentation Techniques:** Flips, rotations, color jitter, random crop, mixup/cutmix
* **Normalization & Standardization**
* **Goal:** Increase generalization, prevent overfitting, handle real-world image variations

---

## 02_Weight_Initialization

**Purpose:** Proper initialization of weights to stabilize training.

* **Methods from scratch:** Zero, Random, Xavier, He
* **PyTorch implementations**
* **Goal:** Avoid vanishing/exploding gradients, improve convergence speed

---

## 03_Regularization

**Purpose:** Prevent overfitting and stabilize learning.

* Dropout (from scratch & PyTorch)
* L1 / L2 weight decay
* Early stopping
* Label smoothing
* **Goal:** Improve generalization, prepare for deep learning regularization

---

## 04_Optimizers_and_LR_Schedulers

**Purpose:** Control how the model learns during training.

* **Optimizers from scratch & PyTorch:** SGD, Momentum, RMSProp, Adam
* **Learning Rate Schedulers:** Step, Exponential, Cosine, Reduce-on-Plateau, Warm-up
* **Comparison experiments** & **utilities**
* **Goal:** Understand convergence dynamics, select optimal optimizer + scheduler

---

## 05_Hyperparameter_Tuning

**Purpose:** Find optimal hyperparameters for best model performance.

* **Methods:** Grid Search, Random Search, Bayesian Optimization
* **Utilities:** Logging, plotting, saving best models
* **Goal:** Improve convergence, generalization, and experiment reproducibility

---

## 06_Activation_Functions

**Purpose:** Provide a library of activation functions with proper formula references.

* **Functions included:** Sigmoid, Tanh, ReLU, Leaky ReLU, GELU, Softmax
* **Goal:** Stabilize learning, introduce non-linearity, enable gradient flow

---

## 07_Architectural_Blocks

**Purpose:** Modular building blocks for constructing deep networks.

* Dense Blocks
* Conv Blocks
* Residual Blocks (ResNet-style)
* Inception / Multi-branch Blocks
* Attention Blocks (Self-Attention, Transformer-style)
* **Goal:** Build flexible, deep, and modern network architectures

---

## 08_Transfer_Learning

**Purpose:** Use pretrained models to accelerate training and improve generalization.

* **Backbones:** ResNet, EfficientNet, VGG, Transformer models
* **Techniques:** Feature extraction, fine-tuning, layer freezing
* **Goal:** Leverage prior knowledge, reduce training time, achieve better performance

---

## 09_Dimension_Analysis

**Purpose:** Track input/output shapes, receptive fields, and tensor dimensions.

* Layer-by-layer analysis
* Compute receptive field & padding requirements
* Visualize transformations
* **Goal:** Debugging, architecture design, interviews

---

## 10_Experiments_and_Visualizations

**Purpose:** Test, evaluate, and visualize models & components.

* Compare optimizers & LR schedulers
* Track loss, accuracy, gradient norms
* Visualize feature maps, learning curves, and hyperparameter performance
* **Goal:** Analyze model behavior and validate theoretical understanding

---

## 🔑 Key Takeaways

1. **Workflow follows logical deep learning order:** Data → Weights → Regularization → Optimizers → Hyperparameters → Activations → Architectural Blocks → Transfer Learning → Dimension Analysis → Experiments
2. **Each module is self-contained** but designed to integrate seamlessly.
3. **From scratch + PyTorch:** Learn both theoretical foundations and practical implementations.
4. **Research-ready, interview-ready, production-ready:** Suitable for study, experiments, and deployment.
