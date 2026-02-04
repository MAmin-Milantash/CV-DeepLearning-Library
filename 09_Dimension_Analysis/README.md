# 📁 09_Dimension_Analysis

## 🧭 Purpose

This folder focuses on **analyzing the dimensionality of data** throughout a neural network pipeline. Understanding shapes and dimensions is critical for:

* Debugging mismatched inputs in layers
* Visualizing feature map transformations
* Understanding bottlenecks and expansion in networks
* Designing multi-branch and complex architectures

It’s **especially useful for CNNs, ResNets, Inception modules, and attention-based networks**.

---

## 🧭 Why Dimension Analysis Matters

* Ensures that **layer inputs/outputs match** expected shapes
* Helps detect **broadcasting issues, flattening mistakes, or reshaping errors**
* Critical for **model debugging and architectural design**
* Facilitates **efficient feature visualization and analysis**

---

## 🪜 Recommended Learning Order

1️⃣ Input data shape exploration and summary
2️⃣ Feature map dimension tracing through Dense/Conv/Residual blocks
3️⃣ Flattening, reshaping, and concatenation handling
4️⃣ Multi-branch and skip connection dimension verification
5️⃣ Utility functions for automated shape logging

---

## 🧱 Folder Structure

```
09_Dimension_Analysis/
├── input_analysis.py         # Explore raw input shapes, channels, batches
├── conv_feature_maps.py      # Trace Conv layers and output dimensions
├── residual_dimensions.py    # Check skip connections & addition layers
├── inception_dimensions.py   # Verify multi-branch concatenation outputs
├── attention_dims.py         # Track shapes in self-attention modules
├── flatten_and_reshape.py    # Utilities for flattening and reshaping tensors
├── utils.py                  # Helper functions: print shapes, assert checks
└── README.md
```

---

## 📄 File Descriptions

### 🔹 `input_analysis.py`

* Analyze input data shapes: images, sequences, or tabular features.
* Compute batch, channel, height, width (for CNN) or features (for MLP).
* Detect inconsistencies before feeding into the network.

**Goal:**
    Make sure raw inputs are compatible with network architecture.
    Checking the shape and dimensions of inputs before entering the network.
---

### 🔹 `conv_feature_maps.py`

* Trace **Conv2D or Conv3D layers**.
* Compute output shapes given kernel, stride, padding, and dilation.
* Visualize feature map dimensions across the network.

**Goal:** 
    Understand how spatial dimensions evolve through Conv layers.
    Tracking output dimensions in convolutional layers.
---

### 🔹 `residual_dimensions.py`

* Check **skip connections** for dimension compatibility.
* Supports **addition or concatenation** in residual blocks.
* Warns if input/output shapes mismatch in ResNet-style modules.

**Goal:** 
    Ensure residual connections are valid and gradients flow correctly.
    Checking dimensional compatibility in skip connections.
---

### 🔹 `inception_dimensions.py`

* Analyze **multi-branch outputs** (1x1, 3x3, 5x5 convs)
* Merge branch outputs and verify channel dimensions.
* Useful for Inception-style networks with complex concatenations.

**Goal:** 
    Prevent concatenation errors and understand multi-scale feature aggregation.
    Checking dimensional compatibility in multi-branch blocks.
---

### 🔹 `attention_dims.py`

* Track **query, key, value** tensor shapes in self-attention layers.
* Verify head splitting, concatenation, and projection dimensions.
* Supports both **sequence and image attention**.

**Goal:** 
    Ensure attention mechanism preserves expected shapes.
    Tracking dimensions in self-attention modules.
---

### 🔹 `flatten_and_reshape.py`

* Utilities for **flattening** Conv features to feed into Dense layers
* Reshape tensors for concatenation or multi-branch integration
* Assert functions to confirm final dimensions match expected values

**Goal:** 
    Make transitions between Conv → Dense or multi-branch blocks seamless.
    Help transform tensors between Conv → Dense or multi-branch.
---

### 🔹 `utils.py`

* Print **layer shapes** dynamically
* Assert shape correctness at runtime
* Visualize **tensor flow** through the network
* Log shapes during training for debugging purposes

**Goal:** Centralize all helper functions for dimension analysis and monitoring.

---

## 🧠 Key Concepts / Notes

| Topic                 | Purpose / Benefit                                 |
| --------------------- | ------------------------------------------------- |
| Input Analysis        | Detect incompatible shapes early                  |
| Conv Feature Maps     | Track spatial and channel dimensions              |
| Residual Connections  | Ensure addition/skip works correctly              |
| Multi-branch Networks | Verify concatenation, channel aggregation         |
| Attention Mechanisms  | Track query/key/value shapes and head projections |
| Flatten / Reshape     | Smooth transition between Conv and Dense layers   |

**Tips for Interviews:**

* Explain how Conv layer output shapes are computed
* Describe skip connection dimension checks
* Discuss multi-branch concatenation issues and solutions
* Explain attention head shape reasoning

---

## 🎯 Goals After This Folder

* Understand **how each layer changes the tensor shape**
* Detect and prevent **dimensionality errors**
* Prepare **networks for visualization and debugging**
* Build **deep, multi-branch, and attention networks confidently**

---