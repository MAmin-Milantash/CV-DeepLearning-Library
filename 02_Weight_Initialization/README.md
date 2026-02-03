# 02_Weight_Initialization

Weight initialization is a crucial step in training neural networks. Proper initialization can:

- Accelerate convergence
- Reduce vanishing/exploding gradients
- Improve stability during training
- Prepare the model for other techniques like Batch Normalization and Dropout

---

## 🗂 Folder Structure

02_Weight_Initialization/
├── weight_init_from_scratch.py
├── weight_init_torch.py
├── utils.py
└── README.md


---

## 1️⃣ weight_init_from_scratch.py

Implements classic weight initialization methods **without using PyTorch**:

| Method | Description | Best for |
|--------|------------|----------|
| Zero Initialization | All weights set to zero | ❌ Not recommended (all neurons behave the same) |
| Random Normal / Uniform | Random values from Gaussian or Uniform distribution | Dense/Conv layers |
| Xavier / Glorot | Scales weights based on fan_in & fan_out | Sigmoid / Tanh activations |
| He / Kaiming | Scales weights based on fan_in | ReLU activations |

**Goal:** Understand the effects of initialization on forward propagation and vanishing/exploding gradients.

---

## 2️⃣ weight_init_torch.py

Implements the same initialization methods **using PyTorch** (`torch.nn.init`):

- `nn.init.zeros_`
- `nn.init.normal_`
- `nn.init.uniform_`
- `nn.init.xavier_uniform_`, `nn.init.xavier_normal_`
- `nn.init.kaiming_uniform_`, `nn.init.kaiming_normal_`

**Goal:** Prepare Dense and Conv layers for real-world models.

---

## 3️⃣ utils.py

Helper functions:

- Apply initialization automatically to all layers
- Print basic statistics (mean, std, min, max) for debugging
- Test convergence in small networks

---

## 🔹 Important Concepts

### Why Initialization Matters
- Zero init → neurons learn the same thing → symmetry problem
- Random normal/uniform → may cause vanishing/exploding gradients
- Xavier → preserves variance for Sigmoid/Tanh
- He → preserves variance for ReLU

### Dense vs Conv Layers
| Layer Type | Recommended Init |
|------------|----------------|
| Dense      | Xavier / He (depends on activation) |
| Conv2d     | He (for ReLU), Xavier (for Sigmoid/Tanh) |
| Conv3d     | He (for ReLU), Xavier (for Sigmoid/Tanh) |

### Interview Tip
- Weight initialization affects **convergence speed** and **learning rate stability**
- Improper initialization → slow learning or model stuck
- Always match initialization to **activation function** and **layer type**

---

## Example (Dense + Conv)

```python
import torch
import torch.nn as nn
from utils import apply_weight_init, print_weight_stats

model = nn.Sequential(
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Conv2d(3, 16, kernel_size=3, padding=1)
)

apply_weight_init(model, method="he")
print_weight_stats(model)
Output: Shows weight distribution after initialization.