# Batch Normalization

This module covers **Batch Normalization (BN)**, a key technique in modern neural networks used to stabilize and accelerate training.

---

## Files

### 1. `batch_norm_from_scratch.py`
- Implements **Batch Normalization from scratch** using NumPy.
- Useful to understand the **mathematics behind BN**:
  - Compute mean and variance of a batch.
  - Normalize input.
  - Scale and shift using `gamma` and `beta`.
- Goal: Understand how BN reduces **internal covariate shift** and stabilizes learning.
- **Key notes:**
  - `gamma` and `beta` allow the model to **rescale and shift** normalized values.
  - During **training**, batch mean and variance are computed.
  - During **inference**, **running mean and running variance** are used for stable predictions.

### 2. `batch_norm_torch.py`
- Implements **Batch Normalization using PyTorch**.
- Includes:
  - `BatchNorm1d` → for Dense layers
  - `BatchNorm2d` → for Convolutional layers
  - `BatchNorm3d` → for 3D Conv / Video data
- PyTorch automatically manages **running mean/variance** for training and inference.

---

## Conceptual Notes

### Dense vs Convolutional Layers

| Feature | Dense | Conv |
|---------|-------|------|
| Input Shape | `(batch_size, features)` | `(batch_size, channels, H, W)` |
| BatchNorm Class | `BatchNorm1d` | `BatchNorm2d` |
| Normalization | Each feature independently | Each channel independently (normalize across H*W) |
| Goal | Stabilize input to each neuron | Stabilize input per channel, helps learning local features |

**Tip:**  
- Convolutional layers → BN **after Conv, before activation**  
- Dense layers → BN **after fully-connected, before activation**

---

### Why Batch Normalization?

1. **Reduces Internal Covariate Shift**  
   - Input to each layer does not change drastically during training → faster and more stable learning.

2. **Faster & Stable Training**  
   - Can use higher learning rates; without BN, high rates may cause divergence.

3. **Prepares for Dropout / Regularization**  
   - Inputs are more stable, so dropout and other regularization methods work better.

4. **Better Generalization**  
   - Normalized batch ensures the model learns features independent of input scale.

---

### Typical Workflow in Model

Input → Conv/Dense → BatchNorm → Activation → Next Layer

- **Dense Layers** → use `BatchNorm1d`
- **Conv Layers** → use `BatchNorm2d`
- **3D Conv / Video** → use `BatchNorm3d`

---

### References

- [Batch Normalization Paper (Ioffe & Szegedy, 2015)](https://arxiv.org/abs/1502.03167)
- [CS231n Notes: Batch Normalization](http://cs231n.stanford.edu/)
