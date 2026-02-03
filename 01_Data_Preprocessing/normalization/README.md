# Normalization & Scaling in Computer Vision

Normalization is a fundamental preprocessing step that directly affects
how neural networks learn from visual data.

In computer vision, improper normalization can lead to:
- Slow or unstable training
- Poor convergence
- Degraded performance in transfer learning

This module focuses on understanding **why normalization works**,
**how different methods differ**, and **when to use each one**.

---

## What Is Normalization?

Normalization transforms input data into a form that is easier
for neural networks to process.

For images, this usually means transforming pixel values so that:
- They lie in a controlled numeric range
- Their statistical distribution is well-behaved

---

## Common Pixel Value Ranges

Typical raw image formats:
- `[0, 255]` for uint8 images
- `[0, 1]` after simple scaling

Neural networks do **not** inherently understand these ranges.
They only react to relative magnitudes and distributions.

---

## Normalization vs Standardization

Although often used interchangeably, they are conceptually different.

### 1. Min-Max Normalization (Scaling)

**Formula:**
\[
x' = \frac{x - x_{min}}{x_{max} - x_{min}}
\]

**Properties:**
- Maps values to a fixed range (usually `[0, 1]`)
- Preserves relative distances
- Sensitive to outliers

**When to use:**
- When input range must be bounded
- Simple pipelines
- Non-deep-learning models

---

### 2. Standardization (Z-score Normalization)

**Formula:**
\[
x' = \frac{x - \mu}{\sigma}
\]

**Properties:**
- Centers data around zero
- Unit variance
- More robust for gradient-based optimization

**Why preferred for CNNs:**
- Activations become more symmetric
- Gradients are more stable
- Faster convergence

---

## Why CNNs Strongly Depend on Normalization

Convolution layers:
- Accumulate values across spatial regions
- Amplify scale inconsistencies
- Propagate distribution shifts layer by layer

Without normalization:
- Early layers saturate
- Later layers receive noisy signals
- Optimization becomes unstable

---

## ImageNet Mean & Std (Very Important)

Pretrained CNNs (ResNet, VGG, etc.) expect inputs normalized using:

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


These statistics come from the ImageNet dataset.

### Key Insight:
- Using ImageNet normalization helps align your data
  with pretrained feature distributions
- Using wrong statistics can harm transfer learning

---

## Dataset-Specific Normalization

When training from scratch or on specialized domains:
- Medical imaging
- Satellite imagery
- Industrial vision

It is often better to:
- Compute dataset-specific mean & std
- Normalize accordingly

---

## Files in This Folder

├── min_max_normalization.py # Manual Min-Max scaling
├── standardization.py # Z-score normalization
├── normalization_torch.py # PyTorch-based normalization pipelines


Each file contains:
- Clear mathematical intuition
- Clean, reusable code
- Practical usage examples

---

## Learning Outcome

After completing this module, you should be able to:
- Explain normalization intuitively and mathematically
- Choose the correct normalization strategy
- Debug training instabilities caused by preprocessing
- Apply correct normalization in transfer learning setups


Min-Max vs. Standardization
- Min-Max: Data within a fixed interval [0,1] or [-1,1]
- Standardization: Data with mean 0 and standard deviation 1
- Standardization is usually better for models with sensitive conditions and better networks