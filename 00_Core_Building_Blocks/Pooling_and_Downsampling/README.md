# Pooling & Downsampling

This folder covers pooling operations used in Convolutional Neural Networks (CNNs).
Pooling layers reduce spatial dimensions while preserving important features.

## Why Pooling Matters
- Reduces computational cost
- Controls overfitting
- Introduces translation invariance
- Expands effective receptive field

## Covered Topics
- Max Pooling (from scratch)
- Average Pooling (from scratch)
- Pooling layers in PyTorch
- Adaptive Pooling (output-size driven pooling)

## Pooling Parameters
- Kernel size
- Stride
- Padding (rarely used in pooling)

## Types of Pooling
| Type | Description |
|-----|-------------|
| Max Pooling | Keeps strongest activation => Important |
| Average Pooling | Smooths features |
| Adaptive Pooling | Output-size controlled pooling |

## Interview Notes
- Pooling reduces spatial resolution (set on W and H), not channel depth
- Max pooling introduces non-linearity
- Max Pooling → selective & aggressive
- Pooling increases robustness but may lose precise localization
- Avg Pooling → smooth
- Adaptive pooling is often used before fully connected layers
- Adaptive Pooling makes the network independent of input size
- Adaptive Pooling Widely used in ResNet, EfficientNet


| Notion         | Result                  |
| ------------- | ---------------------- |
| Pooling       | reduce Dimention            |
| Max Pool      | select strong feature      |
| Avg Pool      | smoothing              |
| Adaptive Pool | Flexible architecture    |
| Downsampling  | increase receptive field |
