# Receptive Field Theory

## Definition
The receptive field of a neuron is the region of the input image
that influences that neuron.

Early layers:
- Small receptive field
- Capture edges & textures

Deeper layers:
- Large receptive field
- Capture objects & semantics

## Important Concepts

### Theoretical Receptive Field
Calculated mathematically using kernel size, stride, and dilation.

### Effective Receptive Field
Actual contribution is Gaussian-like:
- Center pixels matter more
- Borders contribute less

## Why Receptive Field Grows
- Stacking convolutions increases context
- Pooling and stride accelerate growth

## Common Misconception
❌ Kernel size = receptive field  
✅ Receptive field grows layer by layer

## Interview Question
Q: Why use multiple 3×3 convolutions instead of one 7×7?
A:
- Same receptive field
- Fewer parameters
- More non-linearity
