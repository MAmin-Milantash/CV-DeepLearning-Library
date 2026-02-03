# Dimension Calculation Formulae

## Convolution Output Size

For each spatial dimension:

Output = ⌊ (N + 2P − K) / S ⌋ + 1

Where:
- N = input size
- K = kernel size
- S = stride
- P = padding

## Pooling Output Size
Same formula as convolution.

## Channel Dimension
- Convolution: determined by number of filters
- Pooling: unchanged

## Padding Types
- Valid: P = 0
- Same: P = (K − 1) / 2

## Common Interview Trap
Stride reduces spatial size much faster than pooling.
