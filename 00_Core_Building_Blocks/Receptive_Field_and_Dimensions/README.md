# Receptive Field & Dimension Tracking

This folder explains how spatial dimensions and receptive fields evolve
across CNN layers.

Understanding this topic is critical for:
- CNN architecture design
- Debugging shape mismatches
- Interview questions
- Model efficiency and performance

## Covered Topics
- Receptive Field theory
- Dimension calculation formulas
- Manual receptive field computation
- Layer-by-layer CNN dimension tracking
- Practical CNN examples

## Why This Matters
A network does NOT see the whole image at once.
Each neuron sees a local region called its receptive field.

Designing deep networks requires knowing:
- How fast receptive field grows
- How spatial resolution shrinks
- When information is lost

## Key Interview Concepts
- Receptive field vs kernel size
- Effect of stride and pooling
- Effective vs theoretical receptive field
- Why deeper layers see more context
- Receptive field depends on previous stride, not just kernel


| Notion           | result               |
| --------------- | ------------------- |
| Kernel          | local view          |
| Stride          | spatial shrink      |
| Pooling         | fast downsampling   |
| Depth           | context growth      |
| Receptive Field | how much model sees |
