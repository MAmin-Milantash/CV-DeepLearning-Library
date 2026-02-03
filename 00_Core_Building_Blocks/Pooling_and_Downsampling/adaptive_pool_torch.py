import torch
import torch.nn as nn

x = torch.randn(1, 512, 13, 17)

adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
out = adaptive_pool(x)

print("Input shape:", x.shape)
print("Output shape:", out.shape)
