import torch
import torch.nn as nn

x = torch.randn(1, 3, 32, 32)  # RGB image

conv = nn.Conv2d(
    in_channels=3,
    out_channels=16,
    kernel_size=3,
    stride=1,
    padding=1
)

y = conv(x)

print("Input shape:", x.shape)
print("Output shape:", y.shape)
