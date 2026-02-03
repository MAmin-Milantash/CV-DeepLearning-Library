import torch
import torch.nn as nn

x = torch.randn(1, 1, 10, 32, 32)  # video / volume

conv = nn.Conv3d(
    in_channels=1,
    out_channels=4,
    kernel_size=3,
    stride=1,
    padding=1
)

y = conv(x)

print("Input shape:", x.shape)
print("Output shape:", y.shape)
