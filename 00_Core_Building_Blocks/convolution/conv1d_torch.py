import torch
import torch.nn as nn

x = torch.randn(1, 1, 10)  # (batch, channels, length)

conv = nn.Conv1d(
    in_channels=1,
    out_channels=2,
    kernel_size=3,
    stride=1,
    padding=1
)

y = conv(x)

print("Input shape:", x.shape)
print("Output shape:", y.shape)
