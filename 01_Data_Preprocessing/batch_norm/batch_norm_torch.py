import torch
import torch.nn as nn

# Example for 1D features (Dense)
batch_norm1d = nn.BatchNorm1d(num_features=128)

# Example for 2D features (Convolution)
batch_norm2d = nn.BatchNorm2d(num_features=64)  # channels in Conv layer

# Example for 3D features (Video / 3D Conv)
batch_norm3d = nn.BatchNorm3d(num_features=32)

# Forward pass
x = torch.randn(16, 64, 32, 32)  # batch_size, channels, H, W
out = batch_norm2d(x)
print(out.shape)
