import torch
import torch.nn as nn


class Conv2DPyTorch(nn.Module):
    """
    Wrapper around PyTorch's nn.Conv2d
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

    def forward(self, x):
        return self.conv(x)
