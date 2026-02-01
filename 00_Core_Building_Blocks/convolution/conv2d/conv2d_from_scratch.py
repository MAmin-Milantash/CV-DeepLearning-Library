import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2DFromScratch(nn.Module):
    """
    Custom implementation of 2D convolution from scratch
    using PyTorch tensor operations.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        """
        x shape: (batch_size, in_channels, height, width)
        """
        batch_size, _, h, w = x.shape

        x = F.pad(
            x,
            (self.padding, self.padding, self.padding, self.padding)
        )

        h_out = (h + 2 * self.padding - self.kernel_size) // self.stride + 1
        w_out = (w + 2 * self.padding - self.kernel_size) // self.stride + 1

        output = torch.zeros(batch_size, self.out_channels, h_out, w_out)

        for b in range(batch_size):
            for oc in range(self.out_channels):
                for i in range(h_out):
                    for j in range(w_out):
                        h_start = i * self.stride
                        w_start = j * self.stride

                        region = x[b, :, h_start:h_start+self.kernel_size, w_start:w_start+self.kernel_size]
                        output[b, oc, i, j] = torch.sum(region * self.weight[oc]) + self.bias[oc]

        return output
