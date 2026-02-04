import torch
import torch.nn as nn

class InceptionBlock(nn.Module):
    """
    Multi-branch convolution block (Inception-style).
    """
    def __init__(self, in_channels, out1x1, out3x3, out5x5, out_pool):
        super().__init__()
        self.branch1 = nn.Conv2d(in_channels, out1x1, kernel_size=1)
        self.branch3 = nn.Conv2d(in_channels, out3x3, kernel_size=3, padding=1)
        self.branch5 = nn.Conv2d(in_channels, out5x5, kernel_size=5, padding=2)
        self.branch_pool = nn.Conv2d(in_channels, out_pool, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        b1 = self.branch1(x)
        b3 = self.branch3(x)
        b5 = self.branch5(x)
        bp = self.branch_pool(self.pool(x))
        return torch.cat([b1, b3, b5, bp], dim=1)
