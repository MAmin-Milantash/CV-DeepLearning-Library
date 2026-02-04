import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        if in_channels != out_channels or stride != 1:
            self.downsample = nn.Conv2d(in_channels, out_channels, 1, stride)
        else:
            self.downsample = None
    
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.downsample:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        print(f"ResidualBlock output shape: {out.shape}")
        return out

# Example usage
if __name__ == "__main__":
    x = torch.randn(4, 3, 32, 32)
    block = ResidualBlock(3, 16, stride=2)
    out = block(x)
