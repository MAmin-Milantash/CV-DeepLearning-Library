import torch
import torch.nn as nn

class InceptionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.branch1 = nn.Conv2d(in_channels, 16, 1)
        self.branch3 = nn.Conv2d(in_channels, 16, 3, padding=1)
        self.branch5 = nn.Conv2d(in_channels, 16, 5, padding=2)
        self.branch_pool = nn.Conv2d(in_channels, 16, 1)
    
    def forward(self, x):
        b1 = self.branch1(x)
        b3 = self.branch3(x)
        b5 = self.branch5(x)
        pool = self.branch_pool(F.max_pool2d(x, 3, stride=1, padding=1))
        
        out = torch.cat([b1, b3, b5, pool], dim=1)
        print(f"InceptionBlock output shape: {out.shape}")
        return out

# Example usage
if __name__ == "__main__":
    x = torch.randn(4, 3, 32, 32)
    block = InceptionBlock(3)
    out = block(x)