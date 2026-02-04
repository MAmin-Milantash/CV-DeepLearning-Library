import torch
import torch.nn as nn

class AttentionBlock(nn.Module):
    """
    Simple self-attention block for 2D feature maps.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels//8, 1)
        self.key = nn.Conv2d(in_channels, in_channels//8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        B, C, H, W = x.size()
        q = self.query(x).view(B, -1, H*W)           # B x C/8 x N
        k = self.key(x).view(B, -1, H*W)             # B x C/8 x N
        v = self.value(x).view(B, -1, H*W)           # B x C x N

        attn = torch.softmax(torch.bmm(q.transpose(1,2), k), dim=-1)  # B x N x N
        out = torch.bmm(v, attn.transpose(1,2)).view(B, C, H, W)
        return self.gamma*out + x
