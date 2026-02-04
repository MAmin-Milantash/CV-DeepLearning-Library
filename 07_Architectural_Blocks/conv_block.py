import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """
    Conv → BatchNorm → Activation → optional Dropout
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 activation='relu', use_bn=True, dropout_prob=0.0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.activation = nn.ReLU() if activation=='relu' else nn.Identity()
        self.dropout = nn.Dropout2d(dropout_prob) if dropout_prob>0 else nn.Identity()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x
