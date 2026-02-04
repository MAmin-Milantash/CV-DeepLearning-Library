import torch
import torch.nn as nn

class DenseBlock(nn.Module):
    """
    Fully connected block with optional BatchNorm and Dropout.
    """
    def __init__(self, in_features, out_features, activation='relu', use_bn=True, dropout_prob=0.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features) if use_bn else nn.Identity()
        self.activation = nn.ReLU() if activation=='relu' else nn.Identity()
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0 else nn.Identity()
    
    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x
