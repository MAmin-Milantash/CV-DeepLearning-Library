import torch
import torch.nn as nn

def apply_weight_init(model, method="xavier"):
    """Apply weight initialization to all layers in a model"""
    for layer in model.modules():
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            if method == "zero":
                nn.init.zeros_(layer.weight)
            elif method == "normal":
                nn.init.normal_(layer.weight, mean=0.0, std=0.01)
            elif method == "uniform":
                nn.init.uniform_(layer.weight, a=-0.01, b=0.01)
            elif method == "xavier":
                nn.init.xavier_uniform_(layer.weight)
            elif method == "he":
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

def print_weight_stats(model):
    """Print basic statistics of weights"""
    for layer in model.modules():
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            w = layer.weight.data
            print(f"{layer}: mean={w.mean():.4f}, std={w.std():.4f}, min={w.min():.4f}, max={w.max():.4f}")
