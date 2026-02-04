import torch

def count_parameters(model):
    """
    Count trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_shapes(x, model):
    """
    Print intermediate feature map shapes for debugging.
    """
    out = x
    for layer in model.children():
        out = layer(out)
        print(out.shape)
