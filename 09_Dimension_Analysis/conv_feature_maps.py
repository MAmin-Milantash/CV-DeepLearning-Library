import torch
import torch.nn.functional as F

def conv2d_output_shape(H_in, W_in, kernel_size, stride=1, padding=0, dilation=1):
    """
    Compute output shape of Conv2D layer
    """
    H_out = ((H_in + 2*padding - dilation*(kernel_size-1) - 1)//stride) + 1
    W_out = ((W_in + 2*padding - dilation*(kernel_size-1) - 1)//stride) + 1
    return H_out, W_out

def print_conv_feature_map(x, conv_layer, name="Conv"):
    """
    Apply conv layer and print resulting feature map shape
    """
    out = conv_layer(x)
    print(f"{name} output shape: {out.shape}")
    return out

# Example:
if __name__ == "__main__":
    x = torch.randn(8, 3, 32, 32)  # batch_size=8, channels=3, 32x32 image
    conv = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
    out = print_conv_feature_map(x, conv)
