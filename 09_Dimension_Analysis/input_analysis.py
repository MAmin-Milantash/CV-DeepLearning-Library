import torch

def print_input_shape(x, name="Input"):
    """
    Print the shape of the input tensor
    Args:
        x (torch.Tensor): input tensor
        name (str): optional name for tensor
    """
    print(f"{name} shape: {x.shape}")

def summarize_input_batch(x):
    """
    Summarize batch info: batch_size, channels, height, width
    Supports 2D, 3D, and 4D inputs
    """
    if x.dim() == 2:
        # MLP input: (batch_size, features)
        batch_size, features = x.shape
        print(f"MLP Input: batch_size={batch_size}, features={features}")
    elif x.dim() == 3:
        # sequence: (batch_size, seq_len, features)
        batch_size, seq_len, features = x.shape
        print(f"Sequence Input: batch_size={batch_size}, seq_len={seq_len}, features={features}")
    elif x.dim() == 4:
        # image: (batch_size, channels, H, W)
        batch_size, channels, H, W = x.shape
        print(f"Image Input: batch_size={batch_size}, channels={channels}, H={H}, W={W}")
    elif x.dim() == 5:
        # video: (batch_size, channels, D, H, W)
        batch_size, channels, D, H, W = x.shape
        print(f"Video Input: batch_size={batch_size}, channels={channels}, D={D}, H={H}, W={W}")
    else:
        print(f"Unknown input shape: {x.shape}")
