import torch

def flatten_tensor(x):
    batch_size = x.shape[0]
    return x.view(batch_size, -1)

def reshape_tensor(x, shape):
    return x.view(shape)

# Example
if __name__ == "__main__":
    x = torch.randn(4, 3, 32, 32)
    flat = flatten_tensor(x)
    print(f"Flattened shape: {flat.shape}")
    
    reshaped = reshape_tensor(flat, (4, 3, 32, 32))
    print(f"Reshaped shape: {reshaped.shape}")