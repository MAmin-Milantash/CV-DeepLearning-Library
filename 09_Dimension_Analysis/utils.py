def print_shape(tensor, name="Tensor"):
    print(f"{name} shape: {tensor.shape}")

def assert_shape(tensor, expected_shape, name="Tensor"):
    assert tensor.shape == expected_shape, f"{name} shape mismatch! Expected {expected_shape}, got {tensor.shape}"
    print(f"{name} shape verified: {tensor.shape}")

def log_shapes(tensors, names=None):
    if names is None:
        names = [f"Tensor{i}" for i in range(len(tensors))]
    for t, n in zip(tensors, names):
        print_shape(t, n)