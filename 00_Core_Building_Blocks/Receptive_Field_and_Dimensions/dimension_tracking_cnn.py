def compute_output_size(n, k, s, p):
    return (n + 2*p - k) // s + 1


if __name__ == "__main__":
    size = 224
    layers = [
        {"k": 7, "s": 2, "p": 3},
        {"k": 3, "s": 2, "p": 1},
        {"k": 3, "s": 1, "p": 1},
    ]

    for i, layer in enumerate(layers):
        size = compute_output_size(size, layer["k"], layer["s"], layer["p"])
        print(f"Layer {i+1} output size: {size}")
