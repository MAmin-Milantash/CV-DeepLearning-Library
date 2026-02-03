import torch
import torch.nn as nn

def init_weights(layer, method="xavier"):
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

# Example usage
if __name__ == "__main__":
    fc = nn.Linear(5, 3)
    conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)

    init_weights(fc, method="xavier")
    init_weights(conv, method="he")

    print("FC weights:\n", fc.weight)
    print("Conv weights shape:", conv.weight.shape)
