import torch
import torch.nn as nn

sigmoid = nn.Sigmoid()
tanh = nn.Tanh()
relu = nn.ReLU()
leaky_relu = nn.LeakyReLU(0.01)
softmax = nn.Softmax(dim=1)

# Example forward pass
x = torch.tensor([[-1.0, 0.0, 1.0]])
print("Sigmoid:", sigmoid(x))
print("Tanh:", tanh(x))
print("ReLU:", relu(x))
print("LeakyReLU:", leaky_relu(x))
print("Softmax:", softmax(x))