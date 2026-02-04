import torch
import torch.nn as nn
import torch.optim as optim

# Simple model with optional dropout and L2
class RegularizedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.layers(x)

# Training with L2
model = RegularizedMLP(28*28, 128, 10, dropout=0.2)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # L2
criterion = nn.CrossEntropyLoss()