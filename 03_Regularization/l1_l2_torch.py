import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Linear(10, 1)

# L2 regularization via weight_decay
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

criterion = nn.MSELoss()

x = torch.randn(8, 10)
y = torch.randn(8, 1)

# Forward
y_pred = model(x)
loss = criterion(y_pred, y)

# L1 manual regularization
l1_lambda = 1e-5
l1_norm = sum(p.abs().sum() for p in model.parameters())
loss = loss + l1_lambda * l1_norm

loss.backward()
optimizer.step()


""" 
📌 Tip:
    In PyTorch, L2 usually comes from optimizer, not loss.
"""