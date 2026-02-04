import torch
import torch.nn as nn
import torch.optim as optim


# =========================
# Simple Model for Demo
# =========================
class SimpleModel(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=32, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


# =========================
# Optimizer Factory
# =========================
def get_optimizer(name, model, lr=1e-3, weight_decay=0.0):
    """
    Returns a PyTorch optimizer based on name.
    """
    name = name.lower()

    if name == "sgd":
        return optim.SGD(model.parameters(), lr=lr)

    elif name == "sgd_momentum":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    elif name == "rmsprop":
        return optim.RMSprop(model.parameters(), lr=lr)

    elif name == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    elif name == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    else:
        raise ValueError(f"Unknown optimizer: {name}")


# =========================
# Training Step
# =========================
def train_step(model, optimizer, criterion, x, y):
    model.train()

    optimizer.zero_grad()
    preds = model(x)
    loss = criterion(preds, y)
    loss.backward()
    optimizer.step()

    return loss.item()


# =========================
# Demo / Comparison
# =========================
if __name__ == "__main__":
    torch.manual_seed(42)

    # Dummy data
    x = torch.randn(64, 10)
    y = torch.randn(64, 1)

    criterion = nn.MSELoss()

    optimizer_names = [
        "sgd",
        "sgd_momentum",
        "rmsprop",
        "adam",
        "adamw"
    ]

    for opt_name in optimizer_names:
        print(f"\nOptimizer: {opt_name.upper()}")

        model = SimpleModel()
        optimizer = get_optimizer(
            name=opt_name,
            model=model,
            lr=1e-2,
            weight_decay=1e-4
        )

        for step in range(5):
            loss = train_step(model, optimizer, criterion, x, y)
            print(f"Step {step + 1}: Loss = {loss:.4f}")
