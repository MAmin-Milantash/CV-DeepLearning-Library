import torch
import torch.nn as nn
import torch.optim as optim

# Example: simple MLP Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, input_dim=32, latent_dim=16, overcomplete=False):
        super().__init__()
        hidden_dim = latent_dim*2 if overcomplete else latent_dim//2
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

# Data
X = torch.randn(100, 32)
model = Autoencoder(input_dim=32, latent_dim=16, overcomplete=False)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(50):
    optimizer.zero_grad()
    X_hat = model(X)
    loss = criterion(X_hat, X)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")