import torch
import torch.nn as nn
import torch.optim as optim

class DenoisingAE(nn.Module):
    def __init__(self, input_dim=32, latent_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, latent_dim), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(latent_dim, input_dim), nn.Sigmoid())
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

# Add noise
X = torch.randn(100, 32)
noise = 0.1 * torch.randn_like(X)
X_noisy = X + noise

model = DenoisingAE()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(50):
    optimizer.zero_grad()
    X_hat = model(X_noisy)
    loss = criterion(X_hat, X)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")