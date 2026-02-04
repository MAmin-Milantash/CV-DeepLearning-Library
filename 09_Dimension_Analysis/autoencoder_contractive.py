import torch
import torch.nn as nn
import torch.optim as optim

class ContractiveAE(nn.Module):
    def __init__(self, input_dim=32, latent_dim=16, lam=1e-3):
        super().__init__()
        self.lam = lam
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)
        self.activation = nn.ReLU()
    def forward(self, x):
        z = self.activation(self.encoder(x))
        out = self.decoder(z)
        return out
    def contractive_loss(self, x, x_hat):
        mse = nn.MSELoss()(x_hat, x)
        W = self.encoder.weight
        dz = (self.activation(self.encoder(x)) > 0).float()
        contractive = self.lam * torch.sum((W**2).sum(1) * dz**2)
        return mse + contractive

X = torch.randn(100, 32)
model = ContractiveAE()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(50):
    optimizer.zero_grad()
    X_hat = model(X)
    loss = model.contractive_loss(X, X_hat)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")