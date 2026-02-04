import torch
import torch.nn as nn
import torch.optim as optim

class SparseAE(nn.Module):
    def __init__(self, input_dim=32, latent_dim=16, rho=0.05, beta=1e-3):
        super().__init__()
        self.rho = rho
        self.beta = beta
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)
        self.activation = nn.Sigmoid()
    def forward(self, x):
        z = self.activation(self.encoder(x))
        out = self.decoder(z)
        return out
    def sparse_loss(self, x, x_hat):
        mse = nn.MSELoss()(x_hat, x)
        z = self.activation(self.encoder(x))
        rho_hat = z.mean(0)
        kl = torch.sum(self.rho*torch.log(self.rho/rho_hat) + (1-self.rho)*torch.log((1-self.rho)/(1-rho_hat)))
        return mse + self.beta*kl

X = torch.randn(100, 32)
model = SparseAE()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(50):
    optimizer.zero_grad()
    X_hat = model(X)
    loss = model.sparse_loss(X, X_hat)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")