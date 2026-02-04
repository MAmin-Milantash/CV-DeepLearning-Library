import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Simulated feature matrix
X = torch.randn(100, 50).numpy()

# t-SNE to 2D
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
X_tsne = tsne.fit_transform(X)

plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
plt.title("t-SNE of Features")
plt.show()