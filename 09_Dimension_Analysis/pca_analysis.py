import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Simulated feature matrix: 100 samples, 50 features
X = torch.randn(100, 50).numpy()

# Apply PCA to reduce to 2 dimensions
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print("Explained variance ratio:", pca.explained_variance_ratio_)

# Plot
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.title("PCA of Features")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()