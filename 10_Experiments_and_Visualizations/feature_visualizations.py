import matplotlib.pyplot as plt
import torch

def visualize_feature_map(feature_map, n_cols=8):
    """Visualize feature map from Conv or Attention layers."""
    n_features = feature_map.shape[1]
    n_rows = (n_features + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*2, n_rows*2))
    axes = axes.flatten()
    for i in range(n_features):
        axes[i].imshow(feature_map[0, i].cpu().detach(), cmap='viridis')
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

# Example usage
# feature_map = model.conv_block(images)  # output shape: [batch, channels, H, W]
# visualize_feature_map(feature_map)