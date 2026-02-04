import matplotlib.pyplot as plt

def visualize_attention(attention_weights):
    """attention_weights: [seq_len, seq_len] or [H*W, H*W]"""
    plt.imshow(attention_weights.cpu().detach(), cmap='viridis')
    plt.colorbar()
    plt.title("Attention Map")
    plt.show()