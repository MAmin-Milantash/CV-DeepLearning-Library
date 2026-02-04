import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
    
    def forward(self, x):
        # x: (seq_len, batch_size, embed_dim)
        out, _ = self.attn(x, x, x)
        print(f"SelfAttention output shape: {out.shape}")
        return out

# Example usage
if __name__ == "__main__":
    x = torch.randn(10, 2, 32)  # seq_len=10, batch_size=2, embed_dim=32
    sa = SelfAttention(32, 4)
    out = sa(x)