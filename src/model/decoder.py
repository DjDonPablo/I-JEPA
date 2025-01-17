import torch
import torch.nn as nn


# FIXME: this class is NOT good at all. It's just a placeholder.
class PatchDecoder(nn.Module):
    """
    Maps (B, embed_dim) -> (B, 3, patch_size, patch_size).
    A simple MLP for demonstration.
    """
    def __init__(self, embed_dim, patch_size=8):
        super().__init__()
        self.patch_size = patch_size
        out_dim = 3 * patch_size * patch_size
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, out_dim)
        )

    def forward(self, z):
        # z: (B, embed_dim)
        x = self.net(z)  # => (B, out_dim)
        B, outdim = x.shape
        x = x.view(B, 3, self.patch_size, self.patch_size)
        return x  # => (B,3,p,p)