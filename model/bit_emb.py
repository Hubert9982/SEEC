import torch
import torch.nn as nn


class BitEmb(nn.Module):
    """Learned global conditioning embedding used by RAWIC."""

    def __init__(self, num_bits: int, emb_dim: int, out_dim: int):
        super().__init__()
        self.num_bits = num_bits
        self.embedding = nn.Sequential(
            nn.Embedding(num_bits, emb_dim),
            nn.Linear(emb_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, bit_depth: torch.Tensor) -> torch.Tensor:
        embedding = self.embedding(bit_depth.to(torch.long))
        return embedding[:, :, None, None]
