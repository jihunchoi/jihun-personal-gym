import torch
from torch import nn, Tensor


class SinusoidalEmbedding(nn.Module):
    """
    Sinusoidal Positional Embedding for temporal conditioning in diffusion models.
    Reference: https://arxiv.org/abs/1706.03762 (Attention is All You Need)
    Used commonly in DDPM for time embeddings.
    """

    inv_freq: Tensor

    def __init__(self, dim: int, base: int = 10000):
        """
        Initializes the embedding module.

        Args:
            dim: Dimension of the embeddings (must be even).
            base: Base for the frequency calculation (default 10000).
        """
        super().__init__()
        assert dim % 2 == 0, "Embedding dimension must be even."
        self.dim = dim
        self.base = base

        # Create inverse frequencies: 1 / (base ** (2i / dim))
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, t: Tensor) -> Tensor:
        """
        Generates sinusoidal embeddings for the given timesteps.

        Args:
            t: Tensor of timesteps with shape (batch_size,) or (batch_size, 1).

        Returns:
            Embedding tensor of shape (batch_size, dim).
        """
        if t.ndim == 0:
            t = t.unsqueeze(0)
        if t.ndim == 1:
            t = t[:, None]

        # t: (batch_size, 1), inv_freq: (dim // 2)
        # freqs: (batch_size, dim // 2)
        freqs = t.float() * self.inv_freq[None, :]

        # Concatenate sin and cos components: (batch_size, dim)
        emb = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        return emb
