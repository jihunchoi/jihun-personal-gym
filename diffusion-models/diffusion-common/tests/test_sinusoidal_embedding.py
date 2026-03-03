import torch
import pytest
from diffusion_common.utils.sinusoidal_embedding import SinusoidalEmbedding


def test_sinusoidal_embedding_shape():
    dim = 64
    embedder = SinusoidalEmbedding(dim=dim)
    t = torch.tensor([1, 2, 3, 4, 5])

    emb = embedder(t)

    # (batch_size, dim)
    assert emb.shape == (5, dim)


def test_sinusoidal_embedding_scalar():
    dim = 32
    embedder = SinusoidalEmbedding(dim=dim)
    t = torch.tensor(10)

    emb = embedder(t)
    assert emb.shape == (1, dim)


def test_sinusoidal_embedding_batch_shape():
    dim = 32
    embedder = SinusoidalEmbedding(dim=dim)
    batch_size = 4
    t = torch.randint(0, 100, (batch_size,))

    emb = embedder(t)
    assert emb.shape == (batch_size, dim)


def test_invalid_dim():
    with pytest.raises(AssertionError):
        SinusoidalEmbedding(dim=31)


def test_consistency_at_zero():
    # Sin(0) = 0, Cos(0) = 1
    dim = 16
    embedder = SinusoidalEmbedding(dim=dim)
    t = torch.zeros((1,))
    emb = embedder(t)

    # emb[:dim//2] are sin, emb[dim//2:] are cos
    assert torch.allclose(emb[:, : dim // 2], torch.zeros_like(emb[:, : dim // 2]))
    assert torch.allclose(emb[:, dim // 2 :], torch.ones_like(emb[:, dim // 2 :]))


def test_magnitude():
    # Each pair of sin/cos for the same frequency should have a norm of 1
    # sin^2 + cos^2 = 1
    dim = 10
    embedder = SinusoidalEmbedding(dim=dim)
    t = torch.randint(0, 1000, (10,)).float()
    emb = embedder(t)

    # Sin and Cos are concatenated. Let's check:
    sin_part = emb[:, : dim // 2]
    cos_part = emb[:, dim // 2 :]

    # (sin^2 + cos^2) should be 1
    magnitude = sin_part**2 + cos_part**2
    assert torch.allclose(magnitude, torch.ones_like(magnitude))
