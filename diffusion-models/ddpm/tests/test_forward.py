import torch
import pytest
from ddpm.forward import DDPMForwardProcess


def test_initialization():
    betas = [0.1, 0.2, 0.3]
    forward_process = DDPMForwardProcess(betas)

    assert torch.allclose(forward_process.betas, torch.tensor(betas))
    assert forward_process.alphas.shape == torch.Size([3])
    assert forward_process.alpha_bars.shape == torch.Size([3])
    # alpha_bar[0] = alpha[0]
    assert torch.allclose(forward_process.alpha_bars[0], forward_process.alphas[0])
    # alpha_bar[1] = alpha[0] * alpha[1]
    assert torch.allclose(
        forward_process.alpha_bars[1],
        forward_process.alphas[0] * forward_process.alphas[1],
    )


def test_sample_noise():
    betas = [0.1, 0.2]
    forward_process = DDPMForwardProcess(betas)
    shape = (10, 3, 32, 32)
    noise = forward_process.sample_noise(shape)

    assert noise.shape == shape
    # Basic statistical check for Gaussian noise
    assert torch.abs(noise.mean()) < 0.1
    assert torch.abs(noise.std() - 1.0) < 0.1


def test_add_noise_scalar_t():
    betas = [0.1, 0.5]
    forward_process = DDPMForwardProcess(betas)
    batch_size = 4
    x_0 = torch.ones(batch_size, 3, 8, 8)
    noise = torch.randn_like(x_0)
    t = torch.tensor([1, 1, 1, 1], dtype=torch.long)

    x_t = forward_process.add_noise(x_0, noise, t)

    alpha_bar_t = forward_process.alpha_bars[t]
    # Reshape alpha_bar_t for broadcasting
    alpha_bar_t_reshaped = alpha_bar_t.view(-1, 1, 1, 1)
    expected_x_t = (
        alpha_bar_t_reshaped * x_0 + torch.sqrt(1 - alpha_bar_t_reshaped**2) * noise
    )

    assert torch.allclose(x_t, expected_x_t)
    assert x_t.shape == x_0.shape


def test_add_noise_tensor_t():
    betas = [0.1, 0.2, 0.3, 0.4]
    forward_process = DDPMForwardProcess(betas)
    batch_size = 4
    x_0 = torch.randn(batch_size, 3, 8, 8)
    noise = torch.randn_like(x_0)
    t = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    x_t = forward_process.add_noise(x_0, noise, t)

    for i in range(batch_size):
        alpha_bar_i = forward_process.alpha_bars[t[i]]
        expected_x_t_i = (
            alpha_bar_i * x_0[i] + torch.sqrt(1 - alpha_bar_i**2) * noise[i]
        )
        assert torch.allclose(x_t[i], expected_x_t_i)


def test_add_noise_invalid_t_shape():
    betas = [0.1, 0.2]
    forward_process = DDPMForwardProcess(betas)
    x_0 = torch.randn(4, 3, 8, 8)
    noise = torch.randn_like(x_0)

    # Mismatched batch size
    with pytest.raises(ValueError, match="t must be a 1D tensor with shape"):
        forward_process.add_noise(x_0, noise, torch.tensor([0, 1]))

    # Too many dimensions
    with pytest.raises(ValueError, match="t must be a 1D tensor with shape"):
        forward_process.add_noise(x_0, noise, torch.zeros((2, 2), dtype=torch.long))

    # Invalid type
    with pytest.raises(TypeError, match="t must be a torch.Tensor"):
        forward_process.add_noise(x_0, noise, 1)


def test_forward_method():
    betas = [0.1, 0.2]
    forward_process = DDPMForwardProcess(betas)
    batch_size = 4
    x_0 = torch.randn(batch_size, 3, 8, 8)
    t = torch.tensor([1, 1, 1, 1], dtype=torch.long)

    x_t, noise = forward_process(x_0, t)

    assert x_t.shape == x_0.shape
    assert noise.shape == x_0.shape

    # Verify the relationship
    alpha_bar_t = forward_process.alpha_bars[t].view(-1, 1, 1, 1)
    expected_x_t = alpha_bar_t * x_0 + torch.sqrt(1 - alpha_bar_t**2) * noise
    assert torch.allclose(x_t, expected_x_t)
