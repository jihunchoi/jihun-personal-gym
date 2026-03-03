import torch
import pytest
from ddpm.sampling import DDPMSampler
from diffusion_common.decoders.base import Decoder

class MockDecoder(Decoder):
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Check if t has the same batch size as x
        assert t.shape[0] == x.shape[0], f"t.shape[0] ({t.shape[0]}) != x.shape[0] ({x.shape[0]})"
        return torch.zeros_like(x)

def test_ddpm_sampler_batch_size():
    betas = [0.1, 0.2, 0.3]
    decoder = MockDecoder()
    sampler = DDPMSampler(betas, decoder)
    
    batch_size = 8
    x_t = torch.randn(batch_size, 3, 32, 32)
    
    # Test with tensor t of shape (1,)
    x_prev, noise = sampler(x_t, torch.tensor([1]))
    assert x_prev.shape == x_t.shape
    assert noise.shape == x_t.shape
    
    # Test with tensor t of shape (batch_size,)
    x_prev, noise = sampler(x_t, torch.ones(batch_size, dtype=torch.long))
    assert x_prev.shape == x_t.shape

def test_ddpm_sampler_t_zero():
    betas = [0.1, 0.2, 0.3]
    decoder = MockDecoder()
    sampler = DDPMSampler(betas, decoder)
    
    x_t = torch.randn(1, 3, 32, 32)
    
    # Test with t=0
    x_prev, noise = sampler(x_t, torch.tensor([0]))
    assert x_prev.shape == x_t.shape

def test_ddpm_sampler_noise_scaling():
    # Set up a sampler where we can predict the noise scaling
    # If x_t = 0 and predicted_noise = 0:
    # x_{t-1} = sqrt(squared_sigmas[t]) * true_noise
    # Standard deviation of x_{t-1} should be sqrt(squared_sigmas[t])
    
    betas = [0.1, 0.5, 0.9]
    decoder = MockDecoder()
    sampler = DDPMSampler(betas, decoder)
    
    # We need a large batch to get a stable standard deviation
    batch_size = 10000
    x_t = torch.zeros(batch_size, 1, 1, 1)
    t_val = 2 # index 2
    t = torch.tensor([t_val])
    
    x_prev, _ = sampler(x_t, t)
    
    expected_std = torch.sqrt(sampler.squared_sigmas[t_val])
    actual_std = x_prev.std()
    
    # Check if they are close
    assert torch.allclose(actual_std, expected_std, atol=0.05)

def test_ddpm_sampler_invalid_t():
    betas = [0.1, 0.2, 0.3]
    decoder = MockDecoder()
    sampler = DDPMSampler(betas, decoder)
    
    x_t = torch.randn(1, 3, 32, 32)
    
    with pytest.raises(ValueError):
        sampler(x_t, torch.tensor([-1]))
    
    with pytest.raises(ValueError):
        sampler(x_t, torch.tensor([3])) # len(betas) is 3, indices are 0, 1, 2

    # Invalid type
    with pytest.raises(TypeError, match="t must be a torch.Tensor"):
        sampler(x_t, 1)
