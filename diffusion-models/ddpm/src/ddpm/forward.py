# Forward process of DDPM.
from collections.abc import Sequence
import torch
from torch import nn


class DDPMForwardProcess(nn.Module):
    def __init__(self, betas: Sequence[float]) -> None:
        super().__init__()

        self.register_buffer("betas", torch.tensor(betas, dtype=torch.float32))
        self.register_buffer("alphas", torch.sqrt(1.0 - self.betas**2))
        self.register_buffer("alpha_bars", torch.cumprod(self.alphas, dim=0))

    def sample_noise(
        self, shape: torch.Size, device: torch.device = "cpu"
    ) -> torch.Tensor:
        """Generates Gaussian noise from N(0, I)"""
        return torch.randn(shape, device=device)

    def add_noise(
        self, x_0: torch.Tensor, noise: torch.Tensor, t: int | torch.Tensor
    ) -> torch.Tensor:
        """
        Adds noise to x_0 according to the schedule:
        x_t = self.alpha_bars[t] * x_0 + sqrt(1 - alpha_bar_t**2) * noise

        Args:
            x_0: Original image tensor (N, C, H, W)
            noise: Gaussian noise tensor (N, C, H, W)
            t: Integer or tensor step index,
                or tensor of shape (N,) with step indices for each sample in the batch.

        Returns:
            x_t: Noisy image tensor (N, C, H, W)
        """
        device = x_0.device
        alpha_bars = self.alpha_bars.to(device)

        t = torch.as_tensor(t, device=device)

        if t.ndim == 0:
            t = t.unsqueeze(0)  # Convert scalar to tensor of shape (1,)
        elif t.ndim == 1:
            assert t.shape[0] == x_0.shape[0]
        else:
            raise ValueError("t must be a scalar or a 1D tensor with shape (N,)")

        alpha_bar_reshaped = alpha_bars[t].view(-1, 1, 1, 1)  # Reshape for broadcasting
        x_t = alpha_bar_reshaped * x_0 + torch.sqrt(1 - alpha_bar_reshaped**2) * noise

        return x_t

    def forward(
        self, x_0: torch.Tensor, t: int | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Samples noise and applies the forward process to x_0 at step t.

        Args:
            x_0: Original image tensor (N, C, H, W)
            t: Integer or tensor step index,
                or tensor of shape (N,) with step indices for each sample in the batch.

        Returns:
            x_t: Noisy image at step t
            noise: The Gaussian noise that was added
        """
        noise = self.sample_noise(x_0.shape, device=x_0.device)
        x_t = self.add_noise(x_0, noise, t)
        return x_t, noise
