# Sampler for the reverse diffusion process.
from collections.abc import Sequence

import torch
from torch import nn


class DDPMSampler(nn.Module):
    def __init__(self, betas: Sequence[float], noise_predictor: nn.Module) -> None:
        """
        Initializes the DDPMSampler with a noise schedule defined by betas.

        Args:
            betas: A list of beta values defining the noise schedule.
        """
        super().__init__()

        self.register_buffer("betas", torch.tensor(betas, dtype=torch.float32))
        self.register_buffer("alphas", torch.sqrt(1.0 - self.betas**2))
        self.register_buffer("alpha_bars", torch.cumprod(self.alphas, dim=0))

        self.register_buffer("alpha_invs", 1.0 / self.alphas)
        self.register_buffer("one_minus_squared_alphas", 1.0 - self.alphas**2)
        self.register_buffer("one_minus_squared_alpha_bars", 1.0 - self.alpha_bars**2)

        indices = torch.arange(len(betas), dtype=torch.long)
        squared_sigmas = torch.zeros_like(indices)
        squared_sigmas[1:] = (
            self.one_minus_squared_alpha_bars[indices[1:] - 1]
            / self.one_minus_squared_alpha_bars[indices[1:]]
            * self.betas[indices[1:]] ** 2
        )
        self.register_buffer("squared_sigmas", squared_sigmas)

        self.noise_predictor = noise_predictor

    def forward(
        self, x_t: torch.Tensor, t: int | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Performs one step of the reverse diffusion process, predicting x_{t-1} from x_t."""
        t = torch.as_tensor(t, device=x_t.device)

        if t.ndim == 0:
            t = t.unsqueeze(0)  # Convert scalar to tensor of shape (1,)
        elif t.ndim == 1:
            assert t.shape[0] == x_t.shape[0]
        else:
            raise ValueError("t must be a scalar or a 1D tensor with shape (N,)")

        if torch.any(t <= 0):
            raise ValueError("t must be greater than 0 for reverse diffusion sampling.")

        t = t.reshape(-1, 1, 1, 1)  # Reshape for broadcasting

        predicted_noise = self.noise_predictor(
            x_t, t
        )  # Predict noise using the noise predictor model
        true_noise = torch.randn_like(x_t)  # Sample true noise for the current step

        x_t_minus_1 = (
            self.alpha_invs[t]
            * (
                x_t
                - (
                    self.one_minus_squared_alphas[t]
                    / torch.sqrt(self.one_minus_squared_alpha_bars[t])
                    * predicted_noise
                )
            )
            + self.squared_sigmas[t] * true_noise
        )
        return x_t_minus_1, predicted_noise
