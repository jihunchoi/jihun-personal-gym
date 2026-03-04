# Sampler for the reverse diffusion process.
from collections.abc import Sequence

import torch
from diffusion_common.decoders.base import Decoder
from torch import nn


class DDPMSampler(nn.Module):
    def __init__(self, betas: Sequence[float], noise_predictor: Decoder) -> None:
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

        squared_sigmas = torch.zeros_like(self.betas)
        squared_sigmas[1:] = (
            self.one_minus_squared_alpha_bars[:-1]
            / self.one_minus_squared_alpha_bars[1:]
            * self.betas[1:] ** 2
        )
        self.register_buffer("sigmas", torch.sqrt(squared_sigmas))

        self.noise_predictor = noise_predictor

    def forward(
        self, x_t: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Performs one step of the reverse diffusion process, predicting x_{t-1} from x_t."""
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"t must be a torch.Tensor, but got {type(t)}")

        t = t.to(device=x_t.device, dtype=torch.long)

        if t.ndim == 1:
            if t.shape[0] == 1:
                t = t.expand(x_t.shape[0])
            else:
                assert t.shape[0] == x_t.shape[0]
        else:
            raise ValueError("t must be a 1D tensor with shape (N,) or (1,)")

        if torch.any(t < 0) or torch.any(t >= len(self.betas)):
            raise ValueError(
                f"t must be in range [0, {len(self.betas) - 1}] for reverse diffusion sampling."
            )

        predicted_noise = self.noise_predictor(
            x_t, t
        )  # Predict noise using the noise predictor model

        t_reshaped = t.reshape(-1, 1, 1, 1)  # Reshape for broadcasting
        true_noise = torch.randn_like(x_t)  # Sample true noise for the current step

        x_t_minus_1 = (
            self.alpha_invs[t_reshaped]
            * (
                x_t
                - (
                    self.one_minus_squared_alphas[t_reshaped]
                    / torch.sqrt(self.one_minus_squared_alpha_bars[t_reshaped])
                    * predicted_noise
                )
            )
            + self.sigmas[t_reshaped] * true_noise
        )
        return x_t_minus_1, predicted_noise
