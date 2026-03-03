from abc import ABC, abstractmethod
import torch
from torch import nn


class Decoder(nn.Module, ABC):
    """
    Abstract base class for all decoder models in the diffusion process.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the decoder.

        Args:
            x: Input tensor (noisy image) of shape (B, C, H, W).
            t: Timestep tensor of shape (B,).

        Returns:
            Predicted noise or denoised image tensor of shape (B, C, H, W).
        """
        pass
