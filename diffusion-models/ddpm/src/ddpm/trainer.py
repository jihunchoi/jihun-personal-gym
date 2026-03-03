# DDPM trainer.

from pathlib import Path

import torch
from diffusion_common.decoders.base import Decoder
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ddpm.forward import DDPMForwardProcess


class DDPMTrainer:
    def __init__(
        self,
        noise_prediction_model: Decoder,
        forward_process: DDPMForwardProcess,
    ) -> None:
        self.noise_prediction_model = noise_prediction_model
        self.forward_process = forward_process
        self.loss = nn.MSELoss()

    def compute_loss(self, x_0, t) -> torch.Tensor:
        x_t, noise = self.forward_process(x_0=x_0, t=t)
        predicted_noise = self.noise_prediction_model(x=x_t, t=t)
        loss = self.loss(predicted_noise, noise)
        return loss

    def save_checkpoint(
        self, epoch: int, optimizer: torch.optim.Optimizer, checkpoint_dir: Path
    ) -> None:
        """Saves a checkpoint to the specified directory."""
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.noise_prediction_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    def train(
        self,
        dataloader: DataLoader[Dataset],
        optimizer: torch.optim.Optimizer,
        num_epochs: int,
        device: torch.device = "cpu",
        checkpoint_dir: str | Path | None = None,
        checkpoint_freq: int = 1,
    ) -> None:
        self.noise_prediction_model.to(device)
        self.forward_process.to(device)

        if checkpoint_dir is not None:
            checkpoint_dir = Path(checkpoint_dir)

        for epoch in range(num_epochs):
            pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
            for x_0, _ in pbar:
                x_0 = x_0.to(device)
                t = torch.randint(
                    low=0,
                    high=len(self.forward_process.betas),
                    size=(x_0.shape[0],),
                    device=device,
                )
                loss = self.compute_loss(x_0, t)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update progress bar with loss and MPS memory usage if applicable
                postfix = {"loss": f"{loss.item():.4f}"}
                pbar.set_postfix(postfix)

            if checkpoint_dir is not None and (epoch + 1) % checkpoint_freq == 0:
                self.save_checkpoint(epoch, optimizer, checkpoint_dir)
