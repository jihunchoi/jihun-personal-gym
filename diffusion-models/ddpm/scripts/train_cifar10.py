# Train a DDPM model for the CIFAR-10 dataset.

import argparse

import torch
import torch.optim as optim
from diffusion_common import ResUNet
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from ddpm.forward import DDPMForwardProcess
from ddpm.trainer import DDPMTrainer


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Maximum number of epochs to train"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="../datasets",
        help="Directory where the CIFAR-10 dataset is located or to be downloaded",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints",
        help="Directory to save training checkpoints",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training (e.g., 'cpu', 'cuda', 'mps')",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=1000,
        help="Number of diffusion steps in the forward process",
    )
    parser.add_argument(
        "--batch-size", type=int, default=512, help="Batch size for training"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")

    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # 1. Prepare Dataset and DataLoader
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Scale to [-1, 1]
        ]
    )

    dataset = datasets.CIFAR10(
        root=args.data_dir, train=True, download=True, transform=transform
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=2
    )

    # 2. Initialize Model and Forward Process
    # Hyperparameters for the ResUNet model
    input_dim = 3
    output_dim = 3
    layer_dims = [64, 128, 256]
    bridge_dim = 512
    time_embedding_dim = 128

    noise_prediction_model = ResUNet(
        input_dim=input_dim,
        output_dim=output_dim,
        layer_dims=layer_dims,
        bridge_dim=bridge_dim,
        time_embedding_dim=time_embedding_dim,
    )

    # DDPM Linear schedule (example: 1000 steps)
    num_steps = args.num_steps
    betas = torch.linspace(0.01, 0.02, num_steps).tolist()
    forward_process = DDPMForwardProcess(betas)

    # 3. Setup Trainer and Optimizer
    trainer = DDPMTrainer(noise_prediction_model, forward_process)
    optimizer = optim.Adam(noise_prediction_model.parameters(), lr=args.lr)

    # 4. Start Training
    print("Starting training...")
    trainer.train(
        dataloader=dataloader,
        optimizer=optimizer,
        num_epochs=args.epochs,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_freq=1,
    )
    print("Training complete.")


if __name__ == "__main__":
    main()
