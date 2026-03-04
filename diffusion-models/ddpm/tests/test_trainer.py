import torch
from diffusion_common.decoders.base import Decoder
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ddpm.forward import DDPMForwardProcess
from ddpm.trainer import DDPMTrainer


class MockDecoder(Decoder):
    def __init__(self):
        super().__init__()
        # Adding a parameter to make it optimizable
        self.param = nn.Parameter(torch.tensor([1.0]))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Predicting noise as a function of the parameter to ensure it's optimizable
        # We'll just return zeros plus the parameter to keep it simple but functional
        return torch.zeros_like(x) + self.param


class SimpleDataset(Dataset):
    def __init__(self, size=8, shape=(3, 32, 32)):
        self.data = torch.randn(size, *shape)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], 0


def test_trainer_initialization():
    betas = [0.1, 0.2, 0.3]
    forward_process = DDPMForwardProcess(betas)
    decoder = MockDecoder()
    trainer = DDPMTrainer(decoder, forward_process)

    assert trainer.noise_prediction_model == decoder
    assert trainer.forward_process == forward_process
    assert isinstance(trainer.loss, nn.MSELoss)


def test_trainer_compute_loss():
    betas = [0.1, 0.2, 0.3]
    forward_process = DDPMForwardProcess(betas)
    decoder = MockDecoder()
    trainer = DDPMTrainer(decoder, forward_process)

    batch_size = 4
    x_0 = torch.randn(batch_size, 3, 32, 32)
    t = torch.tensor([0, 1, 2, 1], dtype=torch.long)

    loss = trainer.compute_loss(x_0, t)

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert loss.item() >= 0

    # Verify gradient can flow
    loss.backward()
    assert decoder.param.grad is not None


def test_trainer_train_step(capsys):
    betas = [0.1, 0.2, 0.3]
    forward_process = DDPMForwardProcess(betas)
    decoder = MockDecoder()
    trainer = DDPMTrainer(decoder, forward_process)

    # Use a small dataset and batch size
    dataset = SimpleDataset(size=4)
    dataloader = DataLoader(dataset, batch_size=2)
    optimizer = torch.optim.SGD(decoder.parameters(), lr=0.1)

    initial_param_value = decoder.param.item()

    # Run for 1 epoch
    trainer.train(dataloader, optimizer, num_epochs=1)

    # Check if parameter was updated
    assert decoder.param.item() != initial_param_value

    # Check output
    captured = capsys.readouterr()
    # tqdm output is in stderr by default
    assert "Epoch 1/1" in captured.err
    assert "loss=" in captured.err


def test_trainer_train_multiple_epochs(capsys):
    betas = [0.1, 0.2]
    forward_process = DDPMForwardProcess(betas)
    decoder = MockDecoder()
    trainer = DDPMTrainer(decoder, forward_process)

    dataset = SimpleDataset(size=2)
    dataloader = DataLoader(dataset, batch_size=2)
    optimizer = torch.optim.SGD(decoder.parameters(), lr=0.1)

    trainer.train(dataloader, optimizer, num_epochs=2)

    captured = capsys.readouterr()
    # tqdm output is in stderr by default
    assert "Epoch 1/2" in captured.err
    assert "Epoch 2/2" in captured.err


def test_trainer_checkpointing(tmp_path):
    betas = [0.1, 0.2]
    forward_process = DDPMForwardProcess(betas)
    decoder = MockDecoder()
    trainer = DDPMTrainer(decoder, forward_process)

    dataset = SimpleDataset(size=2)
    dataloader = DataLoader(dataset, batch_size=2)
    optimizer = torch.optim.SGD(decoder.parameters(), lr=0.1)

    checkpoint_dir = tmp_path / "checkpoints"
    num_epochs = 2

    trainer.train(
        dataloader,
        optimizer,
        num_epochs=num_epochs,
        checkpoint_dir=checkpoint_dir,
        checkpoint_freq=1,
    )

    # Check if checkpoint files exist
    assert (checkpoint_dir / "checkpoint_epoch_1.pt").exists()
    assert (checkpoint_dir / "checkpoint_epoch_2.pt").exists()

    # Load and verify content of the last checkpoint
    checkpoint = torch.load(checkpoint_dir / "checkpoint_epoch_2.pt", weights_only=True)
    assert checkpoint["epoch"] == 1  # 0-indexed epoch stored
    assert "model_state_dict" in checkpoint
    assert "optimizer_state_dict" in checkpoint

    # Check if weights were saved correctly
    assert torch.allclose(checkpoint["model_state_dict"]["param"], decoder.param)
