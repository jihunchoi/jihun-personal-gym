import torch
from diffusion_common import ResUNet
from diffusion_common.decoders.unet import (
    ResUNetResidualBlock,
    ResUNetEncoderBlock,
    ResUNetDecoderBlock,
    ResUNetBridgeBlock,
)


def test_residual_block_no_halve():
    in_channels = 64
    out_channels = 64
    time_embedding_dim = 128
    batch_size = 2
    h, w = 32, 32

    block = ResUNetResidualBlock(in_channels, out_channels, time_embedding_dim, halve_size=False)
    x = torch.randn(batch_size, in_channels, h, w)
    t_emb = torch.randn(batch_size, time_embedding_dim)
    output = block(x, t_emb)

    assert output.shape == (batch_size, out_channels, h, w)


def test_residual_block_halve():
    in_channels = 64
    out_channels = 128
    time_embedding_dim = 128
    batch_size = 2
    h, w = 32, 32

    block = ResUNetResidualBlock(in_channels, out_channels, time_embedding_dim, halve_size=True)
    x = torch.randn(batch_size, in_channels, h, w)
    t_emb = torch.randn(batch_size, time_embedding_dim)
    output = block(x, t_emb)

    assert output.shape == (batch_size, out_channels, h // 2, w // 2)


def test_encoder_block_input_data():
    in_channels = 3
    out_channels = 64
    time_embedding_dim = 128
    batch_size = 2
    h, w = 32, 32

    block = ResUNetEncoderBlock(in_channels, out_channels, time_embedding_dim, input_is_data=True)
    x = torch.randn(batch_size, in_channels, h, w)
    t_emb = torch.randn(batch_size, time_embedding_dim)
    output = block(x, t_emb)

    assert output.shape == (batch_size, out_channels, h, w)


def test_encoder_block_halve():
    in_channels = 64
    out_channels = 128
    time_embedding_dim = 128
    batch_size = 2
    h, w = 32, 32

    block = ResUNetEncoderBlock(in_channels, out_channels, time_embedding_dim, input_is_data=False)
    x = torch.randn(batch_size, in_channels, h, w)
    t_emb = torch.randn(batch_size, time_embedding_dim)
    output = block(x, t_emb)

    assert output.shape == (batch_size, out_channels, h // 2, w // 2)


def test_bridge_block():
    in_channels = 256
    out_channels = 512
    time_embedding_dim = 128
    batch_size = 2
    h, w = 8, 8

    block = ResUNetBridgeBlock(in_channels, out_channels, time_embedding_dim)
    x = torch.randn(batch_size, in_channels, h, w)
    t_emb = torch.randn(batch_size, time_embedding_dim)
    output = block(x, t_emb)

    assert output.shape == (batch_size, out_channels, h // 2, w // 2)


def test_decoder_block():
    in_channels = 512
    out_channels = 256
    skip_channels = 256
    time_embedding_dim = 128
    batch_size = 2
    h, w = 8, 8

    block = ResUNetDecoderBlock(in_channels, out_channels, skip_channels, time_embedding_dim)
    x = torch.randn(batch_size, in_channels, h, w)
    skip = torch.randn(batch_size, skip_channels, h * 2, w * 2)
    t_emb = torch.randn(batch_size, time_embedding_dim)
    output = block(x, skip, t_emb)

    assert output.shape == (batch_size, out_channels, h * 2, w * 2)


def test_full_resunet():
    input_dim = 3
    output_dim = 3
    layer_dims = [64, 128, 256]
    bridge_dim = 512
    time_embedding_dim = 128
    batch_size = 2
    h, w = 32, 32

    model = ResUNet(input_dim, output_dim, layer_dims, bridge_dim, time_embedding_dim)
    x = torch.randn(batch_size, input_dim, h, w)
    t = torch.randint(0, 1000, (batch_size,))
    output = model(x, t)

    assert output.shape == (batch_size, output_dim, h, w)
