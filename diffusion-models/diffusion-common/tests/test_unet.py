import pytest
import torch
from decoders.unet import (
    ResUNetResidualBlock,
    ResUNetEncoderBlock,
    ResUNetDecoderBlock,
    ResUNetBridgeBlock,
    ResUNet,
)

def test_residual_block_no_halve():
    in_channels = 64
    out_channels = 64
    batch_size = 2
    h, w = 32, 32
    
    block = ResUNetResidualBlock(in_channels, out_channels, halve_size=False)
    x = torch.randn(batch_size, in_channels, h, w)
    output = block(x)
    
    assert output.shape == (batch_size, out_channels, h, w)

def test_residual_block_halve():
    in_channels = 64
    out_channels = 128
    batch_size = 2
    h, w = 32, 32
    
    block = ResUNetResidualBlock(in_channels, out_channels, halve_size=True)
    x = torch.randn(batch_size, in_channels, h, w)
    output = block(x)
    
    assert output.shape == (batch_size, out_channels, h // 2, w // 2)

def test_encoder_block_input_data():
    in_channels = 3
    out_channels = 64
    batch_size = 2
    h, w = 32, 32
    
    block = ResUNetEncoderBlock(in_channels, out_channels, input_is_data=True)
    x = torch.randn(batch_size, in_channels, h, w)
    output = block(x)
    
    assert output.shape == (batch_size, out_channels, h, w)

def test_encoder_block_halve():
    in_channels = 64
    out_channels = 128
    batch_size = 2
    h, w = 32, 32
    
    block = ResUNetEncoderBlock(in_channels, out_channels, input_is_data=False)
    x = torch.randn(batch_size, in_channels, h, w)
    output = block(x)
    
    assert output.shape == (batch_size, out_channels, h // 2, w // 2)

def test_bridge_block():
    in_channels = 256
    out_channels = 512
    batch_size = 2
    h, w = 8, 8
    
    block = ResUNetBridgeBlock(in_channels, out_channels)
    x = torch.randn(batch_size, in_channels, h, w)
    output = block(x)
    
    assert output.shape == (batch_size, out_channels, h // 2, w // 2)

def test_decoder_block():
    in_channels = 512
    out_channels = 256
    skip_channels = 256
    batch_size = 2
    h, w = 8, 8
    
    block = ResUNetDecoderBlock(in_channels, out_channels, skip_channels)
    x = torch.randn(batch_size, in_channels, h, w)
    skip = torch.randn(batch_size, skip_channels, h * 2, w * 2)
    output = block(x, skip)
    
    assert output.shape == (batch_size, out_channels, h * 2, w * 2)

def test_full_resunet():
    input_dim = 3
    output_dim = 3
    layer_dims = [64, 128, 256]
    bridge_dim = 512
    batch_size = 2
    h, w = 32, 32
    
    model = ResUNet(input_dim, output_dim, layer_dims, bridge_dim)
    x = torch.randn(batch_size, input_dim, h, w)
    output = model(x)
    
    assert output.shape == (batch_size, output_dim, h, w)
