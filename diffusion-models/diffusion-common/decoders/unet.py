# Residual U-Net (ResUNet) implementation.
# Reference: https://arxiv.org/abs/1711.10684

import torch
from torch import nn


class ResUNetResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        input_is_data: bool = False,
        halve_size: bool = True,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_is_data = input_is_data
        self.halve_size = halve_size

        conv1_layers = [
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=2 if halve_size else 1,
                padding=1,
            ),
        ]
        if input_is_data:
            # If input is data, don't apply batch normalization and ReLU
            # before the first convolution.
            conv1_layers = conv1_layers[2:]
        self.conv1 = nn.Sequential(*conv1_layers)

        if in_channels == out_channels and not halve_size:
            self.skip_conv = nn.Identity()
        else:
            self.skip_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=2 if halve_size else 1,
            )

        conv2_layers = [
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            ),
        ]
        self.conv2 = nn.Sequential(*conv2_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        return self.skip_conv(x) + x2


class ResUNetEncoderBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, input_is_data: bool = False
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_is_data = input_is_data

        self.res_block = ResUNetResidualBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            input_is_data=input_is_data,
            halve_size=not input_is_data,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res_block(x)


class ResUNetDecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, skip_channels: int) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.upsampler = nn.ConvTranspose2d(
            in_channels=in_channels, out_channels=in_channels, kernel_size=2, stride=2
        )
        self.res_block = ResUNetResidualBlock(
            in_channels=in_channels + skip_channels,
            out_channels=out_channels,
            input_is_data=False,
            halve_size=False,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x1 = self.upsampler(x)
        return self.res_block(torch.cat([x1, skip], dim=1))


class ResUNetBridgeBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.res_block = ResUNetResidualBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            input_is_data=False,
            halve_size=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res_block(x)


class ResUNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        layer_dims: list[int],
        bridge_dim: int,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_dims = layer_dims

        encoder_dims = layer_dims
        decoder_dims = layer_dims[::-1]

        # Encoder blocks
        self.encoders = nn.ModuleList()
        for i in range(len(encoder_dims)):
            encoder_in_channels = input_dim if i == 0 else encoder_dims[i - 1]
            self.encoders.append(
                ResUNetEncoderBlock(
                    in_channels=encoder_in_channels,
                    out_channels=encoder_dims[i],
                    input_is_data=(i == 0),
                )
            )

        # Bridge block
        self.bridge = ResUNetBridgeBlock(
            in_channels=encoder_dims[-1], out_channels=bridge_dim
        )

        # Decoder blocks
        self.decoders = nn.ModuleList()
        for i in range(len(decoder_dims)):
            decoder_in_channels = bridge_dim if i == 0 else decoder_dims[i - 1]
            skip_channels = encoder_dims[-(i + 1)]
            self.decoders.append(
                ResUNetDecoderBlock(
                    in_channels=decoder_in_channels,
                    out_channels=decoder_dims[i],
                    skip_channels=skip_channels,
                )
            )

        # Final convolution to get the desired output channels
        self.output_conv = nn.Conv2d(
            in_channels=decoder_dims[-1],
            out_channels=output_dim,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder path
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)

        # Bridge
        x = self.bridge(x)

        # Decoder path
        for decoder in self.decoders:
            skip = skips.pop()
            x = decoder(x, skip)

        # Final output convolution
        return self.output_conv(x)
