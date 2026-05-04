from __future__ import annotations

import torch
from torch import Tensor, nn


class ConditionalBatchNorm2d(nn.Module):
    """BatchNorm where scale and bias are produced from the class label."""

    def __init__(self, num_features: int, num_classes: int) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        nn.init.ones_(self.embed.weight[:, :num_features])
        nn.init.zeros_(self.embed.weight[:, num_features:])

    def forward(self, x: Tensor, labels: Tensor) -> Tensor:
        gamma, beta = self.embed(labels).chunk(2, dim=1)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        return self.bn(x) * gamma + beta


class ConditionalUpsampleBlock(nn.Module):
    """Upsample block using asymmetric kernels for 2D scientific spectra."""

    def __init__(self, in_channels: int, out_channels: int, num_classes: int) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv_wide = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 5), padding=(0, 2))
        self.conv_tall = nn.Conv2d(out_channels, out_channels, kernel_size=(5, 1), padding=(2, 0))
        self.norm = ConditionalBatchNorm2d(out_channels, num_classes)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: Tensor, labels: Tensor) -> Tensor:
        x = self.upsample(x)
        x = self.conv_wide(x)
        x = self.conv_tall(x)
        x = self.norm(x, labels)
        return self.activation(x)


class Generator(nn.Module):
    """Class-conditional AC-GAN generator for 1-channel 2D spectra."""

    def __init__(
        self,
        noise_dim: int,
        num_classes: int,
        output_shape: tuple[int, int] = (128, 128),
        label_embedding_dim: int = 50,
        base_channels: int = 256,
    ) -> None:
        super().__init__()
        if output_shape[0] % 16 != 0 or output_shape[1] % 16 != 0:
            raise ValueError("output_shape dimensions must be divisible by 16")

        self.noise_dim = noise_dim
        self.num_classes = num_classes
        self.output_shape = output_shape
        self.init_shape = (output_shape[0] // 16, output_shape[1] // 16)
        self.label_embedding = nn.Embedding(num_classes, label_embedding_dim)

        self.fc = nn.Sequential(
            nn.Linear(noise_dim + label_embedding_dim, base_channels * self.init_shape[0] * self.init_shape[1]),
            nn.BatchNorm1d(base_channels * self.init_shape[0] * self.init_shape[1]),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.blocks = nn.ModuleList(
            [
                ConditionalUpsampleBlock(base_channels, base_channels // 2, num_classes),
                ConditionalUpsampleBlock(base_channels // 2, base_channels // 4, num_classes),
                ConditionalUpsampleBlock(base_channels // 4, base_channels // 8, num_classes),
                ConditionalUpsampleBlock(base_channels // 8, base_channels // 16, num_classes),
            ]
        )

        final_channels = base_channels // 16
        self.to_spectrum = nn.Sequential(
            nn.Conv2d(final_channels, final_channels, kernel_size=(1, 5), padding=(0, 2)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(final_channels, 1, kernel_size=(5, 1), padding=(2, 0)),
            nn.Tanh(),
        )

    def forward(self, noise: Tensor, labels: Tensor) -> Tensor:
        label_features = self.label_embedding(labels)
        x = torch.cat([noise, label_features], dim=1)
        x = self.fc(x)
        x = x.view(x.size(0), -1, self.init_shape[0], self.init_shape[1])

        for block in self.blocks:
            x = block(x, labels)

        return self.to_spectrum(x)
