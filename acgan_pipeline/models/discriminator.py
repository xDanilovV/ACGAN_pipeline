from __future__ import annotations

from torch import Tensor, nn


class AsymmetricDownsampleBlock(nn.Module):
    """CNN block with separate horizontal/vertical receptive fields."""

    def __init__(self, in_channels: int, out_channels: int, use_norm: bool = True) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(5, 1), stride=(2, 1), padding=(2, 0)),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        if use_norm:
            layers.insert(2, nn.BatchNorm2d(out_channels))
            layers.append(nn.BatchNorm2d(out_channels))
        self.block = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class Discriminator(nn.Module):
    """AC-GAN discriminator with adversarial and auxiliary class heads."""

    def __init__(
        self,
        num_classes: int,
        input_shape: tuple[int, int] = (128, 128),
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        if input_shape[0] % 16 != 0 or input_shape[1] % 16 != 0:
            raise ValueError("input_shape dimensions must be divisible by 16")

        self.num_classes = num_classes

        self.features = nn.Sequential(
            AsymmetricDownsampleBlock(1, base_channels, use_norm=False),
            AsymmetricDownsampleBlock(base_channels, base_channels * 2),
            AsymmetricDownsampleBlock(base_channels * 2, base_channels * 4),
            AsymmetricDownsampleBlock(base_channels * 4, base_channels * 8),
        )

        feature_shape = (input_shape[0] // 16, input_shape[1] // 16)
        flattened = base_channels * 8 * feature_shape[0] * feature_shape[1]
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
        )
        self.real_fake_head = nn.Linear(512, 1)
        self.class_head = nn.Linear(512, num_classes)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        features = self.features(x)
        shared = self.shared(features)
        real_fake_logits = self.real_fake_head(shared).squeeze(1)
        class_logits = self.class_head(shared)
        return real_fake_logits, class_logits
