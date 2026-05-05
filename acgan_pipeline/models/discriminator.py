from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _sn(module: nn.Module) -> nn.Module:
    return nn.utils.spectral_norm(module)


class AsymmetricDownsampleBlock(nn.Module):
    """CNN block with separate horizontal/vertical receptive fields."""

    def __init__(self, in_channels: int, out_channels: int, use_norm: bool = True) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            _sn(nn.Conv2d(in_channels, out_channels, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2))),
            nn.LeakyReLU(0.2, inplace=True),
            _sn(nn.Conv2d(out_channels, out_channels, kernel_size=(5, 1), stride=(2, 1), padding=(2, 0))),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        if use_norm:
            layers.insert(2, nn.BatchNorm2d(out_channels))
            layers.append(nn.BatchNorm2d(out_channels))
        self.block = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class Discriminator(nn.Module):
    """AC-GAN discriminator with projection-conditioned adversarial score."""

    def __init__(
        self,
        num_classes: int,
        input_shape: tuple[int, int] = (128, 128),
        base_channels: int = 32,
        projection_scale: float = 0.1,
        use_norm: bool = False,
    ) -> None:
        super().__init__()
        if input_shape[0] % 16 != 0 or input_shape[1] % 16 != 0:
            raise ValueError("input_shape dimensions must be divisible by 16")

        self.num_classes = num_classes
        self.projection_scale = projection_scale

        self.features = nn.Sequential(
            AsymmetricDownsampleBlock(1, base_channels, use_norm=False),
            AsymmetricDownsampleBlock(base_channels, base_channels * 2, use_norm=use_norm),
            AsymmetricDownsampleBlock(base_channels * 2, base_channels * 4, use_norm=use_norm),
            AsymmetricDownsampleBlock(base_channels * 4, base_channels * 8, use_norm=use_norm),
        )

        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        flattened = base_channels * 8 * 4 * 4
        self.shared = nn.Sequential(
            nn.Flatten(),
            _sn(nn.Linear(flattened, 512)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
        )
        self.real_fake_head = _sn(nn.Linear(512, 1))
        self.projection = nn.Embedding(num_classes, 512)
        self.class_head = _sn(nn.Linear(512, num_classes))

    def forward(self, x: Tensor, labels: Tensor | None = None) -> tuple[Tensor, Tensor]:
        features = self.features(x)
        shared = self.shared(self.pool(features))
        real_fake_logits = self.real_fake_head(shared).squeeze(1)
        if labels is not None and self.projection_scale > 0:
            projected = torch.sum(
                F.normalize(self.projection(labels), dim=1) * F.normalize(shared, dim=1),
                dim=1,
            )
            real_fake_logits = real_fake_logits + self.projection_scale * projected
        class_logits = self.class_head(shared)
        return real_fake_logits, class_logits
