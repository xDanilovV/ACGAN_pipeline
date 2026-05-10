from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _sn(module: nn.Module) -> nn.Module:
    return nn.utils.spectral_norm(module)


def _maybe_sn(module: nn.Module, enabled: bool) -> nn.Module:
    return _sn(module) if enabled else module


class AsymmetricDownsampleBlock(nn.Module):
    """CNN block with separate horizontal/vertical receptive fields."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        use_norm: bool = True,
        use_spectral_norm: bool = False,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            _maybe_sn(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2)),
                use_spectral_norm,
            ),
            nn.LeakyReLU(0.2, inplace=True),
            _maybe_sn(
                nn.Conv2d(out_channels, out_channels, kernel_size=(5, 1), stride=(2, 1), padding=(2, 0)),
                use_spectral_norm,
            ),
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
        use_spectral_norm: bool = False,
        pool_shape: tuple[int, int] = (8, 4),
        input_pool_shape: tuple[int, int] = (32, 8),
        dropout: float = 0.1,
        class_image_head_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if input_shape[0] % 16 != 0 or input_shape[1] % 16 != 0:
            raise ValueError("input_shape dimensions must be divisible by 16")

        self.num_classes = num_classes
        self.projection_scale = projection_scale
        self.pool_shape = pool_shape
        self.input_pool_shape = input_pool_shape
        self.class_image_head_scale = class_image_head_scale

        self.features = nn.Sequential(
            AsymmetricDownsampleBlock(1, base_channels, use_norm=False, use_spectral_norm=use_spectral_norm),
            AsymmetricDownsampleBlock(
                base_channels,
                base_channels * 2,
                use_norm=use_norm,
                use_spectral_norm=use_spectral_norm,
            ),
            AsymmetricDownsampleBlock(
                base_channels * 2,
                base_channels * 4,
                use_norm=use_norm,
                use_spectral_norm=use_spectral_norm,
            ),
            AsymmetricDownsampleBlock(
                base_channels * 4,
                base_channels * 8,
                use_norm=use_norm,
                use_spectral_norm=use_spectral_norm,
            ),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(pool_shape)
        self.max_pool = nn.AdaptiveMaxPool2d(pool_shape)
        self.input_avg_pool = nn.AdaptiveAvgPool2d(input_pool_shape)
        self.input_max_pool = nn.AdaptiveMaxPool2d(input_pool_shape)
        flattened = (
            base_channels * 8 * pool_shape[0] * pool_shape[1] * 2
            + input_pool_shape[0] * input_pool_shape[1] * 2
        )
        self.shared = nn.Sequential(
            nn.Flatten(),
            _maybe_sn(nn.Linear(flattened, 512), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
        )
        self.real_fake_head = _maybe_sn(nn.Linear(512, 1), use_spectral_norm)
        self.projection = nn.Embedding(num_classes, 512)
        self.class_head = _maybe_sn(nn.Linear(512, num_classes), use_spectral_norm)
        self.class_image_head = nn.Linear(input_shape[0] * input_shape[1], num_classes)

    def forward(self, x: Tensor, labels: Tensor | None = None) -> tuple[Tensor, Tensor]:
        x_features = x.add(1.0).mul(0.5).clamp(0.0, 1.0)
        features = self.features(x_features)
        feature_pooled = torch.cat([self.avg_pool(features), self.max_pool(features)], dim=1).flatten(1)
        input_pooled = torch.cat([self.input_avg_pool(x_features), self.input_max_pool(x_features)], dim=1).flatten(1)
        pooled = torch.cat([feature_pooled, input_pooled], dim=1)
        shared = self.shared(pooled)
        real_fake_logits = self.real_fake_head(shared).squeeze(1)
        if labels is not None and self.projection_scale > 0:
            projected = torch.sum(
                F.normalize(self.projection(labels), dim=1) * F.normalize(shared, dim=1),
                dim=1,
            )
            real_fake_logits = real_fake_logits + self.projection_scale * projected
        class_logits = self.class_head(shared)
        if self.class_image_head_scale > 0:
            class_pixels = _samplewise_standardize(x_features).flatten(1)
            class_logits = class_logits + self.class_image_head_scale * self.class_image_head(class_pixels)
        return real_fake_logits, class_logits


def _samplewise_standardize(x: Tensor) -> Tensor:
    """Preserve absolute peak positions while normalizing per-spectrum contrast."""

    mean = x.mean(dim=(2, 3), keepdim=True)
    std = x.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    return (x - mean) / std
