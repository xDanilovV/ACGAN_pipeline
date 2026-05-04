from __future__ import annotations

from torch import nn

from acgan_pipeline.models.discriminator import Discriminator
from acgan_pipeline.models.generator import Generator


class ACGAN(nn.Module):
    """Convenience container for the generator and discriminator."""

    def __init__(
        self,
        num_classes: int,
        noise_dim: int = 100,
        image_shape: tuple[int, int] = (128, 128),
        generator_channels: int = 256,
        discriminator_channels: int = 32,
    ) -> None:
        super().__init__()
        self.noise_dim = noise_dim
        self.num_classes = num_classes
        self.generator = Generator(
            noise_dim=noise_dim,
            num_classes=num_classes,
            output_shape=image_shape,
            base_channels=generator_channels,
        )
        self.discriminator = Discriminator(
            num_classes=num_classes,
            input_shape=image_shape,
            base_channels=discriminator_channels,
        )
