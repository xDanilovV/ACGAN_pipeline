from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from acgan_pipeline.models.discriminator import Discriminator
from acgan_pipeline.models.generator import Generator
from acgan_pipeline.utils.losses import discriminator_loss, generator_loss
from acgan_pipeline.utils.metrics import AverageMeter, classification_accuracy


@dataclass
class TrainConfig:
    num_epochs: int = 100
    batch_size: int = 32
    noise_dim: int = 100
    lr: float = 2e-4
    lr_g: float | None = None
    lr_d: float | None = None
    betas: tuple[float, float] = (0.5, 0.999)
    class_loss_weight: float = 1.0
    tv_loss_weight: float = 1e-4
    label_smoothing: float = 0.0
    instance_noise_std: float = 0.0
    instance_noise_decay_epochs: int = 50
    projection_scale: float = 0.1
    generator_base_channels: int = 256
    discriminator_base_channels: int = 32
    discriminator_use_norm: bool = False
    discriminator_use_spectral_norm: bool = False
    discriminator_pool_shape: tuple[int, int] = (8, 4)
    discriminator_input_pool_shape: tuple[int, int] = (32, 8)
    discriminator_dropout: float = 0.1
    pretrain_classifier_epochs: int = 0
    pretrain_classifier_lr: float = 1e-3
    generator_steps: int = 1
    discriminator_update_every: int = 1
    sample_every: int = 10
    checkpoint_every: int = 10
    output_dir: str = "outputs"
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


def train_acgan(
    dataset,
    *,
    num_classes: int,
    image_shape: tuple[int, int] = (128, 128),
    config: TrainConfig | None = None,
) -> tuple[Generator, Discriminator, list[dict[str, float]]]:
    """Train an AC-GAN on arbitrary 2D scientific spectra."""

    config = config or TrainConfig()
    _set_seed(config.seed)
    device = torch.device(config.device)
    output_dir = Path(config.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    sample_dir = output_dir / "samples"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=len(dataset) >= config.batch_size,
    )

    generator = Generator(
        config.noise_dim,
        num_classes,
        image_shape,
        base_channels=config.generator_base_channels,
    ).to(device)
    discriminator = Discriminator(
        num_classes,
        image_shape,
        base_channels=config.discriminator_base_channels,
        projection_scale=config.projection_scale,
        use_norm=config.discriminator_use_norm,
        use_spectral_norm=config.discriminator_use_spectral_norm,
        pool_shape=config.discriminator_pool_shape,
        input_pool_shape=config.discriminator_input_pool_shape,
        dropout=config.discriminator_dropout,
    ).to(device)
    generator.apply(_weights_init)
    discriminator.apply(_weights_init)

    optimizer_g = torch.optim.Adam(generator.parameters(), lr=config.lr_g or config.lr, betas=config.betas)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=config.lr_d or config.lr, betas=config.betas)

    history: list[dict[str, float]] = []
    fixed_labels = torch.arange(num_classes, device=device)
    best_cls_acc = -1.0
    started_at = time.perf_counter()

    if config.pretrain_classifier_epochs > 0:
        pretrain_history = pretrain_discriminator_classifier(discriminator, dataloader, config, device)
        _save_history(pretrain_history, output_dir / "classifier_pretrain_history.json")

    for epoch in range(1, config.num_epochs + 1):
        epoch_started_at = time.perf_counter()
        g_meter = AverageMeter()
        d_meter = AverageMeter()
        acc_meter = AverageMeter()
        g_adv_meter = AverageMeter()
        g_cls_meter = AverageMeter()
        g_tv_meter = AverageMeter()
        d_adv_real_meter = AverageMeter()
        d_adv_fake_meter = AverageMeter()
        d_cls_meter = AverageMeter()
        instance_noise_std = _current_instance_noise(config, epoch)
        real_target = 1.0 - config.label_smoothing

        for batch_index, (real_samples, labels) in enumerate(dataloader):
            real_samples = real_samples.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            batch_size = real_samples.size(0)

            # Train discriminator on real spectra and detached synthetic spectra.
            should_update_d = batch_index % max(1, config.discriminator_update_every) == 0
            if should_update_d:
                optimizer_d.zero_grad(set_to_none=True)
                noise = torch.randn(batch_size, config.noise_dim, device=device)
                fake_samples = generator(noise, labels).detach()
                noisy_real = _add_instance_noise(real_samples, instance_noise_std)
                noisy_fake = _add_instance_noise(fake_samples, instance_noise_std)
                real_logits, real_class_logits = discriminator(noisy_real, labels)
                fake_logits, _ = discriminator(noisy_fake, labels)
                d_loss, d_parts = discriminator_loss(
                    real_logits,
                    fake_logits,
                    real_class_logits,
                    labels,
                    class_weight=config.class_loss_weight,
                    real_target=real_target,
                )
                d_loss.backward()
                optimizer_d.step()

                acc = classification_accuracy(real_class_logits, labels)
                d_meter.update(float(d_loss.detach().cpu()), batch_size)
                acc_meter.update(acc, batch_size)
                d_adv_real_meter.update(d_parts["d_adv_real"], batch_size)
                d_adv_fake_meter.update(d_parts["d_adv_fake"], batch_size)
                d_cls_meter.update(d_parts["d_cls"], batch_size)

            # Train generator to fool discriminator and produce class-consistent spectra.
            for _ in range(max(1, config.generator_steps)):
                optimizer_g.zero_grad(set_to_none=True)
                noise = torch.randn(batch_size, config.noise_dim, device=device)
                fake_samples = generator(noise, labels)
                fake_logits, fake_class_logits = discriminator(fake_samples, labels)
                g_loss, g_parts = generator_loss(
                    fake_logits,
                    fake_class_logits,
                    labels,
                    fake_samples,
                    class_weight=config.class_loss_weight,
                    tv_weight=config.tv_loss_weight,
                    real_target=real_target,
                )
                g_loss.backward()
                optimizer_g.step()

                g_meter.update(float(g_loss.detach().cpu()), batch_size)
                g_adv_meter.update(g_parts["g_adv"], batch_size)
                g_cls_meter.update(g_parts["g_cls"], batch_size)
                g_tv_meter.update(g_parts["g_tv"], batch_size)

        epoch_log = {
            "epoch": float(epoch),
            "generator_loss": g_meter.average,
            "discriminator_loss": d_meter.average,
            "g_adv": g_adv_meter.average,
            "g_cls": g_cls_meter.average,
            "g_tv": g_tv_meter.average,
            "d_adv_real": d_adv_real_meter.average,
            "d_adv_fake": d_adv_fake_meter.average,
            "d_cls": d_cls_meter.average,
            "classification_accuracy": acc_meter.average,
            "instance_noise_std": instance_noise_std,
            "seconds": time.perf_counter() - epoch_started_at,
        }
        history.append(epoch_log)
        _save_history(history, output_dir / "training_history.json")
        print(
            f"Epoch {epoch:03d}/{config.num_epochs} | "
            f"G: {g_meter.average:.4f} | D: {d_meter.average:.4f} | "
            f"Cls acc: {acc_meter.average:.4f}"
        )

        if epoch % config.sample_every == 0 or epoch == 1:
            labels_for_grid = fixed_labels.repeat_interleave(max(1, min(4, config.batch_size // num_classes)))
            _save_generated_numpy(generator, labels_for_grid, config.noise_dim, sample_dir / f"epoch_{epoch:04d}.npz")

        if epoch % config.checkpoint_every == 0 or epoch == config.num_epochs:
            save_checkpoint(generator, discriminator, optimizer_g, optimizer_d, epoch, checkpoint_dir / f"epoch_{epoch:04d}.pt")

        if acc_meter.average > best_cls_acc:
            best_cls_acc = acc_meter.average
            save_checkpoint(generator, discriminator, optimizer_g, optimizer_d, epoch, checkpoint_dir / "best_discriminator_cls.pt")

    _save_history(history, output_dir / "training_history.json", total_seconds=time.perf_counter() - started_at)
    return generator, discriminator, history


@torch.no_grad()
def generate_samples(
    generator: Generator,
    num_samples_per_class: int,
    *,
    num_classes: int,
    noise_dim: int = 100,
    device: str | torch.device | None = None,
    batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate class-conditioned spectra and return ``(arrays, labels)``."""

    device = torch.device(device) if device is not None else next(generator.parameters()).device
    generator.eval()
    all_samples: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for class_id in range(num_classes):
        remaining = num_samples_per_class
        while remaining > 0:
            current = min(batch_size, remaining)
            labels = torch.full((current,), class_id, dtype=torch.long, device=device)
            noise = torch.randn(current, noise_dim, device=device)
            samples = generator(noise, labels).detach().cpu().numpy()
            all_samples.append(samples[:, 0])
            all_labels.append(np.full(current, class_id, dtype=np.int64))
            remaining -= current

    return np.concatenate(all_samples, axis=0), np.concatenate(all_labels, axis=0)


def save_checkpoint(
    generator: Generator,
    discriminator: Discriminator,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    epoch: int,
    path: Path,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict(),
        },
        path,
    )


def load_generator_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    num_classes: int,
    noise_dim: int = 100,
    image_shape: tuple[int, int] = (128, 128),
    device: str | torch.device | None = None,
) -> Generator:
    """Load only the generator from a saved AC-GAN checkpoint."""

    device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator(noise_dim, num_classes, image_shape).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint["generator"])
    generator.eval()
    return generator


def pretrain_discriminator_classifier(
    discriminator: Discriminator,
    dataloader: DataLoader,
    config: TrainConfig,
    device: torch.device,
) -> list[dict[str, float]]:
    """Pretrain the AC-GAN discriminator auxiliary classifier on real spectra."""

    optimizer = torch.optim.AdamW(
        discriminator.parameters(),
        lr=config.pretrain_classifier_lr,
        weight_decay=1e-4,
    )
    history: list[dict[str, float]] = []
    for epoch in range(1, config.pretrain_classifier_epochs + 1):
        discriminator.train()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        started_at = time.perf_counter()
        for real_samples, labels in dataloader:
            real_samples = real_samples.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            _, class_logits = discriminator(real_samples)
            loss = nn.functional.cross_entropy(class_logits, labels)
            loss.backward()
            optimizer.step()
            loss_meter.update(float(loss.detach().cpu()), len(labels))
            acc_meter.update(classification_accuracy(class_logits, labels), len(labels))

        row = {
            "epoch": float(epoch),
            "classifier_loss": loss_meter.average,
            "classification_accuracy": acc_meter.average,
            "seconds": time.perf_counter() - started_at,
        }
        history.append(row)
        print(
            f"Pretrain {epoch:03d}/{config.pretrain_classifier_epochs} | "
            f"Cls loss: {loss_meter.average:.4f} | "
            f"Cls acc: {acc_meter.average:.4f}"
        )
    return history


def _save_history(history: list[dict[str, float]], path: Path, total_seconds: float | None = None) -> None:
    payload = {"epochs": history}
    if total_seconds is not None:
        payload["total_seconds"] = total_seconds
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


@torch.no_grad()
def _save_generated_numpy(generator: Generator, labels: torch.Tensor, noise_dim: int, path: Path) -> None:
    was_training = generator.training
    generator.eval()
    device = next(generator.parameters()).device
    labels = labels.to(device)
    noise = torch.randn(labels.size(0), noise_dim, device=device)
    samples = generator(noise, labels).detach().cpu().numpy()
    np.savez_compressed(path, samples=samples[:, 0], labels=labels.cpu().numpy())
    if was_training:
        generator.train()


def _weights_init(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        weight = getattr(module, "weight_orig", module.weight)
        nn.init.normal_(weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
        if module.weight is not None:
            nn.init.normal_(module.weight, mean=1.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _current_instance_noise(config: TrainConfig, epoch: int) -> float:
    if config.instance_noise_std <= 0:
        return 0.0
    if config.instance_noise_decay_epochs <= 0:
        return config.instance_noise_std
    progress = min(max((epoch - 1) / config.instance_noise_decay_epochs, 0.0), 1.0)
    return config.instance_noise_std * (1.0 - progress)


def _add_instance_noise(samples: torch.Tensor, std: float) -> torch.Tensor:
    if std <= 0:
        return samples
    return (samples + torch.randn_like(samples) * std).clamp(-1.0, 1.0)
