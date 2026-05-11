from __future__ import annotations

import json
import random
import copy
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
    discriminator_fake_class_weight: float = 0.0
    tv_loss_weight: float = 1e-4
    generator_intensity_match_weight: float = 2.0
    generator_peak_density_weight: float = 8.0
    generator_border_weight: float = 2.0
    generator_peak_threshold: float = 0.65
    generator_peak_temperature: float = 0.05
    generator_border_width: int = 4
    generator_use_class_templates: bool = False
    generator_template_residual_scale: float = 0.35
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
    discriminator_class_image_head_scale: float = 1.0
    pretrain_classifier_epochs: int = 0
    pretrain_classifier_lr: float = 1e-3
    generator_class_uses_image_head: bool = False
    generator_steps: int = 1
    discriminator_update_every: int = 1
    sample_every: int = 10
    checkpoint_every: int = 10
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 0.0
    early_stopping_metric: str = "g_structure"
    early_stopping_mode: str = "min"
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
        class_image_head_scale=config.discriminator_class_image_head_scale,
    ).to(device)
    generator.apply(_weights_init)
    discriminator.apply(_weights_init)
    if config.generator_use_class_templates:
        class_templates, template_counts = _compute_class_templates(dataset, num_classes, image_shape, config, device)
        generator.set_class_templates(class_templates, residual_scale=config.generator_template_residual_scale)
        np.savez_compressed(
            output_dir / "class_templates.npz",
            templates=class_templates.detach().cpu().numpy()[:, 0],
            counts=template_counts.detach().cpu().numpy(),
        )
        print(
            "Anchored generator to class templates "
            f"(residual scale {config.generator_template_residual_scale:.3f})"
        )

    optimizer_g = torch.optim.Adam(generator.parameters(), lr=config.lr_g or config.lr, betas=config.betas)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=config.lr_d or config.lr, betas=config.betas)

    history: list[dict[str, float]] = []
    fixed_labels = torch.arange(num_classes, device=device)
    best_cls_acc = -1.0
    best_early_value: float | None = None
    best_early_epoch = 0
    epochs_without_improvement = 0
    best_early_generator_state: dict[str, torch.Tensor] | None = None
    best_early_discriminator_state: dict[str, torch.Tensor] | None = None
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
        g_intensity_meter = AverageMeter()
        g_peak_density_meter = AverageMeter()
        g_border_meter = AverageMeter()
        g_structure_meter = AverageMeter()
        d_adv_real_meter = AverageMeter()
        d_adv_fake_meter = AverageMeter()
        d_cls_meter = AverageMeter()
        d_cls_fake_meter = AverageMeter()
        instance_noise_std = _current_instance_noise(config, epoch)
        real_target = 1.0 - config.label_smoothing

        for batch_index, (real_samples, labels) in enumerate(dataloader):
            real_samples = real_samples.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            batch_size = real_samples.size(0)

            should_update_d = batch_index % max(1, config.discriminator_update_every) == 0
            if should_update_d:
                optimizer_d.zero_grad(set_to_none=True)
                noise = torch.randn(batch_size, config.noise_dim, device=device)
                fake_samples = generator(noise, labels).detach()
                noisy_real = _add_instance_noise(real_samples, instance_noise_std)
                noisy_fake = _add_instance_noise(fake_samples, instance_noise_std)
                real_logits, real_class_logits = discriminator(noisy_real, labels)
                fake_logits, fake_class_logits_for_d = discriminator(noisy_fake, labels, use_image_class_head=False)
                d_loss, d_parts = discriminator_loss(
                    real_logits,
                    fake_logits,
                    real_class_logits,
                    labels,
                    fake_class_logits=fake_class_logits_for_d,
                    fake_labels=labels,
                    class_weight=config.class_loss_weight,
                    fake_class_weight=config.discriminator_fake_class_weight,
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
                d_cls_fake_meter.update(d_parts["d_cls_fake"], batch_size)

            for _ in range(max(1, config.generator_steps)):
                optimizer_g.zero_grad(set_to_none=True)
                noise = torch.randn(batch_size, config.noise_dim, device=device)
                fake_samples = generator(noise, labels)
                fake_logits, fake_class_logits = discriminator(
                    fake_samples,
                    labels,
                    use_image_class_head=config.generator_class_uses_image_head,
                )
                g_loss, g_parts = generator_loss(
                    fake_logits,
                    fake_class_logits,
                    labels,
                    fake_samples,
                    real_samples=real_samples,
                    class_weight=config.class_loss_weight,
                    tv_weight=config.tv_loss_weight,
                    intensity_match_weight=config.generator_intensity_match_weight,
                    peak_density_weight=config.generator_peak_density_weight,
                    border_weight=config.generator_border_weight,
                    peak_threshold=config.generator_peak_threshold,
                    peak_temperature=config.generator_peak_temperature,
                    border_width=config.generator_border_width,
                    real_target=real_target,
                )
                g_loss.backward()
                optimizer_g.step()

                g_meter.update(float(g_loss.detach().cpu()), batch_size)
                g_adv_meter.update(g_parts["g_adv"], batch_size)
                g_cls_meter.update(g_parts["g_cls"], batch_size)
                g_tv_meter.update(g_parts["g_tv"], batch_size)
                g_intensity_meter.update(g_parts["g_intensity_match"], batch_size)
                g_peak_density_meter.update(g_parts["g_peak_density"], batch_size)
                g_border_meter.update(g_parts["g_border"], batch_size)
                g_structure_meter.update(g_parts["g_structure"], batch_size)

        epoch_log = {
            "epoch": float(epoch),
            "generator_loss": g_meter.average,
            "discriminator_loss": d_meter.average,
            "g_adv": g_adv_meter.average,
            "g_cls": g_cls_meter.average,
            "g_tv": g_tv_meter.average,
            "g_intensity_match": g_intensity_meter.average,
            "g_peak_density": g_peak_density_meter.average,
            "g_border": g_border_meter.average,
            "g_structure": g_structure_meter.average,
            "d_adv_real": d_adv_real_meter.average,
            "d_adv_fake": d_adv_fake_meter.average,
            "d_cls": d_cls_meter.average,
            "d_cls_fake": d_cls_fake_meter.average,
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

        if config.early_stopping_patience > 0:
            metric_value = epoch_log.get(config.early_stopping_metric)
            if metric_value is None:
                raise ValueError(f"unknown early stopping metric: {config.early_stopping_metric}")
            if _is_improved(
                metric_value,
                best_early_value,
                mode=config.early_stopping_mode,
                min_delta=config.early_stopping_min_delta,
            ):
                best_early_value = float(metric_value)
                best_early_epoch = epoch
                epochs_without_improvement = 0
                best_early_generator_state = copy.deepcopy(generator.state_dict())
                best_early_discriminator_state = copy.deepcopy(discriminator.state_dict())
                save_checkpoint(generator, discriminator, optimizer_g, optimizer_d, epoch, checkpoint_dir / "best_early_stopping.pt")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= config.early_stopping_patience:
                    print(
                        "Early stopping at epoch "
                        f"{epoch:03d}; best {config.early_stopping_metric}="
                        f"{best_early_value:.6f} at epoch {best_early_epoch:03d}"
                    )
                    break

    if config.early_stopping_patience > 0 and best_early_generator_state is not None:
        generator.load_state_dict(best_early_generator_state)
        if best_early_discriminator_state is not None:
            discriminator.load_state_dict(best_early_discriminator_state)
        print(f"Restored best early-stopping generator from epoch {best_early_epoch:03d}")

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
    generator.load_state_dict(checkpoint["generator"], strict=False)
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


@torch.no_grad()
def _compute_class_templates(
    dataset,
    num_classes: int,
    image_shape: tuple[int, int],
    config: TrainConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute class-average tensors in the normalized training space."""

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    templates = torch.zeros(num_classes, 1, *image_shape, device=device)
    counts = torch.zeros(num_classes, device=device)

    for samples, labels in loader:
        samples = samples.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        for class_id in range(num_classes):
            mask = labels == class_id
            if mask.any():
                templates[class_id] += samples[mask].sum(dim=0)
                counts[class_id] += mask.sum()

    total = counts.sum().clamp_min(1.0)
    global_template = templates.sum(dim=0, keepdim=False) / total
    for class_id in range(num_classes):
        if counts[class_id].item() > 0:
            templates[class_id] /= counts[class_id]
        else:
            templates[class_id] = global_template

    return templates.clamp(-1.0, 1.0), counts


def _weights_init(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        weight = getattr(module, "weight_orig", module.weight)
        if isinstance(module, nn.Linear) and module.in_features > 4096 and module.out_features <= 128:
            nn.init.xavier_uniform_(weight)
        else:
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


def _is_improved(value: float, best: float | None, *, mode: str, min_delta: float) -> bool:
    if best is None:
        return True
    if mode == "min":
        return value < best - min_delta
    if mode == "max":
        return value > best + min_delta
    raise ValueError(f"early_stopping_mode must be 'min' or 'max', got {mode!r}")
