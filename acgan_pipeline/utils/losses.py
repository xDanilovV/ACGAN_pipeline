from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def adversarial_bce_loss(
    logits: Tensor,
    target_is_real: bool,
    *,
    real_target: float = 1.0,
    fake_target: float = 0.0,
) -> Tensor:
    """Binary cross entropy on discriminator logits."""

    target_value = real_target if target_is_real else fake_target
    targets = torch.full_like(logits, target_value)
    return F.binary_cross_entropy_with_logits(logits, targets)


def classification_loss(class_logits: Tensor, labels: Tensor) -> Tensor:
    """Cross entropy for the AC-GAN auxiliary class prediction."""

    return F.cross_entropy(class_logits, labels)


def total_variation_loss(samples: Tensor) -> Tensor:
    """Explicit TV loss encouraging local smoothness in generated spectra.

    ``samples`` is expected to have shape ``[N, C, H, W]``. The loss averages
    absolute first-order differences along both GC-IMS axes.
    """

    if samples.ndim != 4:
        raise ValueError("TV loss expects samples with shape [N, C, H, W]")

    vertical_tv = torch.abs(samples[:, :, 1:, :] - samples[:, :, :-1, :]).mean()
    horizontal_tv = torch.abs(samples[:, :, :, 1:] - samples[:, :, :, :-1]).mean()
    return vertical_tv + horizontal_tv


def discriminator_loss(
    real_logits: Tensor,
    fake_logits: Tensor,
    real_class_logits: Tensor,
    labels: Tensor,
    fake_class_logits: Tensor | None = None,
    fake_labels: Tensor | None = None,
    class_weight: float = 1.0,
    fake_class_weight: float = 0.0,
    real_target: float = 1.0,
    fake_target: float = 0.0,
) -> tuple[Tensor, dict[str, float]]:
    adv_real = adversarial_bce_loss(real_logits, True, real_target=real_target, fake_target=fake_target)
    adv_fake = adversarial_bce_loss(fake_logits, False, real_target=real_target, fake_target=fake_target)
    cls_real = classification_loss(real_class_logits, labels)
    cls_fake = torch.zeros((), dtype=real_logits.dtype, device=real_logits.device)
    if fake_class_logits is not None and fake_labels is not None and fake_class_weight > 0:
        cls_fake = classification_loss(fake_class_logits, fake_labels)
    loss = adv_real + adv_fake + class_weight * cls_real + fake_class_weight * cls_fake
    parts = {
        "d_adv_real": float(adv_real.detach().cpu()),
        "d_adv_fake": float(adv_fake.detach().cpu()),
        "d_cls": float(cls_real.detach().cpu()),
        "d_cls_fake": float(cls_fake.detach().cpu()),
    }
    return loss, parts


def generator_loss(
    fake_logits: Tensor,
    fake_class_logits: Tensor,
    labels: Tensor,
    fake_samples: Tensor,
    real_samples: Tensor | None = None,
    class_weight: float = 1.0,
    tv_weight: float = 1e-4,
    intensity_match_weight: float = 0.0,
    peak_density_weight: float = 0.0,
    border_weight: float = 0.0,
    peak_threshold: float = 0.65,
    peak_temperature: float = 0.05,
    border_width: int = 4,
    real_target: float = 1.0,
) -> tuple[Tensor, dict[str, float]]:
    adv = adversarial_bce_loss(fake_logits, True, real_target=real_target)
    cls = classification_loss(fake_class_logits, labels)
    tv = total_variation_loss(fake_samples)
    structure = spectral_structure_loss(
        fake_samples,
        real_samples,
        intensity_weight=intensity_match_weight,
        peak_density_weight=peak_density_weight,
        border_weight=border_weight,
        peak_threshold=peak_threshold,
        peak_temperature=peak_temperature,
        border_width=border_width,
    )
    loss = adv + class_weight * cls + tv_weight * tv + structure["loss"]
    parts = {
        "g_adv": float(adv.detach().cpu()),
        "g_cls": float(cls.detach().cpu()),
        "g_tv": float(tv.detach().cpu()),
        "g_intensity_match": float(structure["intensity"].detach().cpu()),
        "g_peak_density": float(structure["peak_density"].detach().cpu()),
        "g_border": float(structure["border"].detach().cpu()),
        "g_structure": float(structure["loss"].detach().cpu()),
    }
    return loss, parts


def spectral_structure_loss(
    fake_samples: Tensor,
    real_samples: Tensor | None,
    *,
    intensity_weight: float,
    peak_density_weight: float,
    border_weight: float,
    peak_threshold: float,
    peak_temperature: float,
    border_width: int,
) -> dict[str, Tensor]:
    """Match simple GC-IMS tensor statistics between real and generated batches.

    The AC-GAN adversarial/class losses alone can reward class-looking texture.
    These penalties are deliberately low-level: they push generated spectra
    toward the real batch's global signal density and away from border-heavy
    artifacts without changing the core AC-GAN architecture.
    """

    zero = torch.zeros((), dtype=fake_samples.dtype, device=fake_samples.device)
    if real_samples is None:
        return {"loss": zero, "intensity": zero, "peak_density": zero, "border": zero}

    fake01 = _to_unit_interval(fake_samples)
    real01 = _to_unit_interval(real_samples.detach())

    intensity = F.l1_loss(fake01.mean(), real01.mean())

    temperature = max(float(peak_temperature), 1e-4)
    fake_peak_fraction = torch.sigmoid((fake01 - peak_threshold) / temperature).mean()
    real_peak_fraction = torch.sigmoid((real01 - peak_threshold) / temperature).mean()
    peak_density = F.l1_loss(fake_peak_fraction, real_peak_fraction)

    border = F.l1_loss(_border_mean(fake01, border_width), _border_mean(real01, border_width))

    loss = intensity_weight * intensity + peak_density_weight * peak_density + border_weight * border
    return {"loss": loss, "intensity": intensity, "peak_density": peak_density, "border": border}


def _to_unit_interval(samples: Tensor) -> Tensor:
    return samples.add(1.0).mul(0.5).clamp(0.0, 1.0)


def _border_mean(samples: Tensor, border_width: int) -> Tensor:
    if samples.ndim != 4:
        raise ValueError("border statistic expects samples with shape [N, C, H, W]")
    width = max(1, min(int(border_width), samples.shape[-2] // 2, samples.shape[-1] // 2))
    top = samples[:, :, :width, :]
    bottom = samples[:, :, -width:, :]
    left = samples[:, :, :, :width]
    right = samples[:, :, :, -width:]
    return torch.cat([top.flatten(), bottom.flatten(), left.flatten(), right.flatten()]).mean()
