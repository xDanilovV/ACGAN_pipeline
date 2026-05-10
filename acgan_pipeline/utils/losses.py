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
    class_weight: float = 1.0,
    tv_weight: float = 1e-4,
    real_target: float = 1.0,
) -> tuple[Tensor, dict[str, float]]:
    adv = adversarial_bce_loss(fake_logits, True, real_target=real_target)
    cls = classification_loss(fake_class_logits, labels)
    tv = total_variation_loss(fake_samples)
    loss = adv + class_weight * cls + tv_weight * tv
    parts = {
        "g_adv": float(adv.detach().cpu()),
        "g_cls": float(cls.detach().cpu()),
        "g_tv": float(tv.detach().cpu()),
    }
    return loss, parts
