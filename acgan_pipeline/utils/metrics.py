from __future__ import annotations

import torch
from torch import Tensor


@torch.no_grad()
def classification_accuracy(class_logits: Tensor, labels: Tensor) -> float:
    predictions = class_logits.argmax(dim=1)
    return float((predictions == labels).float().mean().cpu())


class AverageMeter:
    """Small utility for epoch-level logging."""

    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += n

    @property
    def average(self) -> float:
        return self.total / max(self.count, 1)
