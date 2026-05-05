from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset


ArrayLoader = Callable[[Any], tuple[np.ndarray, np.ndarray]]


class GCIMSDataset(Dataset):
    """Dataset-agnostic wrapper for 2D GC-IMS-like spectra and labels.

    The class intentionally does not know how a project stores its raw files.
    Pass arrays directly, or pass a path-like source plus a loader function that
    returns ``(samples, labels)``. Samples are expected as ``[N, H, W]`` or
    ``[N, 1, H, W]`` and are resized/normalized on access.
    """

    def __init__(
        self,
        samples: np.ndarray | Sequence[np.ndarray] | str | Path,
        labels: np.ndarray | Sequence[int] | None = None,
        *,
        loader: ArrayLoader | None = None,
        target_shape: tuple[int, int] = (128, 128),
        resize_mode: str = "area",
        normalize: bool = True,
        min_value: float | None = None,
        max_value: float | None = None,
        transform: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        if loader is not None:
            samples, labels = loader(samples)

        if labels is None:
            raise ValueError("labels must be provided when no loader is used")

        self.samples = np.asarray(samples, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.target_shape = target_shape
        self.resize_mode = resize_mode
        self.normalize = normalize
        self.transform = transform

        if len(self.samples) != len(self.labels):
            raise ValueError("samples and labels must have the same length")

        if self.samples.ndim not in (3, 4):
            raise ValueError("samples must have shape [N, H, W] or [N, 1, H, W]")

        if self.samples.ndim == 4 and self.samples.shape[1] != 1:
            raise ValueError("only single-channel spectra are supported: [N, 1, H, W]")

        self.min_value = float(np.min(self.samples)) if min_value is None else min_value
        self.max_value = float(np.max(self.samples)) if max_value is None else max_value

        if self.normalize and self.max_value <= self.min_value:
            raise ValueError("max_value must be greater than min_value for normalization")

    def __len__(self) -> int:
        return int(len(self.samples))

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        sample = torch.from_numpy(self.samples[index]).float()
        if sample.ndim == 2:
            sample = sample.unsqueeze(0)

        sample = self._resize(sample)
        if self.normalize:
            sample = self._normalize_to_minus_one_one(sample)

        if self.transform is not None:
            sample = self.transform(sample)

        label = torch.tensor(self.labels[index], dtype=torch.long)
        return sample, label

    @property
    def num_classes(self) -> int:
        return int(np.unique(self.labels).size)

    def _resize(self, sample: Tensor) -> Tensor:
        if tuple(sample.shape[-2:]) == self.target_shape:
            return sample
        kwargs = {"size": self.target_shape, "mode": self.resize_mode}
        if self.resize_mode in {"linear", "bilinear", "bicubic", "trilinear"}:
            kwargs["align_corners"] = False
        resized = F.interpolate(sample.unsqueeze(0), **kwargs)
        return resized.squeeze(0)

    def _normalize_to_minus_one_one(self, sample: Tensor) -> Tensor:
        sample = (sample - self.min_value) / (self.max_value - self.min_value)
        sample = sample.clamp(0.0, 1.0)
        return sample.mul(2.0).sub(1.0)
