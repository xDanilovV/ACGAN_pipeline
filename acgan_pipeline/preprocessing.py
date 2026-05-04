from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class PreprocessingConfig:
    """Named GC-IMS preprocessing choices saved with each experiment run.

    Axis convention for plain arrays is ``[retention_time, drift_time]``.
    ``keep_drift`` is the practical RIP-removal hook: after converting to a
    RIP-relative coordinate system with gc-ims-tools, scientific workflows often
    keep only drift-time values above the RIP region. For array-only data we use
    index ranges so the same pipeline remains dataset-agnostic.
    """

    crop_rt: tuple[int | None, int | None] = (None, None)
    keep_drift: tuple[int | None, int | None] = (None, None)
    subtract_first_rows: int = 0
    log1p: bool = False
    clip_percentiles: tuple[float, float] | None = None


def preprocess_dataset(
    samples: np.ndarray,
    config: PreprocessingConfig,
) -> tuple[np.ndarray, dict[str, object]]:
    """Apply reproducible array-level preprocessing before tensor conversion."""

    processed = np.asarray(samples, dtype=np.float32)
    if processed.ndim != 3:
        raise ValueError("preprocess_dataset expects samples shaped [N, H, W]")

    processed = np.stack([preprocess_spectrum(sample, config) for sample in processed]).astype(np.float32)
    report = {
        "config": asdict(config),
        "input_shape": list(np.asarray(samples).shape),
        "output_shape": list(processed.shape),
        "input_min": float(np.min(samples)),
        "input_max": float(np.max(samples)),
        "output_min": float(np.min(processed)),
        "output_max": float(np.max(processed)),
    }
    return processed, report


def preprocess_spectrum(sample: np.ndarray, config: PreprocessingConfig) -> np.ndarray:
    """Preprocess one 2D spectrum while preserving scientific axes."""

    spectrum = np.asarray(sample, dtype=np.float32)
    if spectrum.ndim != 2:
        raise ValueError("preprocess_spectrum expects one [H, W] array")

    if config.crop_rt != (None, None):
        start, stop = config.crop_rt
        spectrum = spectrum[slice(start, stop), :]

    if config.keep_drift != (None, None):
        start, stop = config.keep_drift
        spectrum = spectrum[:, slice(start, stop)]

    if config.subtract_first_rows > 0:
        baseline = spectrum[: config.subtract_first_rows, :].mean(axis=0, keepdims=True)
        spectrum = spectrum - baseline

    if config.log1p:
        spectrum = np.log1p(np.maximum(spectrum, 0.0))

    if config.clip_percentiles is not None:
        low, high = np.percentile(spectrum, config.clip_percentiles)
        spectrum = np.clip(spectrum, low, high)

    return spectrum.astype(np.float32)


def ensure_channel_first(samples: np.ndarray) -> np.ndarray:
    """Convert ``[N, H, W]`` samples to ``[N, 1, H, W]`` if needed."""

    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim == 3:
        return samples[:, None, :, :]
    if samples.ndim == 4 and samples.shape[1] == 1:
        return samples
    raise ValueError("expected samples shaped [N, H, W] or [N, 1, H, W]")


def normalize_minus_one_one(
    samples: np.ndarray,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Normalize arrays to the GAN-friendly ``[-1, 1]`` range."""

    samples = np.asarray(samples, dtype=np.float32)
    min_value = float(np.min(samples)) if min_value is None else min_value
    max_value = float(np.max(samples)) if max_value is None else max_value
    if max_value <= min_value:
        raise ValueError("max_value must be greater than min_value")

    normalized = (samples - min_value) / (max_value - min_value)
    normalized = np.clip(normalized, 0.0, 1.0) * 2.0 - 1.0
    return normalized.astype(np.float32), {"min_value": min_value, "max_value": max_value}


def denormalize_minus_one_one(
    samples: np.ndarray,
    *,
    min_value: float,
    max_value: float,
) -> np.ndarray:
    """Map generated ``[-1, 1]`` samples back to an original intensity range."""

    samples = np.asarray(samples, dtype=np.float32)
    samples = (samples + 1.0) / 2.0
    return samples * (max_value - min_value) + min_value
