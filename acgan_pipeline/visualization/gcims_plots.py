from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np


def export_spectrum_plot(
    values: np.ndarray,
    path: str | Path,
    *,
    title: str = "GC-IMS spectrum",
    ret_time: np.ndarray | None = None,
    drift_time: np.ndarray | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> Path:
    """Export a spectrum image through the gc-ims-tools ``ims.Spectrum`` API."""

    ims = _import_gcims()
    values = np.asarray(values, dtype=np.float32)
    ret_time = np.arange(values.shape[0]) if ret_time is None else ret_time
    drift_time = np.arange(values.shape[1]) if drift_time is None else drift_time
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    spectrum = ims.Spectrum(
        name=title,
        values=values,
        ret_time=np.asarray(ret_time),
        drift_time=np.asarray(drift_time),
        time=datetime.now(),
        meta_attr={},
    )
    kwargs = {}
    if vmin is not None:
        kwargs["vmin"] = vmin
    if vmax is not None:
        kwargs["vmax"] = vmax
    fig, ax = spectrum.plot(**kwargs)
    ax.set_title(title)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    _close_figure(fig)
    return path


def export_preprocessing_comparison(
    before: np.ndarray,
    after: np.ndarray,
    path: str | Path,
    *,
    title: str,
) -> Path:
    """Save side-by-side pre/post spectra using gc-ims-tools objects."""

    ims = _import_gcims()
    import matplotlib.pyplot as plt

    before = np.asarray(before, dtype=np.float32)
    after = np.asarray(after, dtype=np.float32)
    before_spectrum = ims.Spectrum(
        name=f"{title} before",
        values=before,
        ret_time=np.arange(before.shape[0]),
        drift_time=np.arange(before.shape[1]),
        time=datetime.now(),
        meta_attr={},
    )
    after_spectrum = ims.Spectrum(
        name=f"{title} after",
        values=after,
        ret_time=np.arange(after.shape[0]),
        drift_time=np.arange(after.shape[1]),
        time=datetime.now(),
        meta_attr={},
    )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    _plot_on_axis(before_spectrum, axes[0], "Before")
    _plot_on_axis(after_spectrum, axes[1], "After")
    fig.suptitle(title)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    _close_figure(fig)
    return path


def export_real_vs_generated_comparison(
    real: np.ndarray,
    generated: np.ndarray,
    path: str | Path,
    *,
    class_name: str,
) -> Path:
    """Save one real spectrum next to one generated spectrum for a class."""

    ims = _import_gcims()
    import matplotlib.pyplot as plt

    real = np.asarray(real, dtype=np.float32)
    generated = np.asarray(generated, dtype=np.float32)
    vmin = float(min(np.min(real), np.min(generated)))
    vmax = float(max(np.max(real), np.max(generated)))
    real_spectrum = ims.Spectrum(
        name=f"Real {class_name}",
        values=real,
        ret_time=np.arange(real.shape[0]),
        drift_time=np.arange(real.shape[1]),
        time=datetime.now(),
        meta_attr={},
    )
    generated_spectrum = ims.Spectrum(
        name=f"Generated {class_name}",
        values=generated,
        ret_time=np.arange(generated.shape[0]),
        drift_time=np.arange(generated.shape[1]),
        time=datetime.now(),
        meta_attr={},
    )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    _plot_on_axis(real_spectrum, axes[0], "Real", vmin=vmin, vmax=vmax)
    _plot_on_axis(generated_spectrum, axes[1], "Generated", vmin=vmin, vmax=vmax)
    fig.suptitle(f"Real vs generated spectrum: {class_name}")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    _close_figure(fig)
    return path


def export_raw_processed_synthetic_triplet(
    raw: np.ndarray,
    processed: np.ndarray,
    synthetic: np.ndarray,
    path: str | Path,
    *,
    class_name: str,
) -> Path:
    """Save raw real, preprocessed real, and generated spectra in one figure."""

    ims = _import_gcims()
    import matplotlib.pyplot as plt

    raw = np.asarray(raw, dtype=np.float32)
    processed = np.asarray(processed, dtype=np.float32)
    synthetic = np.asarray(synthetic, dtype=np.float32)
    processed_vmin = float(min(np.min(processed), np.min(synthetic)))
    processed_vmax = float(max(np.max(processed), np.max(synthetic)))
    spectra = [
        ims.Spectrum(
            name=f"Raw {class_name}",
            values=raw,
            ret_time=np.arange(raw.shape[0]),
            drift_time=np.arange(raw.shape[1]),
            time=datetime.now(),
            meta_attr={},
        ),
        ims.Spectrum(
            name=f"Preprocessed {class_name}",
            values=processed,
            ret_time=np.arange(processed.shape[0]),
            drift_time=np.arange(processed.shape[1]),
            time=datetime.now(),
            meta_attr={},
        ),
        ims.Spectrum(
            name=f"Generated {class_name}",
            values=synthetic,
            ret_time=np.arange(synthetic.shape[0]),
            drift_time=np.arange(synthetic.shape[1]),
            time=datetime.now(),
            meta_attr={},
        ),
    ]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    _plot_on_axis(spectra[0], axes[0], "Before preprocessing")
    _plot_on_axis(spectra[1], axes[1], "After preprocessing", vmin=processed_vmin, vmax=processed_vmax)
    _plot_on_axis(spectra[2], axes[2], "Generated", vmin=processed_vmin, vmax=processed_vmax)
    fig.suptitle(f"Raw, preprocessed, and generated spectrum: {class_name}")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    _close_figure(fig)
    return path


def _plot_on_axis(spectrum, ax, title: str, *, vmin: float | None = None, vmax: float | None = None) -> None:
    values = spectrum.values
    image = ax.imshow(values, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("Drift time")
    ax.set_ylabel("Retention time")
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def _import_gcims():
    try:
        import ims
    except ImportError as exc:
        raise ImportError(
            "GC-IMS visualization requires gc-ims-tools. Install it with "
            "`pip install gc-ims-tools` or `pip install -r requirements.txt`."
        ) from exc
    return ims


def _close_figure(fig) -> None:
    import matplotlib.pyplot as plt

    plt.close(fig)
