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


def _plot_on_axis(spectrum, ax, title: str) -> None:
    values = spectrum.values
    image = ax.imshow(values, aspect="auto", origin="lower")
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
