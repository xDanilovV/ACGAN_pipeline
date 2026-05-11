from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class MeaPreprocessingConfig:
    """Preprocessing applied while reading native G.A.S Dortmund ``.mea`` files.

    gc-ims-tools exposes common GC-IMS operations directly on ``ims.Spectrum``.
    The default RIP handling uses ``riprel().cut_dt(1.05, None)``: convert drift
    time into RIP-relative coordinates, then keep the area just after the RIP.
    This mirrors the documented package workflow and keeps the setting easy to
    tune once we inspect your spectra.
    """

    rip_relative: bool = True
    drift_start: float | None = 1.05
    drift_stop: float | None = None
    retention_start: float | None = None
    retention_stop: float | None = None
    intensity_baseline_percentile: float | None = None
    intensity_clip_low_percentile: float | None = None
    intensity_clip_high_percentile: float | None = None
    intensity_log1p: bool = False
    intensity_percentile_max_pixels: int = 250_000


@dataclass(frozen=True)
class PeakCropConfig:
    """Shared peak-aware crop computed from all spectra before resizing.

    The crop is not chosen independently for each sample. We first detect rows
    and columns that repeatedly contain high-intensity peak support across the
    dataset, then apply one shared bounding box to every sample.
    """

    enabled: bool = False
    percentile: float = 99.7
    relative_threshold: float = 0.05
    support_fraction: float = 0.03
    margin_rt: int = 256
    margin_dt: int = 48
    percentile_max_pixels: int = 250_000


def load_mea_folder(
    folder: str | Path,
    labels_csv: str | Path | None = None,
    *,
    config: MeaPreprocessingConfig | None = None,
    label_mode: str = "class",
    target_shape: tuple[int, int] | None = None,
    resize_mode: str = "area",
    peak_crop: PeakCropConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    """Load all ``.mea`` files from a folder.

    If ``labels_csv`` is provided, it must contain one filename column and one
    label column. Otherwise labels are inferred from folder structure. For your
    fermentation data this means:

    ``data_fermentation / GCIMS_*_cultures / class_name / batch / sample.mea``.
    """

    folder = Path(folder)
    label_map = _read_label_map(labels_csv) if labels_csv is not None else None
    paths = sorted(folder.rglob("*.mea"))
    if not paths:
        raise ValueError(f"no .mea files found in {folder}")

    peak_crop = peak_crop or PeakCropConfig(enabled=False)
    shared_crop = None
    crop_report = None
    if peak_crop.enabled:
        shared_crop, crop_report = compute_shared_peak_crop(paths, config=config, crop_config=peak_crop)

    spectra = []
    labels = []
    metadata = []
    for path in paths:
        label = _label_for_path(path, folder, label_map, label_mode=label_mode)
        if label is None:
            raise ValueError(f"missing label for {path.name}")
        values, info = load_mea_file(path, config=config)
        if shared_crop is not None:
            values = _apply_index_crop(values, shared_crop)
            info["peak_crop_shape"] = list(values.shape)
            info["peak_crop"] = crop_report
        if target_shape is not None:
            values = _resize_array(values, target_shape, resize_mode=resize_mode)
            info["tensor_shape"] = list(values.shape)
            info["resize_mode"] = resize_mode
        spectra.append(values)
        labels.append(label)
        info["label"] = label
        info["culture_type"] = _culture_type_for_path(path, folder)
        info["batch"] = path.parent.name
        metadata.append(info)

    encoded_labels, label_mapping = _encode_labels(labels)
    for item in metadata:
        item["label_mapping"] = label_mapping
    return np.stack(spectra).astype(np.float32), encoded_labels, metadata


def load_mea_file(
    path: str | Path,
    *,
    config: MeaPreprocessingConfig | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Read and preprocess one ``.mea`` file with gc-ims-tools."""

    ims = _import_gcims()
    path = Path(path)
    config = config or MeaPreprocessingConfig()
    spectrum = ims.Spectrum.read_mea(str(path))
    original_shape = tuple(int(v) for v in spectrum.values.shape)

    if config.rip_relative:
        spectrum.riprel()
    if config.drift_start is not None or config.drift_stop is not None:
        spectrum.cut_dt(config.drift_start if config.drift_start is not None else spectrum.drift_time[0], config.drift_stop)
    if config.retention_start is not None or config.retention_stop is not None:
        spectrum.cut_rt(
            config.retention_start if config.retention_start is not None else spectrum.ret_time[0],
            config.retention_stop,
        )

    values = np.asarray(spectrum.values, dtype=np.float32)
    raw_min = float(np.min(values))
    raw_max = float(np.max(values))
    values = _transform_intensity(values, config)
    info = {
        "path": str(path),
        "name": getattr(spectrum, "name", path.stem),
        "original_shape": list(original_shape),
        "processed_shape": list(values.shape),
        "preprocessing": asdict(config),
        "raw_intensity_min": raw_min,
        "raw_intensity_max": raw_max,
        "processed_intensity_min": float(np.min(values)),
        "processed_intensity_max": float(np.max(values)),
        "retention_time_min": float(np.min(spectrum.ret_time)),
        "retention_time_max": float(np.max(spectrum.ret_time)),
        "drift_time_min": float(np.min(spectrum.drift_time)),
        "drift_time_max": float(np.max(spectrum.drift_time)),
    }
    return values, info


def compute_shared_peak_crop(
    paths: list[Path],
    *,
    config: MeaPreprocessingConfig | None,
    crop_config: PeakCropConfig,
) -> tuple[tuple[int, int, int, int], dict[str, object]]:
    """Compute one shared index crop from recurring high-intensity support."""

    row_support: np.ndarray | None = None
    col_support: np.ndarray | None = None
    processed_shapes = []
    contributing_files = 0
    for path in paths:
        values, _ = load_mea_file(path, config=config)
        processed_shapes.append(list(values.shape))
        threshold = max(
            float(_fast_percentile(values, crop_config.percentile, crop_config.percentile_max_pixels)),
            float(np.max(values)) * crop_config.relative_threshold,
        )
        mask = values >= threshold
        if not np.any(mask):
            continue
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        row_support = _add_support(row_support, rows)
        col_support = _add_support(col_support, cols)
        contributing_files += 1

    if row_support is None or col_support is None:
        raise ValueError("peak crop failed: no peak support detected")

    min_support = max(1, int(np.ceil(len(paths) * crop_config.support_fraction)))
    row_indices = np.flatnonzero(row_support >= min_support)
    col_indices = np.flatnonzero(col_support >= min_support)
    if row_indices.size == 0 or col_indices.size == 0:
        raise ValueError("peak crop failed: support threshold was too strict")

    row_start = max(0, int(row_indices[0]) - crop_config.margin_rt)
    row_stop = min(len(row_support), int(row_indices[-1]) + crop_config.margin_rt + 1)
    col_start = max(0, int(col_indices[0]) - crop_config.margin_dt)
    col_stop = min(len(col_support), int(col_indices[-1]) + crop_config.margin_dt + 1)
    crop = (row_start, row_stop, col_start, col_stop)
    report = {
        "enabled": True,
        "config": asdict(crop_config),
        "crop_indices": {
            "retention_start": row_start,
            "retention_stop": row_stop,
            "drift_start": col_start,
            "drift_stop": col_stop,
        },
        "crop_shape": [row_stop - row_start, col_stop - col_start],
        "num_files": len(paths),
        "contributing_files": contributing_files,
        "min_support_count": min_support,
        "processed_shape_examples": processed_shapes[:5],
    }
    return crop, report


def _read_label_map(labels_csv: str | Path) -> dict[str, str]:
    labels_csv = Path(labels_csv)
    with labels_csv.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("labels CSV must have a header")
        file_col = _first_existing(reader.fieldnames, ["file", "filename", "path", "sample", "name"])
        label_col = _first_existing(reader.fieldnames, ["label", "class", "target", "group"])
        if file_col is None or label_col is None:
            raise ValueError("labels CSV needs file/filename/path and label/class columns")
        return {row[file_col]: row[label_col] for row in reader}


def _label_for_path(
    path: Path,
    root: Path,
    label_map: dict[str, str] | None,
    *,
    label_mode: str = "class",
) -> str | None:
    if label_map is not None:
        rel = str(path.relative_to(root))
        return label_map.get(path.name, label_map.get(path.stem, label_map.get(rel)))

    parts = path.relative_to(root).parts
    if len(parts) < 2:
        return None
    if label_mode == "culture_type":
        return parts[0] if parts[0] in {"GCIMS_mixed_cultures", "GCIMS_pure_cultures"} else None
    if label_mode != "class":
        raise ValueError("label_mode must be 'class' or 'culture_type'")
    if parts[0] in {"GCIMS_mixed_cultures", "GCIMS_pure_cultures"} and len(parts) >= 3:
        return parts[1]
    return parts[0]


def _culture_type_for_path(path: Path, root: Path) -> str | None:
    parts = path.relative_to(root).parts
    if parts and parts[0] in {"GCIMS_mixed_cultures", "GCIMS_pure_cultures"}:
        return parts[0]
    return None


def _first_existing(fieldnames: list[str], candidates: list[str]) -> str | None:
    lowered = {name.lower(): name for name in fieldnames}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def _encode_labels(labels: list[str]) -> tuple[np.ndarray, dict[str, int]]:
    unique = sorted(set(labels))
    mapping = {label: index for index, label in enumerate(unique)}
    return np.asarray([mapping[label] for label in labels], dtype=np.int64), mapping


def _add_support(existing: np.ndarray | None, support: np.ndarray) -> np.ndarray:
    support_int = support.astype(np.int32)
    if existing is None:
        return support_int
    if len(support_int) > len(existing):
        padded = np.zeros(len(support_int), dtype=np.int32)
        padded[: len(existing)] = existing
        existing = padded
    existing[: len(support_int)] += support_int
    return existing


def _apply_index_crop(values: np.ndarray, crop: tuple[int, int, int, int]) -> np.ndarray:
    row_start, row_stop, col_start, col_stop = crop
    out_shape = (max(0, row_stop - row_start), max(0, col_stop - col_start))
    cropped = np.zeros(out_shape, dtype=values.dtype)

    src_row_start = max(0, row_start)
    src_col_start = max(0, col_start)
    src_row_stop = min(row_stop, values.shape[0])
    src_col_stop = min(col_stop, values.shape[1])
    if src_row_stop <= src_row_start or src_col_stop <= src_col_start:
        return cropped

    dst_row_start = src_row_start - row_start
    dst_col_start = src_col_start - col_start
    dst_row_stop = dst_row_start + (src_row_stop - src_row_start)
    dst_col_stop = dst_col_start + (src_col_stop - src_col_start)
    cropped[dst_row_start:dst_row_stop, dst_col_start:dst_col_stop] = values[
        src_row_start:src_row_stop,
        src_col_start:src_col_stop,
    ]
    return cropped


def _transform_intensity(values: np.ndarray, config: MeaPreprocessingConfig) -> np.ndarray:
    transformed = np.asarray(values, dtype=np.float32)
    if config.intensity_baseline_percentile is not None:
        baseline = _fast_percentile(
            transformed,
            config.intensity_baseline_percentile,
            config.intensity_percentile_max_pixels,
        )
        transformed = transformed - np.float32(baseline)

    if config.intensity_log1p or config.intensity_baseline_percentile is not None:
        transformed = np.maximum(transformed, 0.0)

    if config.intensity_clip_low_percentile is not None or config.intensity_clip_high_percentile is not None:
        clip_min = None
        clip_max = None
        if config.intensity_clip_low_percentile is not None:
            clip_min = _fast_percentile(
                transformed,
                config.intensity_clip_low_percentile,
                config.intensity_percentile_max_pixels,
            )
        if config.intensity_clip_high_percentile is not None:
            clip_max = _fast_percentile(
                transformed,
                config.intensity_clip_high_percentile,
                config.intensity_percentile_max_pixels,
            )
        transformed = np.clip(transformed, clip_min, clip_max)

    if config.intensity_log1p:
        transformed = np.log1p(np.maximum(transformed, 0.0))

    return transformed.astype(np.float32, copy=False)


def _fast_percentile(values: np.ndarray, percentile: float, max_pixels: int) -> float:
    flat = np.ravel(values)
    if max_pixels > 0 and flat.size > max_pixels:
        stride = int(np.ceil(flat.size / max_pixels))
        flat = flat[::stride]
    return float(np.percentile(flat, percentile))


def _import_gcims():
    try:
        import ims
    except ImportError as exc:
        raise ImportError(
            "Reading .mea files requires gc-ims-tools. Install it with "
            "`pip install gc-ims-tools` or `pip install -r requirements.txt`."
        ) from exc
    return ims


def _resize_array(values: np.ndarray, target_shape: tuple[int, int], *, resize_mode: str = "area") -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(values, dtype=np.float32))[None, None]
    kwargs = {"size": target_shape, "mode": resize_mode}
    if resize_mode in {"linear", "bilinear", "bicubic", "trilinear"}:
        kwargs["align_corners"] = False
    resized = F.interpolate(tensor, **kwargs)
    return resized[0, 0].numpy().astype(np.float32)
