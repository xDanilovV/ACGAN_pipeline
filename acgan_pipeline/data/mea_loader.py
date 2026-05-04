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


def load_mea_folder(
    folder: str | Path,
    labels_csv: str | Path | None = None,
    *,
    config: MeaPreprocessingConfig | None = None,
    label_mode: str = "class",
    target_shape: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    """Load all ``.mea`` files from a folder.

    If ``labels_csv`` is provided, it must contain one filename column and one
    label column. Otherwise labels are inferred from folder structure. For your
    fermentation data this means:

    ``data_fermentation / GCIMS_*_cultures / class_name / batch / sample.mea``.
    """

    folder = Path(folder)
    label_map = _read_label_map(labels_csv) if labels_csv is not None else None
    spectra = []
    labels = []
    metadata = []
    for path in sorted(folder.rglob("*.mea")):
        label = _label_for_path(path, folder, label_map, label_mode=label_mode)
        if label is None:
            raise ValueError(f"missing label for {path.name}")
        values, info = load_mea_file(path, config=config)
        if target_shape is not None:
            values = _resize_array(values, target_shape)
            info["tensor_shape"] = list(values.shape)
        spectra.append(values)
        labels.append(label)
        info["label"] = label
        info["culture_type"] = _culture_type_for_path(path, folder)
        info["batch"] = path.parent.name
        metadata.append(info)

    if not spectra:
        raise ValueError(f"no .mea files found in {folder}")

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
    info = {
        "path": str(path),
        "name": getattr(spectrum, "name", path.stem),
        "original_shape": list(original_shape),
        "processed_shape": list(values.shape),
        "preprocessing": asdict(config),
        "retention_time_min": float(np.min(spectrum.ret_time)),
        "retention_time_max": float(np.max(spectrum.ret_time)),
        "drift_time_min": float(np.min(spectrum.drift_time)),
        "drift_time_max": float(np.max(spectrum.drift_time)),
    }
    return values, info


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


def _import_gcims():
    try:
        import ims
    except ImportError as exc:
        raise ImportError(
            "Reading .mea files requires gc-ims-tools. Install it with "
            "`pip install gc-ims-tools` or `pip install -r requirements.txt`."
        ) from exc
    return ims


def _resize_array(values: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(values, dtype=np.float32))[None, None]
    resized = F.interpolate(tensor, size=target_shape, mode="bilinear", align_corners=False)
    return resized[0, 0].numpy().astype(np.float32)
