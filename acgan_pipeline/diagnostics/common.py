from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from acgan_pipeline.config import load_config
from acgan_pipeline.data.dataset import GCIMSDataset
from acgan_pipeline.data.mea_loader import MeaPreprocessingConfig, PeakCropConfig, load_mea_folder
from acgan_pipeline.main import (
    _fixed_target_shape,
    _label_summary,
    _resolve_target_shape,
    load_npz_dataset,
)
from acgan_pipeline.preprocessing import PreprocessingConfig, preprocess_dataset


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)


def load_experiment_inputs(config_path: str | None, data_path: str) -> tuple[np.ndarray, np.ndarray, tuple[int, int], dict[str, Any], SimpleNamespace]:
    config = load_config(config_path)
    args = SimpleNamespace(**config)
    args.data = data_path
    initial_target_shape = _fixed_target_shape(args)

    if args.input_format == "mea":
        mea_config = MeaPreprocessingConfig(
            drift_start=args.rip_drift_start if args.rip_drift_start is not None else 1.05,
            drift_stop=args.rip_drift_stop,
            retention_start=args.crop_rt_start,
            retention_stop=args.crop_rt_stop,
            intensity_baseline_percentile=args.intensity_baseline_percentile,
            intensity_clip_low_percentile=args.intensity_clip_low_percentile,
            intensity_clip_high_percentile=args.intensity_clip_high_percentile,
            intensity_log1p=args.intensity_log1p,
            intensity_percentile_max_pixels=args.intensity_percentile_max_pixels,
        )
        peak_crop_config = PeakCropConfig(
            enabled=args.peak_crop,
            percentile=args.peak_percentile,
            relative_threshold=args.peak_relative_threshold,
            support_fraction=args.peak_support_fraction,
            margin_rt=args.peak_margin_rt,
            margin_dt=args.peak_margin_dt,
            percentile_max_pixels=args.peak_percentile_max_pixels,
        )
        samples, labels, metadata = load_mea_folder(
            data_path,
            args.labels_csv,
            config=mea_config,
            label_mode=args.mea_label_mode,
            target_shape=initial_target_shape,
            resize_mode=args.resize_mode,
            peak_crop=peak_crop_config,
        )
        report: dict[str, Any] = {"mea_metadata": metadata, "note": "Loaded and preprocessed with gc-ims-tools."}
    else:
        raw_samples, labels = load_npz_dataset(data_path)
        preprocessing_config = PreprocessingConfig(
            crop_rt=_optional_int_range(args.crop_rt_start, args.crop_rt_stop),
            keep_drift=_optional_int_range(args.rip_drift_start, args.rip_drift_stop),
        )
        samples, report = preprocess_dataset(raw_samples, preprocessing_config)

    target_shape = _resolve_target_shape(samples, args, initial_target_shape)
    report["target_shape"] = list(target_shape)
    report["shape_mode"] = args.shape_mode
    report["label_summary"] = _label_summary(labels, report)
    return samples, labels, target_shape, report, args


def normalized_dataset_arrays(
    samples: np.ndarray,
    labels: np.ndarray,
    *,
    target_shape: tuple[int, int],
    resize_mode: str,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    dataset = GCIMSDataset(samples, labels, target_shape=target_shape, resize_mode=resize_mode)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for batch, batch_labels in loader:
        xs.append(batch[:, 0].numpy())
        ys.append(batch_labels.numpy())
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0), dataset.min_value, dataset.max_value


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _optional_int_range(start: float | None, stop: float | None) -> tuple[int | None, int | None]:
    return None if start is None else int(start), None if stop is None else int(stop)
