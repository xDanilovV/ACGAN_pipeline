from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_CONFIG: dict[str, Any] = {
    "input_format": "mea",
    "labels_csv": None,
    "mea_label_mode": "class",
    "shape_mode": "auto",
    "height": None,
    "width": None,
    "resize_mode": "area",
    "auto_max_pixels": 65536,
    "auto_max_height": 512,
    "auto_max_width": 256,
    "auto_multiple": 16,
    "epochs": 60,
    "batch_size": 16,
    "noise_dim": 100,
    "lr": 2e-4,
    "lr_g": 2e-4,
    "lr_d": 2e-5,
    "class_loss_weight": 1.0,
    "tv_loss_weight": 1e-4,
    "label_smoothing": 0.1,
    "instance_noise_std": 0.1,
    "instance_noise_decay_epochs": 30,
    "projection_scale": 0.0,
    "generator_base_channels": 256,
    "discriminator_base_channels": 64,
    "discriminator_use_norm": False,
    "discriminator_use_spectral_norm": False,
    "discriminator_pool_height": 8,
    "discriminator_pool_width": 4,
    "discriminator_input_pool_height": 32,
    "discriminator_input_pool_width": 8,
    "discriminator_dropout": 0.0,
    "pretrain_classifier_epochs": 30,
    "pretrain_classifier_lr": 1e-3,
    "generator_steps": 1,
    "discriminator_update_every": 1,
    "sample_every": 5,
    "checkpoint_every": 5,
    "seed": 42,
    "output_dir": "outputs",
    "samples_per_class": 5,
    "rip_drift_start": None,
    "rip_drift_stop": None,
    "crop_rt_start": None,
    "crop_rt_stop": None,
    "peak_crop": True,
    "peak_percentile": 99.7,
    "peak_relative_threshold": 0.05,
    "peak_support_fraction": 0.03,
    "peak_margin_rt": 256,
    "peak_margin_dt": 48,
    "eval_epochs": 25,
    "test_fraction": 0.2,
    "classifier": "svm",
    "generator_checkpoint": None,
    "skip_evaluation": False,
    "skip_visualization": False,
    "synthetic_viz_denormalized": True,
    "viz_class_id": 0,
}


def load_config(path: str | Path | None) -> dict[str, Any]:
    config = DEFAULT_CONFIG.copy()
    if path is None:
        return config
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        loaded = json.load(f)
    unknown = sorted(set(loaded) - set(DEFAULT_CONFIG))
    if unknown:
        raise ValueError(f"unknown config keys in {config_path}: {unknown}")
    config.update(loaded)
    return config
