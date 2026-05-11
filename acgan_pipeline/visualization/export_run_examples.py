from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from acgan_pipeline.config import DEFAULT_CONFIG, load_config
from acgan_pipeline.data.dataset import GCIMSDataset
from acgan_pipeline.data.mea_loader import (
    MeaPreprocessingConfig,
    PeakCropConfig,
    load_mea_folder,
)
from acgan_pipeline.main import _fixed_target_shape, _resolve_target_shape
from acgan_pipeline.visualization.gcims_plots import export_spectrum_plot


def main() -> None:
    parser = argparse.ArgumentParser(description="Export separate raw, processed, and generated spectra from a run.")
    parser.add_argument("--run-dir", required=True, help="Completed AC-GAN output directory containing synthetic_samples.npz.")
    parser.add_argument("--data", required=True, help="Original .mea dataset root used for the run.")
    parser.add_argument("--config", help="Config file used for the run. Defaults to run-dir/effective_config.json when present.")
    parser.add_argument("--output-dir", help="Directory for exported PNGs. Defaults to run-dir/separate_spectra.")
    parser.add_argument("--class-id", type=int, default=0, help="Encoded class id to export.")
    parser.add_argument("--sample-index", type=int, default=0, help="Real sample offset within the selected class.")
    parser.add_argument("--synthetic-index", type=int, default=0, help="Synthetic sample offset within the selected class.")
    parser.add_argument(
        "--normalized-synthetic",
        action="store_true",
        help="Export generated tensor in normalized [-1, 1] scale instead of denormalizing to the processed real scale.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.output_dir) if args.output_dir else run_dir / "separate_spectra"
    out_dir.mkdir(parents=True, exist_ok=True)

    config = _load_run_config(run_dir, args.config)
    config["data"] = args.data
    config["output_dir"] = str(run_dir)
    ns = SimpleNamespace(**config)

    if ns.input_format != "mea":
        raise ValueError("separate run example export currently expects input_format='mea'")

    mea_config = MeaPreprocessingConfig(
        drift_start=ns.rip_drift_start if ns.rip_drift_start is not None else 1.05,
        drift_stop=ns.rip_drift_stop,
        retention_start=ns.crop_rt_start,
        retention_stop=ns.crop_rt_stop,
        intensity_baseline_percentile=ns.intensity_baseline_percentile,
        intensity_clip_low_percentile=ns.intensity_clip_low_percentile,
        intensity_clip_high_percentile=ns.intensity_clip_high_percentile,
        intensity_log1p=ns.intensity_log1p,
        intensity_percentile_max_pixels=ns.intensity_percentile_max_pixels,
    )
    peak_crop = PeakCropConfig(
        enabled=ns.peak_crop,
        percentile=ns.peak_percentile,
        relative_threshold=ns.peak_relative_threshold,
        support_fraction=ns.peak_support_fraction,
        margin_rt=ns.peak_margin_rt,
        margin_dt=ns.peak_margin_dt,
        percentile_max_pixels=ns.peak_percentile_max_pixels,
    )

    fixed_shape = _fixed_target_shape(ns)
    processed_samples, labels, metadata = load_mea_folder(
        args.data,
        ns.labels_csv,
        config=mea_config,
        label_mode=ns.mea_label_mode,
        target_shape=fixed_shape,
        resize_mode=ns.resize_mode,
        peak_crop=peak_crop,
    )
    target_shape = _resolve_target_shape(processed_samples, ns, fixed_shape)
    real_indices = np.flatnonzero(labels == args.class_id)
    if real_indices.size == 0:
        raise ValueError(f"class_id {args.class_id} was not found in real labels")
    if args.sample_index >= real_indices.size:
        raise ValueError(f"sample_index {args.sample_index} exceeds {real_indices.size} real samples in class {args.class_id}")
    real_idx = int(real_indices[args.sample_index])

    synthetic_path = run_dir / "synthetic_samples.npz"
    synthetic = np.load(synthetic_path)
    synthetic_samples = synthetic["samples"]
    synthetic_labels = synthetic["labels"]
    synthetic_indices = np.flatnonzero(synthetic_labels == args.class_id)
    if synthetic_indices.size == 0:
        raise ValueError(f"class_id {args.class_id} was not found in synthetic labels")
    if args.synthetic_index >= synthetic_indices.size:
        raise ValueError(
            f"synthetic_index {args.synthetic_index} exceeds {synthetic_indices.size} synthetic samples in class {args.class_id}"
        )
    synthetic_idx = int(synthetic_indices[args.synthetic_index])

    real_path = Path(str(metadata[real_idx]["path"]))
    class_name = _class_name(metadata, args.class_id)
    raw_output = out_dir / "01_raw_native_ims.png"
    processed_output = out_dir / "02_processed_sample.png"
    generated_output = out_dir / "03_generated_sample.png"

    _export_native_mea_plot(real_path, raw_output, title=f"Raw native spectrum: {class_name}")
    export_spectrum_plot(
        processed_samples[real_idx],
        processed_output,
        title=f"Processed spectrum: {class_name}",
    )

    generated = np.asarray(synthetic_samples[synthetic_idx], dtype=np.float32)
    if not args.normalized_synthetic:
        dataset = GCIMSDataset(
            processed_samples,
            labels,
            target_shape=target_shape,
            resize_mode=ns.resize_mode,
        )
        generated = ((generated + 1.0) / 2.0) * (dataset.max_value - dataset.min_value) + dataset.min_value
    export_spectrum_plot(
        generated,
        generated_output,
        title=f"Generated spectrum: {class_name}",
    )

    manifest = {
        "class_id": args.class_id,
        "class_name": class_name,
        "real_index": real_idx,
        "synthetic_index": synthetic_idx,
        "real_path": str(real_path),
        "target_shape": list(target_shape),
        "normalized_synthetic": bool(args.normalized_synthetic),
        "outputs": {
            "raw_native": str(raw_output),
            "processed": str(processed_output),
            "generated": str(generated_output),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved separate spectrum PNGs to {out_dir.resolve()}")


def _load_run_config(run_dir: Path, config_path: str | None) -> dict[str, Any]:
    if config_path is not None:
        return load_config(config_path)
    config = DEFAULT_CONFIG.copy()
    effective_path = run_dir / "effective_config.json"
    if effective_path.exists():
        effective = json.loads(effective_path.read_text(encoding="utf-8"))
        config.update({key: value for key, value in effective.items() if key in DEFAULT_CONFIG})
    return config


def _export_native_mea_plot(path: Path, output_path: Path, *, title: str) -> None:
    try:
        import ims
    except ImportError as exc:
        raise ImportError(
            "Raw native GC-IMS plotting requires gc-ims-tools. Install it with "
            "`pip install gc-ims-tools` or `pip install -r requirements.txt`."
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    spectrum = ims.Spectrum.read_mea(str(path))
    fig, ax = spectrum.plot()
    ax.set_title(title)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    _close_figure(fig)


def _class_name(metadata: list[dict[str, object]], class_id: int) -> str:
    if metadata:
        mapping = metadata[0].get("label_mapping", {})
        inverse = {int(value): str(key) for key, value in mapping.items()}
        return inverse.get(class_id, f"class {class_id}")
    return f"class {class_id}"


def _close_figure(fig) -> None:
    import matplotlib.pyplot as plt

    plt.close(fig)


if __name__ == "__main__":
    main()
