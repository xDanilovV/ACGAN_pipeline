from __future__ import annotations

import argparse
import json
from pathlib import Path

from acgan_pipeline.data.mea_loader import (
    MeaPreprocessingConfig,
    PeakCropConfig,
    _apply_index_crop,
    _resize_array,
    compute_shared_peak_crop,
    load_mea_file,
)
from acgan_pipeline.visualization.gcims_plots import export_preprocessing_comparison, export_spectrum_plot


def main() -> None:
    parser = argparse.ArgumentParser(description="Export GC-IMS preprocessing resize previews.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", default="outputs_preprocessing_preview")
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--resize-mode", choices=["area", "bilinear", "bicubic", "nearest"], default="area")
    parser.add_argument("--rip-drift-start", type=float, default=1.05)
    parser.add_argument("--rip-drift-stop", type=float, default=None)
    parser.add_argument("--crop-rt-start", type=float, default=None)
    parser.add_argument("--crop-rt-stop", type=float, default=None)
    parser.add_argument("--peak-crop", action="store_true")
    parser.add_argument("--peak-percentile", type=float, default=99.7)
    parser.add_argument("--peak-relative-threshold", type=float, default=0.05)
    parser.add_argument("--peak-support-fraction", type=float, default=0.03)
    parser.add_argument("--peak-margin-rt", type=int, default=256)
    parser.add_argument("--peak-margin-dt", type=int, default=48)
    args = parser.parse_args()

    root = Path(args.data)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = next(root.rglob("*.mea"))
    mea_config = MeaPreprocessingConfig(
        drift_start=args.rip_drift_start,
        drift_stop=args.rip_drift_stop,
        retention_start=args.crop_rt_start,
        retention_stop=args.crop_rt_stop,
    )
    values, _ = load_mea_file(
        path,
        config=mea_config,
    )
    if args.peak_crop:
        shared_crop, crop_report = compute_shared_peak_crop(
            sorted(root.rglob("*.mea")),
            config=mea_config,
            crop_config=PeakCropConfig(
                enabled=True,
                percentile=args.peak_percentile,
                relative_threshold=args.peak_relative_threshold,
                support_fraction=args.peak_support_fraction,
                margin_rt=args.peak_margin_rt,
                margin_dt=args.peak_margin_dt,
            ),
        )
        values = _apply_index_crop(values, shared_crop)
        (out / "peak_crop_report.json").write_text(json.dumps(crop_report, indent=2), encoding="utf-8")
    resized = _resize_array(values, (args.height, args.width), resize_mode=args.resize_mode)
    export_spectrum_plot(values, out / "processed_original_resolution.png", title=f"Processed original: {path.name}")
    export_spectrum_plot(resized, out / f"resized_{args.height}x{args.width}_{args.resize_mode}.png", title=f"Resized {args.height}x{args.width}")
    export_preprocessing_comparison(
        values[:: max(values.shape[0] // args.height, 1), :: max(values.shape[1] // args.width, 1)],
        resized,
        out / "comparison.png",
        title="Processed original sample vs GAN tensor",
    )
    print(f"Saved preview images to {out.resolve()}")


if __name__ == "__main__":
    main()
