from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from acgan_pipeline.data.dataset import GCIMSDataset
from acgan_pipeline.data.mea_loader import MeaPreprocessingConfig, load_mea_folder
from acgan_pipeline.evaluation import run_core_evaluation_suite
from acgan_pipeline.preprocessing import PreprocessingConfig, preprocess_dataset
from acgan_pipeline.training.train_acgan import TrainConfig, generate_samples, train_acgan
from acgan_pipeline.visualization.gcims_plots import export_preprocessing_comparison


def load_npz_dataset(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Example loader for ``.npz`` files containing ``samples`` and ``labels``.

    Replace this function with a project-specific loader for MATLAB files,
    vendor exports, HDF5, CSV folders, or other GC-IMS storage formats.
    """

    data = np.load(path)
    return data["samples"], data["labels"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reusable AC-GAN pipeline for 2D GC-IMS-like data.")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset file consumed by the selected loader.")
    parser.add_argument("--labels-csv", type=str, default=None, help="Optional CSV labels file for .mea folders. If omitted, labels are inferred from class folders.")
    parser.add_argument("--mea-label-mode", choices=["class", "culture_type"], default="class")
    parser.add_argument("--input-format", choices=["npz", "mea"], default="npz")
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--noise-dim", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--samples-per-class", type=int, default=100)
    parser.add_argument("--rip-drift-start", type=float, default=None, help="First RIP-relative drift-time value to keep.")
    parser.add_argument("--rip-drift-stop", type=float, default=None, help="Last RIP-relative drift-time value to keep.")
    parser.add_argument("--crop-rt-start", type=float, default=None)
    parser.add_argument("--crop-rt-stop", type=float, default=None)
    parser.add_argument("--eval-epochs", type=int, default=25)
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--skip-visualization", action="store_true")
    parser.add_argument("--synthetic-viz-denormalized", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.input_format == "mea":
        mea_config = MeaPreprocessingConfig(
            drift_start=args.rip_drift_start if args.rip_drift_start is not None else 1.05,
            drift_stop=args.rip_drift_stop,
            retention_start=args.crop_rt_start,
            retention_stop=args.crop_rt_stop,
        )
        processed_samples, labels, mea_metadata = load_mea_folder(
            args.data,
            args.labels_csv,
            config=mea_config,
            label_mode=args.mea_label_mode,
            target_shape=(args.height, args.width),
        )
        raw_samples = processed_samples
        preprocessing_report = {"mea_metadata": mea_metadata, "note": "Loaded and preprocessed with gc-ims-tools."}
    else:
        raw_samples, labels = load_npz_dataset(args.data)
        preprocessing_config = PreprocessingConfig(
            crop_rt=_optional_int_range(args.crop_rt_start, args.crop_rt_stop),
            keep_drift=_optional_int_range(args.rip_drift_start, args.rip_drift_stop),
        )
        processed_samples, preprocessing_report = preprocess_dataset(raw_samples, preprocessing_config)
    with (output_dir / "preprocessing_report.json").open("w", encoding="utf-8") as f:
        json.dump(preprocessing_report, f, indent=2)

    if not args.skip_visualization:
        export_preprocessing_comparison(
            raw_samples[0],
            processed_samples[0],
            output_dir / "preprocessing_examples" / "real_sample_000.png",
            title="Real spectrum preprocessing",
        )

    dataset = GCIMSDataset(
        processed_samples,
        labels,
        target_shape=(args.height, args.width),
    )
    config = TrainConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        noise_dim=args.noise_dim,
        output_dir=args.output_dir,
    )
    generator, _, _ = train_acgan(
        dataset,
        num_classes=dataset.num_classes,
        image_shape=(args.height, args.width),
        config=config,
    )
    synthetic_samples, synthetic_labels = generate_samples(
        generator,
        args.samples_per_class,
        num_classes=dataset.num_classes,
        noise_dim=args.noise_dim,
    )
    output_path = Path(args.output_dir) / "synthetic_samples.npz"
    np.savez_compressed(output_path, samples=synthetic_samples, labels=synthetic_labels)
    print(f"Saved synthetic samples to {output_path}")

    if not args.skip_visualization:
        synthetic_for_plot = synthetic_samples[0]
        if args.synthetic_viz_denormalized:
            synthetic_for_plot = _denormalize_for_visualization(synthetic_for_plot, processed_samples)
            denormalized = _denormalize_for_visualization(synthetic_samples, processed_samples)
            np.savez_compressed(
                output_dir / "synthetic_samples_denormalized_for_visualization.npz",
                samples=denormalized,
                labels=synthetic_labels,
            )
        export_preprocessing_comparison(
            synthetic_for_plot,
            synthetic_for_plot,
            output_dir / "preprocessing_examples" / "synthetic_sample_000.png",
            title="Synthetic AC-GAN spectrum",
        )

    if not args.skip_evaluation:
        evaluation = run_core_evaluation_suite(
            real_samples=processed_samples,
            real_labels=labels,
            synthetic_samples=synthetic_samples,
            synthetic_labels=synthetic_labels,
            image_shape=(args.height, args.width),
            num_epochs=args.eval_epochs,
            output_dir=output_dir / "evaluation",
        )
        print(json.dumps(evaluation, indent=2))


def _denormalize_for_visualization(sample: np.ndarray, reference_samples: np.ndarray) -> np.ndarray:
    min_value = float(np.min(reference_samples))
    max_value = float(np.max(reference_samples))
    return ((sample + 1.0) / 2.0) * (max_value - min_value) + min_value


def _optional_int_range(start: float | None, stop: float | None) -> tuple[int | None, int | None]:
    return None if start is None else int(start), None if stop is None else int(stop)


if __name__ == "__main__":
    main()
