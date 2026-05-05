from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from acgan_pipeline.data.dataset import GCIMSDataset
from acgan_pipeline.data.mea_loader import MeaPreprocessingConfig, PeakCropConfig, load_mea_folder
from acgan_pipeline.evaluation import run_core_evaluation_suite, stratified_train_test_split
from acgan_pipeline.preprocessing import PreprocessingConfig, preprocess_dataset
from acgan_pipeline.training.train_acgan import TrainConfig, generate_samples, load_generator_from_checkpoint, train_acgan
from acgan_pipeline.visualization.gcims_plots import export_preprocessing_comparison, export_real_vs_generated_comparison


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
    parser.add_argument("--resize-mode", choices=["area", "bilinear", "bicubic", "nearest"], default="area")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--noise-dim", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--class-loss-weight", type=float, default=1.0)
    parser.add_argument("--tv-loss-weight", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--instance-noise-std", type=float, default=0.0)
    parser.add_argument("--instance-noise-decay-epochs", type=int, default=50)
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--samples-per-class", type=int, default=100)
    parser.add_argument("--rip-drift-start", type=float, default=None, help="First RIP-relative drift-time value to keep.")
    parser.add_argument("--rip-drift-stop", type=float, default=None, help="Last RIP-relative drift-time value to keep.")
    parser.add_argument("--crop-rt-start", type=float, default=None)
    parser.add_argument("--crop-rt-stop", type=float, default=None)
    parser.add_argument("--peak-crop", action="store_true", help="Compute a shared peak-aware crop before resizing.")
    parser.add_argument("--peak-percentile", type=float, default=99.7)
    parser.add_argument("--peak-relative-threshold", type=float, default=0.05)
    parser.add_argument("--peak-support-fraction", type=float, default=0.03)
    parser.add_argument("--peak-margin-rt", type=int, default=256)
    parser.add_argument("--peak-margin-dt", type=int, default=48)
    parser.add_argument("--eval-epochs", type=int, default=25)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--classifier", choices=["svm", "cnn"], default="svm", help="Downstream evaluation classifier.")
    parser.add_argument("--generator-checkpoint", type=str, default=None, help="Load a saved generator checkpoint and skip AC-GAN training.")
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--skip-visualization", action="store_true")
    parser.add_argument("--synthetic-viz-denormalized", action="store_true")
    parser.add_argument("--viz-class-id", type=int, default=0, help="Class id for the real-vs-generated example PNG.")
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
        peak_crop_config = PeakCropConfig(
            enabled=args.peak_crop,
            percentile=args.peak_percentile,
            relative_threshold=args.peak_relative_threshold,
            support_fraction=args.peak_support_fraction,
            margin_rt=args.peak_margin_rt,
            margin_dt=args.peak_margin_dt,
        )
        processed_samples, labels, mea_metadata = load_mea_folder(
            args.data,
            args.labels_csv,
            config=mea_config,
            label_mode=args.mea_label_mode,
            target_shape=(args.height, args.width),
            resize_mode=args.resize_mode,
            peak_crop=peak_crop_config,
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

    train_indices = None
    test_indices = None
    gan_samples = processed_samples
    gan_labels = labels
    if not args.skip_evaluation:
        train_indices, test_indices = stratified_train_test_split(labels, args.test_fraction, args.seed)
        gan_samples = processed_samples[train_indices]
        gan_labels = labels[train_indices]

    dataset = GCIMSDataset(
        gan_samples,
        gan_labels,
        target_shape=(args.height, args.width),
        resize_mode=args.resize_mode,
    )
    config = TrainConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        noise_dim=args.noise_dim,
        lr=args.lr,
        class_loss_weight=args.class_loss_weight,
        tv_loss_weight=args.tv_loss_weight,
        label_smoothing=args.label_smoothing,
        instance_noise_std=args.instance_noise_std,
        instance_noise_decay_epochs=args.instance_noise_decay_epochs,
        sample_every=args.sample_every,
        checkpoint_every=args.checkpoint_every,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    if args.generator_checkpoint is not None:
        generator = load_generator_from_checkpoint(
            args.generator_checkpoint,
            num_classes=dataset.num_classes,
            noise_dim=args.noise_dim,
            image_shape=(args.height, args.width),
        )
        print(f"Loaded generator from {args.generator_checkpoint}")
    else:
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
        class_id = min(max(args.viz_class_id, 0), dataset.num_classes - 1)
        synthetic_for_plot = synthetic_samples[0]
        if args.synthetic_viz_denormalized:
            synthetic_for_plot = _denormalize_for_visualization(synthetic_for_plot, dataset.min_value, dataset.max_value)
            denormalized = _denormalize_for_visualization(synthetic_samples, dataset.min_value, dataset.max_value)
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
        real_idx = int(np.flatnonzero(labels == class_id)[0])
        synthetic_idx = int(np.flatnonzero(synthetic_labels == class_id)[0])
        generated_for_comparison = synthetic_samples[synthetic_idx]
        if args.synthetic_viz_denormalized:
            generated_for_comparison = _denormalize_for_visualization(
                generated_for_comparison,
                dataset.min_value,
                dataset.max_value,
            )
        class_name = _class_name_from_report(preprocessing_report, class_id)
        export_real_vs_generated_comparison(
            processed_samples[real_idx],
            generated_for_comparison,
            output_dir / "preprocessing_examples" / f"real_vs_generated_class_{class_id}.png",
            class_name=class_name,
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
            classifier_type=args.classifier,
            seed=args.seed,
            normalization_min=dataset.min_value,
            normalization_max=dataset.max_value,
            test_fraction=args.test_fraction,
            train_indices=train_indices,
            test_indices=test_indices,
        )
        summary = evaluation["summary"]
        metrics_path = output_dir / "evaluation" / "metrics.json"
        print(
            "Evaluation saved to "
            f"{metrics_path} | real-only acc: {summary['real_only_accuracy']:.4f} | "
            f"real+synthetic acc: {summary['real_plus_synthetic_accuracy']:.4f} | "
            f"improvement: {summary['accuracy_improvement_real_plus_synthetic']:.4f}"
        )


def _denormalize_for_visualization(sample: np.ndarray, min_value: float, max_value: float) -> np.ndarray:
    return ((sample + 1.0) / 2.0) * (max_value - min_value) + min_value


def _optional_int_range(start: float | None, stop: float | None) -> tuple[int | None, int | None]:
    return None if start is None else int(start), None if stop is None else int(stop)


def _class_name_from_report(preprocessing_report: dict, class_id: int) -> str:
    metadata = preprocessing_report.get("mea_metadata")
    if isinstance(metadata, list) and metadata:
        mapping = metadata[0].get("label_mapping", {})
        inverse = {int(value): str(key) for key, value in mapping.items()}
        return inverse.get(class_id, f"class {class_id}")
    return f"class {class_id}"


if __name__ == "__main__":
    main()
