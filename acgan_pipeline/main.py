from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from acgan_pipeline.config import load_config
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
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=None)
    config_args, remaining = config_parser.parse_known_args()
    defaults = load_config(config_args.config)

    parser = argparse.ArgumentParser(
        description="Reusable AC-GAN pipeline for 2D GC-IMS-like data.",
        parents=[config_parser],
    )
    parser.set_defaults(**defaults)
    parser.add_argument("--data", type=str, required=True, help="Path to dataset file consumed by the selected loader.")
    parser.add_argument("--labels-csv", type=str, help="Optional CSV labels file for .mea folders. If omitted, labels are inferred from class folders.")
    parser.add_argument("--mea-label-mode", choices=["class", "culture_type"])
    parser.add_argument("--input-format", choices=["npz", "mea"])
    parser.add_argument("--shape-mode", choices=["auto", "fixed"])
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--auto-max-pixels", type=int)
    parser.add_argument("--auto-max-height", type=int)
    parser.add_argument("--auto-max-width", type=int)
    parser.add_argument("--auto-multiple", type=int)
    parser.add_argument("--resize-mode", choices=["area", "bilinear", "bicubic", "nearest"])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--noise-dim", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--lr-g", type=float)
    parser.add_argument("--lr-d", type=float)
    parser.add_argument("--class-loss-weight", type=float)
    parser.add_argument("--tv-loss-weight", type=float)
    parser.add_argument("--label-smoothing", type=float)
    parser.add_argument("--instance-noise-std", type=float)
    parser.add_argument("--instance-noise-decay-epochs", type=int)
    parser.add_argument("--projection-scale", type=float)
    parser.add_argument("--generator-base-channels", type=int)
    parser.add_argument("--discriminator-base-channels", type=int)
    parser.add_argument("--discriminator-use-norm", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--no-discriminator-use-norm", action="store_false", default=argparse.SUPPRESS, dest="discriminator_use_norm")
    parser.add_argument("--generator-steps", type=int)
    parser.add_argument("--discriminator-update-every", type=int)
    parser.add_argument("--sample-every", type=int)
    parser.add_argument("--checkpoint-every", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--samples-per-class", type=int)
    parser.add_argument("--rip-drift-start", type=float, help="First RIP-relative drift-time value to keep.")
    parser.add_argument("--rip-drift-stop", type=float, help="Last RIP-relative drift-time value to keep.")
    parser.add_argument("--crop-rt-start", type=float)
    parser.add_argument("--crop-rt-stop", type=float)
    parser.add_argument("--peak-crop", action="store_true", default=argparse.SUPPRESS, help="Compute a shared peak-aware crop before resizing.")
    parser.add_argument("--no-peak-crop", action="store_false", default=argparse.SUPPRESS, dest="peak_crop")
    parser.add_argument("--peak-percentile", type=float)
    parser.add_argument("--peak-relative-threshold", type=float)
    parser.add_argument("--peak-support-fraction", type=float)
    parser.add_argument("--peak-margin-rt", type=int)
    parser.add_argument("--peak-margin-dt", type=int)
    parser.add_argument("--eval-epochs", type=int)
    parser.add_argument("--test-fraction", type=float)
    parser.add_argument("--classifier", choices=["svm", "cnn"], help="Downstream evaluation classifier.")
    parser.add_argument("--generator-checkpoint", type=str, help="Load a saved generator checkpoint and skip AC-GAN training.")
    parser.add_argument("--skip-evaluation", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--run-evaluation", action="store_false", default=argparse.SUPPRESS, dest="skip_evaluation")
    parser.add_argument("--skip-visualization", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--run-visualization", action="store_false", default=argparse.SUPPRESS, dest="skip_visualization")
    parser.add_argument("--synthetic-viz-denormalized", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--synthetic-viz-normalized", action="store_false", default=argparse.SUPPRESS, dest="synthetic_viz_denormalized")
    parser.add_argument("--viz-class-id", type=int, help="Class id for the real-vs-generated example PNG.")
    return parser.parse_args(remaining)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    initial_target_shape = _fixed_target_shape(args)

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
            target_shape=initial_target_shape,
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
    target_shape = _resolve_target_shape(processed_samples, args, initial_target_shape)
    preprocessing_report["target_shape"] = list(target_shape)
    preprocessing_report["shape_mode"] = args.shape_mode
    preprocessing_report["label_summary"] = _label_summary(labels, preprocessing_report)
    with (output_dir / "preprocessing_report.json").open("w", encoding="utf-8") as f:
        json.dump(preprocessing_report, f, indent=2)
    with (output_dir / "effective_config.json").open("w", encoding="utf-8") as f:
        json.dump({**vars(args), "target_shape": list(target_shape)}, f, indent=2)
    print(f"Using tensor shape {target_shape[0]}x{target_shape[1]} from {args.shape_mode} shape mode")

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
        target_shape=target_shape,
        resize_mode=args.resize_mode,
    )
    config = TrainConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        noise_dim=args.noise_dim,
        lr=args.lr,
        lr_g=args.lr_g,
        lr_d=args.lr_d,
        class_loss_weight=args.class_loss_weight,
        tv_loss_weight=args.tv_loss_weight,
        label_smoothing=args.label_smoothing,
        instance_noise_std=args.instance_noise_std,
        instance_noise_decay_epochs=args.instance_noise_decay_epochs,
        projection_scale=args.projection_scale,
        generator_base_channels=args.generator_base_channels,
        discriminator_base_channels=args.discriminator_base_channels,
        discriminator_use_norm=args.discriminator_use_norm,
        generator_steps=args.generator_steps,
        discriminator_update_every=args.discriminator_update_every,
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
            image_shape=target_shape,
        )
        print(f"Loaded generator from {args.generator_checkpoint}")
    else:
        generator, _, _ = train_acgan(
            dataset,
            num_classes=dataset.num_classes,
            image_shape=target_shape,
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
            image_shape=target_shape,
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


def _fixed_target_shape(args: argparse.Namespace) -> tuple[int, int] | None:
    if args.shape_mode == "fixed":
        if args.height is None or args.width is None:
            raise ValueError("fixed shape mode requires both --height and --width")
        return _validate_model_shape((args.height, args.width))
    if args.height is not None or args.width is not None:
        if args.height is None or args.width is None:
            raise ValueError("manual shape override requires both --height and --width")
        return _validate_model_shape((args.height, args.width))
    return None


def _resolve_target_shape(
    samples: np.ndarray,
    args: argparse.Namespace,
    fixed_shape: tuple[int, int] | None,
) -> tuple[int, int]:
    if fixed_shape is not None:
        return fixed_shape
    if samples.ndim not in (3, 4):
        raise ValueError("expected samples shaped [N, H, W] or [N, 1, H, W]")
    height, width = samples.shape[-2:]
    return _auto_target_shape(
        int(height),
        int(width),
        max_pixels=args.auto_max_pixels,
        max_height=args.auto_max_height,
        max_width=args.auto_max_width,
        multiple=args.auto_multiple,
    )


def _auto_target_shape(
    height: int,
    width: int,
    *,
    max_pixels: int,
    max_height: int,
    max_width: int,
    multiple: int,
) -> tuple[int, int]:
    if height <= 0 or width <= 0:
        raise ValueError("cannot infer tensor shape from an empty spectrum")
    if multiple <= 0:
        raise ValueError("auto_multiple must be positive")

    scale = min(
        1.0,
        (max_pixels / float(height * width)) ** 0.5,
        max_height / float(height),
        max_width / float(width),
    )
    target_height = _round_to_multiple(height * scale, multiple)
    target_width = _round_to_multiple(width * scale, multiple)
    return _validate_model_shape((target_height, target_width))


def _round_to_multiple(value: float, multiple: int) -> int:
    return max(multiple, int(round(value / multiple)) * multiple)


def _validate_model_shape(shape: tuple[int, int]) -> tuple[int, int]:
    height, width = shape
    if height % 16 != 0 or width % 16 != 0:
        raise ValueError(f"tensor shape must be divisible by 16 for the current model, got {height}x{width}")
    return int(height), int(width)


def _label_summary(labels: np.ndarray, preprocessing_report: dict) -> dict[str, object]:
    labels = np.asarray(labels, dtype=np.int64)
    counts = {int(label): int(np.sum(labels == label)) for label in np.unique(labels)}
    metadata = preprocessing_report.get("mea_metadata")
    mapping: dict[str, int] = {}
    examples: dict[int, str] = {}
    if isinstance(metadata, list) and metadata:
        raw_mapping = metadata[0].get("label_mapping", {})
        mapping = {str(key): int(value) for key, value in raw_mapping.items()}
        for item in metadata:
            label_name = item.get("label")
            if label_name in mapping:
                label_id = mapping[str(label_name)]
                examples.setdefault(label_id, str(item.get("path", "")))
    inverse = {value: key for key, value in mapping.items()}
    return {
        "num_samples": int(len(labels)),
        "num_classes": int(len(counts)),
        "counts_by_id": counts,
        "label_mapping": mapping,
        "counts_by_name": {inverse.get(label_id, str(label_id)): count for label_id, count in counts.items()},
        "example_path_by_id": examples,
    }


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
