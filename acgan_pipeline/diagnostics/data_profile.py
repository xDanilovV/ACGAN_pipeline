from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from acgan_pipeline.diagnostics.common import (
    add_common_args,
    load_experiment_inputs,
    normalized_dataset_arrays,
    save_json,
    set_seed,
)
from acgan_pipeline.models.generator import Generator
from acgan_pipeline.training.train_acgan import _weights_init, generate_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile real GC-IMS tensors against an untrained AC-GAN generator.")
    add_common_args(parser)
    parser.add_argument("--generated-per-class", type=int, default=10)
    parser.add_argument("--max-class-examples", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples, labels, target_shape, preprocessing_report, config = load_experiment_inputs(args.config, args.data)
    set_seed(config.seed)
    real_tensors, encoded_labels, min_value, max_value = normalized_dataset_arrays(
        samples,
        labels,
        target_shape=target_shape,
        resize_mode=config.resize_mode,
        batch_size=config.batch_size,
    )

    num_classes = int(np.unique(encoded_labels).size)
    generator = Generator(
        config.noise_dim,
        num_classes,
        target_shape,
        base_channels=config.generator_base_channels,
    )
    generator.apply(_weights_init)
    init_samples, init_labels = generate_samples(
        generator,
        args.generated_per_class,
        num_classes=num_classes,
        noise_dim=config.noise_dim,
        device=torch.device("cpu"),
        batch_size=128,
    )

    stats = {
        "target_shape": list(target_shape),
        "normalization": {"min_value": min_value, "max_value": max_value},
        "real": summarize_array(real_tensors),
        "untrained_generator": summarize_array(init_samples),
        "real_by_class": summarize_by_class(real_tensors, encoded_labels),
        "untrained_generator_by_class": summarize_by_class(init_samples, init_labels),
    }
    save_json(stats, output_dir / "intensity_stats.json")
    save_json(preprocessing_report["label_summary"], output_dir / "class_counts.json")
    save_json(preprocessing_report, output_dir / "preprocessing_report.json")

    save_histogram_comparison(real_tensors, init_samples, output_dir / "real_vs_untrained_generator_histograms.png")
    save_class_average_grid(real_tensors, encoded_labels, preprocessing_report, output_dir / "real_class_averages.png")
    save_example_grid(real_tensors, encoded_labels, preprocessing_report, output_dir / "real_examples_by_class.png", args.max_class_examples)
    save_example_grid(init_samples, init_labels, preprocessing_report, output_dir / "untrained_generator_examples_by_class.png", args.max_class_examples)
    print(f"Saved data profile to {output_dir}")


def summarize_array(values: np.ndarray) -> dict[str, object]:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    percentiles = [0, 1, 5, 25, 50, 75, 95, 99, 100]
    return {
        "shape": list(np.asarray(values).shape),
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "percentiles": {str(p): float(np.percentile(flat, p)) for p in percentiles},
        "fraction_le_minus_0_9": float(np.mean(flat <= -0.9)),
        "fraction_le_minus_0_5": float(np.mean(flat <= -0.5)),
        "fraction_ge_0_5": float(np.mean(flat >= 0.5)),
        "fraction_ge_0_9": float(np.mean(flat >= 0.9)),
    }


def summarize_by_class(samples: np.ndarray, labels: np.ndarray) -> dict[str, dict[str, object]]:
    return {
        str(int(class_id)): summarize_array(samples[np.asarray(labels) == class_id])
        for class_id in np.unique(labels)
    }


def save_histogram_comparison(real: np.ndarray, generated: np.ndarray, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    bins = np.linspace(-1.0, 1.0, 101)
    ax.hist(real.reshape(-1), bins=bins, density=True, alpha=0.55, label="Real preprocessed", color="#1f77b4")
    ax.hist(generated.reshape(-1), bins=bins, density=True, alpha=0.55, label="Untrained generator", color="#ff7f0e")
    ax.set_xlabel("Normalized intensity")
    ax.set_ylabel("Density")
    ax.set_title("Pixel intensity distribution")
    ax.legend()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_class_average_grid(samples: np.ndarray, labels: np.ndarray, report: dict, path: Path) -> None:
    averages = []
    titles = []
    inverse = _inverse_mapping(report)
    for class_id in sorted(np.unique(labels)):
        averages.append(np.mean(samples[labels == class_id], axis=0))
        titles.append(f"{int(class_id)}: {inverse.get(int(class_id), str(int(class_id)))}")
    _save_grid(averages, titles, path, columns=2, cmap="viridis")


def save_example_grid(samples: np.ndarray, labels: np.ndarray, report: dict, path: Path, max_examples: int) -> None:
    images = []
    titles = []
    inverse = _inverse_mapping(report)
    for class_id in sorted(np.unique(labels)):
        indices = np.flatnonzero(labels == class_id)[:max_examples]
        for i, index in enumerate(indices, start=1):
            images.append(samples[index])
            titles.append(f"{int(class_id)} {inverse.get(int(class_id), '')} #{i}")
    _save_grid(images, titles, path, columns=max_examples, cmap="viridis")


def _save_grid(images: list[np.ndarray], titles: list[str], path: Path, *, columns: int, cmap: str) -> None:
    import matplotlib.pyplot as plt

    if not images:
        return
    rows = int(np.ceil(len(images) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 3.5 * rows), constrained_layout=True)
    axes_arr = np.atleast_1d(axes).reshape(rows, columns)
    for ax in axes_arr.flat:
        ax.axis("off")
    for ax, image, title in zip(axes_arr.flat, images, titles):
        im = ax.imshow(image, cmap=cmap, aspect="auto", vmin=-1, vmax=1)
        ax.set_title(title, fontsize=8)
        ax.axis("on")
    fig.colorbar(im, ax=axes_arr.ravel().tolist(), fraction=0.02, pad=0.01)
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def _inverse_mapping(report: dict) -> dict[int, str]:
    mapping = report.get("label_summary", {}).get("label_mapping", {})
    return {int(value): str(key) for key, value in mapping.items()}


if __name__ == "__main__":
    main()
