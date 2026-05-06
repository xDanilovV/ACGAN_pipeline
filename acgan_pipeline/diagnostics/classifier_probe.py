from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from acgan_pipeline.data.dataset import GCIMSDataset
from acgan_pipeline.diagnostics.common import add_common_args, load_experiment_inputs, save_json, set_seed
from acgan_pipeline.evaluation import classification_report, stratified_train_test_split
from acgan_pipeline.models.discriminator import Discriminator
from acgan_pipeline.training.train_acgan import _weights_init


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the AC-GAN discriminator classifier on real spectra only.")
    add_common_args(parser)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples, labels, target_shape, preprocessing_report, config = load_experiment_inputs(args.config, args.data)
    set_seed(config.seed)
    dataset = GCIMSDataset(samples, labels, target_shape=target_shape, resize_mode=config.resize_mode)
    train_indices, test_indices = stratified_train_test_split(labels, config.test_fraction, config.seed)

    train_loader = DataLoader(
        Subset(dataset, train_indices.tolist()),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=args.device == "cuda",
    )
    test_loader = DataLoader(
        Subset(dataset, test_indices.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=args.device == "cuda",
    )

    device = torch.device(args.device)
    model = Discriminator(
        dataset.num_classes,
        target_shape,
        base_channels=config.discriminator_base_channels,
        projection_scale=0.0,
        use_norm=config.discriminator_use_norm,
    ).to(device)
    model.apply(_weights_init)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_count = 0
        for batch, batch_labels in train_loader:
            batch = batch.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            _, class_logits = model(batch)
            loss = F.cross_entropy(class_logits, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(batch_labels)
            total_correct += int((class_logits.argmax(dim=1) == batch_labels).sum().detach().cpu())
            total_count += len(batch_labels)
        test_metrics = evaluate(model, test_loader, device, dataset.num_classes)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(total_count, 1),
            "train_accuracy": total_correct / max(total_count, 1),
            "test_accuracy": test_metrics["accuracy"],
            "test_macro_f1": test_metrics["macro_f1"],
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train acc: {row['train_accuracy']:.4f} | "
            f"test acc: {row['test_accuracy']:.4f} | "
            f"macro F1: {row['test_macro_f1']:.4f}"
        )

    final_metrics = evaluate(model, test_loader, device, dataset.num_classes)
    final_metrics["num_train"] = int(len(train_indices))
    final_metrics["num_test"] = int(len(test_indices))
    final_metrics["target_shape"] = list(target_shape)
    final_metrics["history"] = history
    save_json(final_metrics, output_dir / "metrics.json")
    save_json(preprocessing_report, output_dir / "preprocessing_report.json")
    save_confusion_outputs(np.asarray(final_metrics["confusion_matrix"], dtype=np.int64), output_dir)
    print(f"Saved classifier probe to {output_dir}")


@torch.no_grad()
def evaluate(model: Discriminator, loader: DataLoader, device: torch.device, num_classes: int) -> dict[str, object]:
    model.eval()
    y_true = []
    y_pred = []
    for batch, batch_labels in loader:
        batch = batch.to(device, non_blocking=True)
        _, class_logits = model(batch)
        y_true.append(batch_labels.numpy())
        y_pred.append(class_logits.argmax(dim=1).cpu().numpy())
    return classification_report(np.concatenate(y_true), np.concatenate(y_pred), num_classes)


def save_confusion_outputs(matrix: np.ndarray, output_dir: Path) -> None:
    with (output_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(matrix.astype(int).tolist())
    save_confusion_matrix_png(matrix, output_dir / "confusion_matrix.png")


def save_confusion_matrix_png(matrix: np.ndarray, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title("AC-GAN discriminator classifier probe")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_yticks(np.arange(matrix.shape[0]))
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = int(matrix[row, col])
            if value:
                ax.text(col, row, str(value), ha="center", va="center", color="black", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
