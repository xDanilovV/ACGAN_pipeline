from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class ClassifierConfig:
    num_epochs: int = 25
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    test_fraction: float = 0.2
    seed: int = 42


class SpectraTensorDataset(Dataset):
    """Classifier dataset for real or generated spectra."""

    def __init__(
        self,
        samples: np.ndarray,
        labels: np.ndarray,
        *,
        image_shape: tuple[int, int],
        normalize: bool,
        min_value: float | None = None,
        max_value: float | None = None,
    ) -> None:
        self.samples = np.asarray(samples, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.image_shape = image_shape
        self.normalize = normalize
        self.min_value = float(np.min(self.samples)) if min_value is None else min_value
        self.max_value = float(np.max(self.samples)) if max_value is None else max_value
        if len(self.samples) != len(self.labels):
            raise ValueError("samples and labels must have the same length")

    def __len__(self) -> int:
        return int(len(self.samples))

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        sample = torch.from_numpy(self.samples[index]).float()
        if sample.ndim == 2:
            sample = sample.unsqueeze(0)
        if tuple(sample.shape[-2:]) != self.image_shape:
            sample = F.interpolate(
                sample.unsqueeze(0),
                size=self.image_shape,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        if self.normalize:
            sample = (sample - self.min_value) / (self.max_value - self.min_value)
            sample = sample.clamp(0.0, 1.0).mul(2.0).sub(1.0)
        return sample, torch.tensor(self.labels[index], dtype=torch.long)


class SpectraClassifier(nn.Module):
    """Small CNN used only for downstream AC-GAN utility experiments."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 5), padding=(0, 2)),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=(5, 1), padding=(2, 0)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=(1, 5), padding=(0, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=(5, 1), padding=(2, 0)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        return self.classifier(torch.flatten(x, 1))


def run_core_evaluation_suite(
    *,
    real_samples: np.ndarray,
    real_labels: np.ndarray,
    synthetic_samples: np.ndarray,
    synthetic_labels: np.ndarray,
    image_shape: tuple[int, int],
    num_epochs: int,
    output_dir: str | Path,
    seed: int = 42,
) -> dict[str, dict[str, object]]:
    """Run thesis-facing classifier experiments for AC-GAN augmentation."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = ClassifierConfig(num_epochs=num_epochs, seed=seed)
    num_classes = int(np.unique(real_labels).size)

    train_idx, test_idx = stratified_train_test_split(real_labels, config.test_fraction, seed)
    real_train_samples = real_samples[train_idx]
    real_test_samples = real_samples[test_idx]
    real_min = float(np.min(real_train_samples))
    real_max = float(np.max(real_train_samples))
    real_train_processed = _resize_samples(
        _normalize_to_minus_one_one(real_train_samples, real_min, real_max),
        image_shape,
    )
    real_test_processed = _resize_samples(
        _normalize_to_minus_one_one(real_test_samples, real_min, real_max),
        image_shape,
    )
    synthetic_processed = _resize_samples(synthetic_samples, image_shape)
    real_train = (real_train_processed, real_labels[train_idx])
    real_test = (real_test_processed, real_labels[test_idx])
    synth = (synthetic_processed, synthetic_labels)

    experiments = {
        "real_only_test_real": train_and_evaluate_classifier(
            train_data=real_train,
            test_data=real_test,
            image_shape=image_shape,
            num_classes=num_classes,
            config=config,
            train_is_synthetic=True,
            test_is_synthetic=True,
        ),
        "real_plus_synthetic_test_real": train_and_evaluate_classifier(
            train_data=_concat_arrays(real_train, synth),
            test_data=real_test,
            image_shape=image_shape,
            num_classes=num_classes,
            config=config,
            train_is_synthetic=True,
            test_is_synthetic=True,
            synthetic_count=len(synthetic_labels),
        ),
        "real_only_test_synthetic": train_and_evaluate_classifier(
            train_data=real_train,
            test_data=synth,
            image_shape=image_shape,
            num_classes=num_classes,
            config=config,
            train_is_synthetic=True,
            test_is_synthetic=True,
        ),
        "synthetic_only_test_real": train_and_evaluate_classifier(
            train_data=synth,
            test_data=real_test,
            image_shape=image_shape,
            num_classes=num_classes,
            config=config,
            train_is_synthetic=True,
            test_is_synthetic=True,
        ),
    }

    improvement = (
        experiments["real_plus_synthetic_test_real"]["accuracy"]
        - experiments["real_only_test_real"]["accuracy"]
    )
    experiments["summary"] = {
        "accuracy_improvement_real_plus_synthetic": improvement,
        "config": asdict(config),
        "num_real_train": int(len(real_train[1])),
        "num_real_test": int(len(real_test[1])),
        "num_synthetic": int(len(synthetic_labels)),
    }
    _save_evaluation_outputs(experiments, output_dir)
    return experiments


def train_and_evaluate_classifier(
    *,
    train_data: tuple[np.ndarray, np.ndarray],
    test_data: tuple[np.ndarray, np.ndarray],
    image_shape: tuple[int, int],
    num_classes: int,
    config: ClassifierConfig,
    train_is_synthetic: bool,
    test_is_synthetic: bool,
    synthetic_count: int = 0,
) -> dict[str, object]:
    """Train a downstream classifier and return classification metrics."""

    torch.manual_seed(config.seed)
    train_samples, train_labels = train_data
    test_samples, test_labels = test_data
    real_min = float(np.min(train_samples))
    real_max = float(np.max(train_samples))
    if real_max <= real_min:
        raise ValueError("training samples have no intensity range")

    train_dataset = SpectraTensorDataset(
        train_samples,
        train_labels,
        image_shape=image_shape,
        normalize=not train_is_synthetic,
        min_value=real_min,
        max_value=real_max,
    )
    test_dataset = SpectraTensorDataset(
        test_samples,
        test_labels,
        image_shape=image_shape,
        normalize=not test_is_synthetic,
        min_value=real_min,
        max_value=real_max,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.device == "cuda",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.device == "cuda",
    )

    device = torch.device(config.device)
    model = SpectraClassifier(num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    for _ in range(config.num_epochs):
        model.train()
        for samples, labels in train_loader:
            samples = samples.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(samples), labels)
            loss.backward()
            optimizer.step()

    y_true, y_pred = _predict(model, test_loader, device)
    metrics = classification_report(y_true, y_pred, num_classes)
    metrics["num_train"] = int(len(train_labels))
    metrics["num_test"] = int(len(test_labels))
    metrics["num_synthetic_in_train"] = int(synthetic_count)
    return metrics


def classification_report(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict[str, object]:
    matrix = confusion_matrix(y_true, y_pred, num_classes)
    accuracy = float(np.trace(matrix) / max(np.sum(matrix), 1))
    precision = []
    recall = []
    f1 = []
    support = []
    for class_id in range(num_classes):
        tp = float(matrix[class_id, class_id])
        fp = float(matrix[:, class_id].sum() - tp)
        fn = float(matrix[class_id, :].sum() - tp)
        class_support = float(matrix[class_id, :].sum())
        p = tp / max(tp + fp, 1.0)
        r = tp / max(tp + fn, 1.0)
        score = 2.0 * p * r / max(p + r, 1e-12)
        precision.append(p)
        recall.append(r)
        f1.append(score)
        support.append(class_support)

    support_arr = np.asarray(support, dtype=np.float64)
    weights = support_arr / max(float(support_arr.sum()), 1.0)
    return {
        "accuracy": accuracy,
        "balanced_accuracy": float(np.mean(recall)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "weighted_f1": float(np.sum(np.asarray(f1) * weights)),
        "per_class_precision": precision,
        "per_class_recall": recall,
        "per_class_f1": f1,
        "support": [int(value) for value in support],
        "confusion_matrix": matrix.astype(int).tolist(),
    }


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[int(true_label), int(pred_label)] += 1
    return matrix


def stratified_train_test_split(labels: np.ndarray, test_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64)
    rng = np.random.default_rng(seed)
    train_indices: list[np.ndarray] = []
    test_indices: list[np.ndarray] = []
    for class_id in np.unique(labels):
        class_indices = np.flatnonzero(labels == class_id)
        rng.shuffle(class_indices)
        test_count = max(1, int(round(len(class_indices) * test_fraction)))
        test_indices.append(class_indices[:test_count])
        train_indices.append(class_indices[test_count:])
    return np.concatenate(train_indices), np.concatenate(test_indices)


@torch.no_grad()
def _predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []
    for samples, labels in loader:
        samples = samples.to(device, non_blocking=True)
        logits = model(samples)
        y_true.append(labels.numpy())
        y_pred.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(y_true), np.concatenate(y_pred)


def _concat_arrays(
    left: tuple[np.ndarray, np.ndarray],
    right: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    return np.concatenate([left[0], right[0]], axis=0), np.concatenate([left[1], right[1]], axis=0)


def _normalize_to_minus_one_one(samples: np.ndarray, min_value: float, max_value: float) -> np.ndarray:
    if max_value <= min_value:
        raise ValueError("cannot normalize samples with zero intensity range")
    normalized = (np.asarray(samples, dtype=np.float32) - min_value) / (max_value - min_value)
    return np.clip(normalized, 0.0, 1.0) * 2.0 - 1.0


def _resize_samples(samples: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim == 3:
        tensor = torch.from_numpy(samples[:, None])
    elif samples.ndim == 4 and samples.shape[1] == 1:
        tensor = torch.from_numpy(samples)
    else:
        raise ValueError("expected samples shaped [N, H, W] or [N, 1, H, W]")
    if tuple(tensor.shape[-2:]) != image_shape:
        tensor = F.interpolate(tensor, size=image_shape, mode="bilinear", align_corners=False)
    return tensor[:, 0].numpy()


def _save_evaluation_outputs(results: dict[str, dict[str, object]], output_dir: Path) -> None:
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    for name, metrics in results.items():
        matrix = metrics.get("confusion_matrix")
        if matrix is None:
            continue
        with (output_dir / f"{name}_confusion_matrix.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(matrix)
