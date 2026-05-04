from __future__ import annotations

import numpy as np


def augment_dataset(
    real_data: tuple[np.ndarray, np.ndarray],
    synthetic_data: tuple[np.ndarray, np.ndarray],
    strategy: str,
    *,
    ratio: float = 1.0,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Combine real and synthetic data using a class-aware strategy.

    Strategies:
    - ``equal``: add the same number of synthetic samples per class.
    - ``minority``: augment only the smallest real-data class.
    - ``ratio``: add ``ratio * real_count[class]`` synthetic samples per class.
    """

    real_samples, real_labels = real_data
    synthetic_samples, synthetic_labels = synthetic_data
    real_labels = np.asarray(real_labels, dtype=np.int64)
    synthetic_labels = np.asarray(synthetic_labels, dtype=np.int64)
    rng = np.random.default_rng(random_state)

    real_counts = _class_counts(real_labels)
    selected_indices: list[np.ndarray] = []

    if strategy == "equal":
        per_class = min(_class_counts(synthetic_labels).values())
        for class_id in real_counts:
            selected_indices.append(_sample_indices(synthetic_labels, class_id, per_class, rng))
    elif strategy == "minority":
        minority_class = min(real_counts, key=real_counts.get)
        target_count = max(real_counts.values())
        needed = max(0, target_count - real_counts[minority_class])
        selected_indices.append(_sample_indices(synthetic_labels, minority_class, needed, rng))
    elif strategy == "ratio":
        if ratio < 0:
            raise ValueError("ratio must be non-negative")
        for class_id, count in real_counts.items():
            needed = int(round(count * ratio))
            selected_indices.append(_sample_indices(synthetic_labels, class_id, needed, rng))
    else:
        raise ValueError("strategy must be one of: equal, minority, ratio")

    if selected_indices:
        synthetic_idx = np.concatenate([idx for idx in selected_indices if idx.size > 0])
    else:
        synthetic_idx = np.array([], dtype=np.int64)

    if synthetic_idx.size == 0:
        return np.asarray(real_samples), real_labels

    augmented_samples = np.concatenate([real_samples, synthetic_samples[synthetic_idx]], axis=0)
    augmented_labels = np.concatenate([real_labels, synthetic_labels[synthetic_idx]], axis=0)
    return augmented_samples, augmented_labels


def train_classifier_real_only(*args, **kwargs):
    """Placeholder for downstream baseline classifier training."""

    raise NotImplementedError("Implement a project-specific classifier for real-only training.")


def train_classifier_real_plus_synthetic(*args, **kwargs):
    """Placeholder for downstream augmented classifier training."""

    raise NotImplementedError("Implement a project-specific classifier for real + synthetic training.")


def _class_counts(labels: np.ndarray) -> dict[int, int]:
    classes, counts = np.unique(labels, return_counts=True)
    return {int(class_id): int(count) for class_id, count in zip(classes, counts)}


def _sample_indices(labels: np.ndarray, class_id: int, count: int, rng: np.random.Generator) -> np.ndarray:
    if count <= 0:
        return np.array([], dtype=np.int64)
    candidates = np.flatnonzero(labels == class_id)
    if candidates.size == 0:
        raise ValueError(f"no synthetic samples available for class {class_id}")
    replace = candidates.size < count
    return rng.choice(candidates, size=count, replace=replace)
