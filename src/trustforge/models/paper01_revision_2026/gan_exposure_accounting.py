"""Train-position exposure accounting for revision-2026 GAN.

Records actual sampling with replacement. No training execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class GANExposureAccounting:
    """Account for every sampled real training position and class."""

    train_labels: np.ndarray
    class_count: int
    _draw_counts: np.ndarray = field(init=False, repr=False)
    _class_draws: np.ndarray = field(init=False, repr=False)
    batches: int = field(default=0, init=False)

    def __post_init__(self):
        labels = np.asarray(self.train_labels)

        if (
            labels.ndim != 1
            or labels.size == 0
            or labels.dtype.kind not in "iu"
            or isinstance(self.class_count, bool)
            or not isinstance(self.class_count, (int, np.integer))
            or self.class_count <= 0
        ):
            raise ValueError("invalid training-label accounting contract")

        if np.any(labels < 0) or np.any(labels >= self.class_count):
            raise ValueError("training labels outside governed class range")

        counts = np.bincount(labels, minlength=self.class_count)
        if np.any(counts == 0):
            raise ValueError("every governed class must contain training data")

        self.train_labels = labels.astype(np.int32, copy=True)
        self.train_labels.flags.writeable = False
        self._draw_counts = np.zeros(labels.size, dtype=np.int64)
        self._class_draws = np.zeros(self.class_count, dtype=np.int64)

    def record_batch(self, *, positions, one_hot_labels):
        positions = np.asarray(positions)
        labels = np.asarray(one_hot_labels)

        if (
            positions.ndim != 1
            or positions.size == 0
            or positions.dtype.kind not in "iu"
        ):
            raise ValueError("invalid training positions")

        if np.any(positions < 0) or np.any(
            positions >= self.train_labels.size
        ):
            raise ValueError("training position outside governed split")

        if (
            labels.shape != (positions.size, self.class_count)
            or labels.dtype != np.float32
            or not np.all(np.isfinite(labels))
            or not np.all((labels == 0.0) | (labels == 1.0))
            or not np.all(labels.sum(axis=1) == 1.0)
        ):
            raise ValueError("invalid real-batch one-hot labels")

        ids = np.argmax(labels, axis=1)
        if not np.array_equal(ids, self.train_labels[positions]):
            raise ValueError("sampled labels disagree with train positions")

        np.add.at(self._draw_counts, positions, 1)
        self._class_draws += np.bincount(
            ids, minlength=self.class_count
        )
        self.batches += 1

    def report(self, *, expected_batches, expected_draws):
        if self.batches != expected_batches:
            raise ValueError("incomplete batch accounting")

        if int(self._class_draws.sum()) != expected_draws:
            raise ValueError("training exposure total mismatch")

        if int(self._draw_counts.sum()) != expected_draws:
            raise ValueError("position-level exposure total mismatch")

        result = []
        for class_id in range(self.class_count):
            mask = self.train_labels == class_id
            class_counts = self._draw_counts[mask]
            draws = int(self._class_draws[class_id])
            unique = int(np.count_nonzero(class_counts))
            size = int(np.count_nonzero(mask))

            result.append({
                "class_id": class_id,
                "train_size": size,
                "draws": draws,
                "unique_positions": unique,
                "repeated_draws": draws - unique,
                "nominal_passes": draws / size,
                "unique_coverage": unique / size,
            })

        return {
            "batches": int(self.batches),
            "total_real_draws": int(expected_draws),
            "total_unique_positions": int(
                np.count_nonzero(self._draw_counts)
            ),
            "per_class": result,
        }


def training_labels_from_class_positions(
    class_positions, *, train_size, class_count
):
    """Recover exact train-position labels from governed class indices."""
    if (
        isinstance(train_size, bool)
        or not isinstance(train_size, (int, np.integer))
        or train_size <= 0
        or isinstance(class_count, bool)
        or not isinstance(class_count, (int, np.integer))
        or class_count <= 0
    ):
        raise ValueError("invalid governed training dimensions")

    if len(class_positions) != class_count:
        raise ValueError("class-position count mismatch")

    labels = np.full(int(train_size), -1, dtype=np.int32)

    for class_id, raw in enumerate(class_positions):
        positions = np.asarray(raw)

        if (
            positions.ndim != 1
            or positions.size == 0
            or positions.dtype.kind not in "iu"
        ):
            raise ValueError("invalid class training positions")

        if np.any(positions < 0) or np.any(positions >= train_size):
            raise ValueError("training position outside governed split")

        # NumPy advanced indexing does not detect repeated indices
        # within the same assignment. Reject them explicitly.
        if np.unique(positions).size != positions.size:
            raise ValueError("duplicate positions within a class")

        if np.any(labels[positions] != -1):
            raise ValueError("overlapping class positions")

        labels[positions] = class_id

    if np.any(labels == -1):
        raise ValueError("training positions do not cover split exactly")

    labels.flags.writeable = False
    return labels
