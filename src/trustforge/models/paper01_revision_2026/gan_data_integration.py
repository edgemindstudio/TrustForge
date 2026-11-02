"""Governed GAN training-data integration for Paper 1 revision 2026.

Builds train-only class-position indices and validates sampled
canonical batches. No independent splitting is permitted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trustforge.models.paper01_revision_2026.gan_class_sampler import (
    GANTrainingClassSampler,
)


CHUNK_SIZE = 8192
CANONICAL_SHAPE = (40, 40, 3)

FULL_LABEL_CLASSES = {
    "USTCRevision2026",
    "CICMalDroidRevision2026",
}
POSITION_LABEL_CLASSES = {
    "MalNetImageRevision2026",
    "MaleXRevision2026",
}


def training_class_positions(
    dataset: object,
    *,
    class_count: int,
) -> tuple[np.ndarray, ...]:
    """Index all governed train positions without decoding images."""

    if isinstance(class_count, (bool, np.bool_)) or not isinstance(
        class_count, (int, np.integer)
    ) or class_count <= 0:
        raise ValueError("class_count must be a positive integer")

    dataset_type = type(dataset).__name__
    if dataset_type not in (
        FULL_LABEL_CLASSES | POSITION_LABEL_CLASSES
    ):
        raise TypeError(
            f"unsupported governed dataset: {dataset_type}"
        )

    n = int(dataset.split_size("train"))
    if n <= 0:
        raise ValueError("training split must not be empty")

    ids = np.empty(n, dtype=np.int32)

    if dataset_type in FULL_LABEL_CLASSES:
        raw = np.asarray(dataset.labels("train"))
        if raw.shape != (n,) or raw.dtype.kind not in "iu":
            raise ValueError("invalid complete training labels")
        if np.any(raw < 0) or np.any(raw >= class_count):
            raise ValueError("training label outside class mapping")
        ids[:] = raw
    else:
        for start in range(0, n, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, n)
            positions = np.arange(start, end, dtype=np.int64)
            raw = np.asarray(dataset.labels("train", positions))
            if raw.shape != (end - start,) or raw.dtype.kind not in "iu":
                raise ValueError("invalid chunked training labels")
            if np.any(raw < 0) or np.any(raw >= class_count):
                raise ValueError("training label outside class mapping")
            ids[start:end] = raw

    observed = np.bincount(ids, minlength=class_count)
    if observed.shape != (class_count,) or np.any(observed == 0):
        raise ValueError("every governed class must have training data")

    positions = tuple(
        np.flatnonzero(ids == class_id).astype(np.int64)
        for class_id in range(class_count)
    )

    if sum(len(p) for p in positions) != n:
        raise RuntimeError("training index coverage mismatch")

    return positions


@dataclass
class GANTrainingData:
    """Train-only batches with authoritative conditioning validation."""

    dataset: object
    class_count: int
    seed: int

    def __post_init__(self) -> None:
        positions = training_class_positions(
            self.dataset,
            class_count=self.class_count,
        )
        self.sampler = GANTrainingClassSampler(
            positions,
            seed=self.seed,
        )

    def draw(self, batch_size: int) -> tuple[
        np.ndarray, np.ndarray, np.ndarray
    ]:
        """Return canonical images, one-hot labels, logical positions."""

        positions, class_ids = self.sampler.draw(batch_size)

        images, labels = self.dataset.canonical_batch(
            "train",
            positions,
        )
        images = np.asarray(images)
        labels = np.asarray(labels)

        b = len(positions)
        if images.shape != (b, *CANONICAL_SHAPE):
            raise ValueError("invalid canonical image shape")
        if images.dtype != np.float32:
            raise ValueError("canonical images must be float32")
        if not np.all(np.isfinite(images)):
            raise ValueError("nonfinite canonical images")
        if np.any(images < 0.0) or np.any(images > 1.0):
            raise ValueError("canonical images outside [0,1]")

        if labels.shape != (b, self.class_count):
            raise ValueError("invalid one-hot label shape")
        if labels.dtype != np.float32:
            raise ValueError("one-hot labels must be float32")
        if not np.all(np.isfinite(labels)):
            raise ValueError("nonfinite one-hot labels")
        if not np.all((labels == 0.0) | (labels == 1.0)):
            raise ValueError("labels must contain only zero or one")
        if not np.all(labels.sum(axis=1) == 1.0):
            raise ValueError("labels must be exactly one-hot")

        if not np.array_equal(
            np.argmax(labels, axis=1).astype(np.int32),
            class_ids,
        ):
            raise ValueError(
                "sampled classes disagree with dataset labels"
            )

        return images, labels, positions
