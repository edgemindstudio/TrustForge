"""
TrustForge dataset partitioning contract.

This module implements the revision-2026 partition policy without changing the
historical behavior of common.data.load_dataset_npy().

Policy cases
------------
Case 1:
    Provider supplies train / validation / test.
    Preserve all three partitions exactly.

Case 2:
    Provider supplies train / test.
    Preserve test and derive validation from 10% of provider training data.

Case 3:
    Provider supplies an unsplit labeled corpus.
    Derive train / validation / test as 70% / 10% / 20%.

All benchmark-derived partitions are deterministic, stratified by label, and
use partition seed 42 unless explicitly overridden.

The partition seed is infrastructure/reproducibility state and is distinct from
experimental/model seeds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


DEFAULT_PARTITION_SEED = 42
DEFAULT_CASE2_VAL_FRACTION = 0.10
DEFAULT_CASE3_RATIOS = (0.70, 0.10, 0.20)


@dataclass(frozen=True)
class DatasetPartitions:
    """Common TrustForge logical train / validation / test contract."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def _label_ids(y: np.ndarray) -> np.ndarray:
    """
    Convert supported label representations to one-dimensional class IDs.

    Supported forms:
    - integer/vector labels: (N,)
    - column labels:         (N, 1)
    - one-hot labels:        (N, K)
    """
    y = np.asarray(y)

    if y.ndim == 1:
        return y

    if y.ndim == 2 and y.shape[1] == 1:
        return y[:, 0]

    if y.ndim == 2 and y.shape[1] > 1:
        return np.argmax(y, axis=1)

    raise ValueError(
        f"Unsupported label shape {y.shape}; expected (N,), (N,1), or (N,K)."
    )


def _validate_pair(x: np.ndarray, y: np.ndarray, name: str) -> None:
    """Require aligned feature and label sample counts."""
    if len(x) != len(y):
        raise ValueError(
            f"{name} feature/label length mismatch: {len(x)} != {len(y)}"
        )


def _stratified_indices(
    y: np.ndarray,
    fractions: Tuple[float, ...],
    *,
    seed: int,
) -> Tuple[np.ndarray, ...]:
    """
    Partition sample indices deterministically within each class.

    Fractions must sum to 1.0. Per-class allocation uses floor for all but the
    final partition; the final partition receives the remainder so that every
    sample is assigned exactly once.
    """
    fractions = tuple(float(f) for f in fractions)

    if not fractions:
        raise ValueError("At least one partition fraction is required.")

    if any(f <= 0.0 for f in fractions):
        raise ValueError(f"All fractions must be > 0, got {fractions}")

    if not np.isclose(sum(fractions), 1.0):
        raise ValueError(f"Fractions must sum to 1.0, got {fractions}")

    labels = _label_ids(y)
    rng = np.random.default_rng(seed)

    buckets = [[] for _ in fractions]

    for label in np.unique(labels):
        class_indices = np.flatnonzero(labels == label)
        class_indices = rng.permutation(class_indices)

        n = len(class_indices)
        start = 0

        for i, fraction in enumerate(fractions):
            if i == len(fractions) - 1:
                stop = n
            else:
                stop = start + int(np.floor(n * fraction))

            buckets[i].append(class_indices[start:stop])
            start = stop

    result = []

    for pieces in buckets:
        if pieces:
            indices = np.concatenate(pieces)
            indices = np.sort(indices)
        else:
            indices = np.empty(0, dtype=np.int64)

        result.append(indices)

    return tuple(result)


def preserve_provider_partitions(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> DatasetPartitions:
    """
    Case 1: preserve provider-defined train / validation / test partitions.
    """
    _validate_pair(x_train, y_train, "train")
    _validate_pair(x_val, y_val, "validation")
    _validate_pair(x_test, y_test, "test")

    return DatasetPartitions(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
    )


def derive_validation_from_train(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    val_fraction: float = DEFAULT_CASE2_VAL_FRACTION,
    seed: int = DEFAULT_PARTITION_SEED,
) -> DatasetPartitions:
    """
    Case 2: preserve provider test and derive validation from provider train.
    """
    _validate_pair(x_train, y_train, "train")
    _validate_pair(x_test, y_test, "test")

    val_fraction = float(val_fraction)

    if not (0.0 < val_fraction < 1.0):
        raise ValueError(
            f"val_fraction must be in (0,1), got {val_fraction}"
        )

    train_fraction = 1.0 - val_fraction

    train_idx, val_idx = _stratified_indices(
        y_train,
        (train_fraction, val_fraction),
        seed=seed,
    )

    return DatasetPartitions(
        x_train=x_train[train_idx],
        y_train=y_train[train_idx],
        x_val=x_train[val_idx],
        y_val=y_train[val_idx],
        x_test=x_test,
        y_test=y_test,
    )


def split_unsplit_corpus(
    x: np.ndarray,
    y: np.ndarray,
    *,
    train_fraction: float = DEFAULT_CASE3_RATIOS[0],
    val_fraction: float = DEFAULT_CASE3_RATIOS[1],
    test_fraction: float = DEFAULT_CASE3_RATIOS[2],
    seed: int = DEFAULT_PARTITION_SEED,
) -> DatasetPartitions:
    """
    Case 3: derive train / validation / test from an unsplit labeled corpus.
    """
    _validate_pair(x, y, "corpus")

    train_idx, val_idx, test_idx = _stratified_indices(
        y,
        (train_fraction, val_fraction, test_fraction),
        seed=seed,
    )

    return DatasetPartitions(
        x_train=x[train_idx],
        y_train=y[train_idx],
        x_val=x[val_idx],
        y_val=y[val_idx],
        x_test=x[test_idx],
        y_test=y[test_idx],
    )
