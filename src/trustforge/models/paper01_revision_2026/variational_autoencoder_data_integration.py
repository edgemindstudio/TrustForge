"""Canonical dataset integration for Paper 1 revised VAE.

This module consumes governed logical positions and returns
validated canonical image/label batches.

No dataset is instantiated here. No split is derived here.
No real-dataset training or production synthesis is authorized.
"""

from __future__ import annotations

import numpy as np


CANONICAL_IMAGE_SHAPE = (40, 40, 3)


def _positive_integer(value, *, name):
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")

    if not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")

    result = int(value)

    if result <= 0:
        raise ValueError(f"{name} must be positive")

    return result


def _require_adapter(adapter):
    for method in ("split_size", "canonical_batch"):
        if not callable(getattr(adapter, method, None)):
            raise TypeError(
                f"adapter must provide callable {method}"
            )


def _validate_positions(positions, *, split_size):
    if not isinstance(positions, np.ndarray):
        raise TypeError("logical positions must be numpy.ndarray")

    if positions.ndim != 1:
        raise ValueError("logical positions must be rank one")

    if positions.dtype != np.int64:
        raise TypeError("logical positions must be int64")

    if positions.size == 0:
        raise ValueError("logical position batch must be nonempty")

    if np.any(positions < 0):
        raise ValueError("negative logical position")

    if np.any(positions >= split_size):
        raise ValueError("logical position exceeds partition bounds")

    if np.unique(positions).size != positions.size:
        raise ValueError("duplicate logical position in batch")

    return positions


def canonical_vae_batch(
    *,
    adapter,
    split,
    logical_positions,
    num_classes,
):
    """Return validated (images, one_hot_labels) as float32 arrays.

    Only train and validation may be requested. The caller
    governs the traversal schedule and source partition.

    This function does not assert coverage across batches.
    """
    if not isinstance(split, str):
        raise TypeError("split must be a string")

    if split not in ("train", "validation"):
        raise ValueError(
            "VAE batch integration prohibits this split"
        )

    num_classes = _positive_integer(
        num_classes,
        name="num_classes",
    )

    if num_classes < 2:
        raise ValueError("VAE requires at least two classes")

    _require_adapter(adapter)

    size = _positive_integer(
        adapter.split_size(split),
        name="split size",
    )

    positions = _validate_positions(
        logical_positions,
        split_size=size,
    )

    # Protect caller-owned traversal arrays from adapter mutation.
    images, labels = adapter.canonical_batch(
        split,
        positions.copy(),
    )

    if not isinstance(images, np.ndarray):
        raise TypeError("canonical images must be numpy.ndarray")

    if not isinstance(labels, np.ndarray):
        raise TypeError("canonical labels must be numpy.ndarray")

    expected_images = (
        positions.size,
        *CANONICAL_IMAGE_SHAPE,
    )

    if images.shape != expected_images:
        raise ValueError(
            f"unexpected image shape: {images.shape}"
        )

    if labels.shape != (positions.size, num_classes):
        raise ValueError(
            f"unexpected one-hot label shape: {labels.shape}"
        )

    if images.dtype != np.float32:
        raise TypeError("canonical images must be float32")

    if labels.dtype != np.float32:
        raise TypeError("canonical labels must be float32")

    if not np.all(np.isfinite(images)):
        raise ValueError("nonfinite canonical image data")

    if np.any(images < 0.0) or np.any(images > 1.0):
        raise ValueError("canonical images outside [0,1]")

    if not np.all(np.isfinite(labels)):
        raise ValueError("nonfinite canonical label data")

    if not np.all((labels == 0.0) | (labels == 1.0)):
        raise ValueError("canonical labels are not binary one-hot")

    if not np.all(np.sum(labels, axis=1) == 1.0):
        raise ValueError(
            "canonical labels must have exactly one active class"
        )

    return images, labels


__all__ = [
    "CANONICAL_IMAGE_SHAPE",
    "canonical_vae_batch",
]
