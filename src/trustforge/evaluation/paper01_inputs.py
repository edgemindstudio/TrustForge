"""Bounded canonical real-data views for Paper 1 evaluation.

Validates supplied batches and declared identity. Does not authenticate dataset
provenance, select training traversal, train classifiers or authorize evaluation.
"""

import re

import numpy as np


CLASS_COUNTS = {
    "ustc_tfc2016": 9,
    "cicmaldroid2020": 5,
    "malnet_image": 47,
    "malex": 2,
}
MAX_BATCH_SIZE = 128
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def validate_batch(images, labels, *, count, class_count):
    """Require explicit canonical images and strict one-hot labels."""
    if (
        not isinstance(images, np.ndarray)
        or images.shape != (count, 40, 40, 3)
        or images.dtype != np.float32
    ):
        raise ValueError("invalid canonical image shape or dtype")
    if (
        not np.isfinite(images).all()
        or np.any(images < 0)
        or np.any(images > 1)
    ):
        raise ValueError("invalid canonical image values")
    if (
        not isinstance(labels, np.ndarray)
        or labels.shape != (count, class_count)
        or labels.dtype != np.float32
        or not np.isfinite(labels).all()
        or not np.all((labels == 0) | (labels == 1))
        or not np.all(labels.sum(axis=1) == 1)
    ):
        raise ValueError("invalid canonical one-hot labels")


class RealSplitView:
    """Position-addressed view; holds an adapter, never a full image array."""

    def __init__(self, *, adapter, dataset_id, split, provenance_sha256):
        if dataset_id not in CLASS_COUNTS:
            raise ValueError("unsupported dataset")
        if split not in ("train", "validation", "test"):
            raise ValueError("unsupported evaluation split")
        if (
            type(provenance_sha256) is not str
            or HEX64.fullmatch(provenance_sha256) is None
        ):
            raise ValueError("declared provenance digest required")
        if not all(callable(getattr(adapter, name, None)) for name in (
            "split_size", "canonical_batch",
        )):
            raise TypeError("adapter lacks required interfaces")
        size = adapter.split_size(split)
        if type(size) is not int or size <= 0:
            raise ValueError("split size must be a positive integer")
        self._adapter = adapter
        self._dataset_id = dataset_id
        self._split = split
        self._size = size
        self._provenance = provenance_sha256

    @property
    def size(self):
        return self._size

    @property
    def identity(self):
        return {
            "dataset_id": self._dataset_id,
            "split": self._split,
            "size": self._size,
            "class_count": CLASS_COUNTS[self._dataset_id],
            "declared_provenance_sha256": self._provenance,
        }

    def batch(self, positions):
        if (
            not isinstance(positions, np.ndarray)
            or positions.dtype != np.int64
            or positions.ndim != 1
            or not 1 <= positions.size <= MAX_BATCH_SIZE
            or np.any(positions < 0)
            or np.any(positions >= self._size)
            or np.unique(positions).size != positions.size
        ):
            raise ValueError("invalid bounded logical positions")
        observed = self._adapter.split_size(self._split)
        if type(observed) is not int or observed != self._size:
            raise RuntimeError("adapter split size changed")
        requested = positions.copy()
        adapter_positions = requested.copy()
        images, labels = self._adapter.canonical_batch(
            self._split, adapter_positions,
        )
        if not np.array_equal(adapter_positions, requested):
            raise RuntimeError("adapter mutated requested positions")
        validate_batch(
            images, labels, count=len(requested),
            class_count=CLASS_COUNTS[self._dataset_id],
        )
        return requested, images, labels

    def iter_ordered(self, *, batch_size=MAX_BATCH_SIZE):
        """Complete ordered traversal; does not prescribe training shuffling."""
        if type(batch_size) is not int or not 1 <= batch_size <= MAX_BATCH_SIZE:
            raise ValueError("invalid bounded batch size")
        for start in range(0, self._size, batch_size):
            yield self.batch(np.arange(
                start, min(start + batch_size, self._size), dtype=np.int64,
            ))
