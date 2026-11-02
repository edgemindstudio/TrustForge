"""Governed real-data class sampler for Paper 1 revision-2026 GAN.

This module does not load images, train models, or select synthetic
generation labels. Its inputs must represent the governed training split.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


GAN_CLASS_SAMPLING_ALPHA = 0.85


@dataclass
class GANTrainingClassSampler:
    """Draw governed training positions with power-law class weighting."""

    class_positions: tuple[np.ndarray, ...]
    seed: int
    alpha: float = GAN_CLASS_SAMPLING_ALPHA

    def __post_init__(self) -> None:
        if isinstance(self.seed, (bool, np.bool_)) or not isinstance(
            self.seed, (int, np.integer)
        ) or not 0 <= int(self.seed) < 2**64:
            raise ValueError("seed must be an unsigned 64-bit integer")

        if not np.isfinite(self.alpha) or self.alpha != 0.85:
            raise ValueError("GAN class-sampling exponent must be 0.85")

        if not self.class_positions:
            raise ValueError("at least one class is required")

        checked = []
        for class_id, positions in enumerate(self.class_positions):
            raw = np.asarray(positions)

            if raw.ndim != 1 or raw.size == 0:
                raise ValueError(
                    f"class {class_id} must have nonempty 1D positions"
                )

            if raw.dtype.kind not in "iu":
                raise ValueError(
                    f"class {class_id} positions must be integers"
                )

            if raw.dtype.kind == "i" and np.any(raw < 0):
                raise ValueError("training positions cannot be negative")

            checked.append(np.asarray(raw, dtype=np.int64).copy())

        joined = np.concatenate(checked)
        if np.unique(joined).size != joined.size:
            raise ValueError("training positions must be disjoint")

        self.class_positions = tuple(checked)
        self.counts = np.asarray(
            [len(x) for x in checked], dtype=np.int64
        )

        weights = self.counts.astype(np.float64) ** self.alpha
        self.probabilities = weights / weights.sum()
        self.rng = np.random.default_rng(int(self.seed))

    def draw(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (logical_positions, class_ids) for real training data."""

        if isinstance(batch_size, bool) or not isinstance(
            batch_size, (int, np.integer)
        ) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        classes = self.rng.choice(
            len(self.class_positions),
            size=int(batch_size),
            p=self.probabilities,
        )

        positions = np.empty(int(batch_size), dtype=np.int64)

        for class_id, available in enumerate(self.class_positions):
            selected = np.flatnonzero(classes == class_id)
            if selected.size:
                offsets = self.rng.integers(
                    0, len(available), size=len(selected)
                )
                positions[selected] = available[offsets]

        return positions, classes.astype(np.int32)
