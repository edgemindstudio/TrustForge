"""Structured synthetic VAE diagnostic data.

Synthetic development only. No real dataset access, training,
checkpoint creation, or artifact publication.
"""

from __future__ import annotations

import numpy as np


IMAGE_SHAPE = (40, 40, 3)
CLASS_COUNT = 2


class StructuredSyntheticVAEAdapter:
    """Deterministic two-class structured synthetic image adapter."""

    def __init__(
        self,
        *,
        train_count: int = 512,
        validation_count: int = 128,
        seed: int = 2026,
    ):
        for name, value in (
            ("train_count", train_count),
            ("validation_count", validation_count),
        ):
            if type(value) is not int or value < 2:
                raise ValueError(f"{name} must be an integer >= 2")

        if type(seed) is not int or not 0 <= seed < 2**31:
            raise ValueError("seed must be a bounded integer")

        self.seed = seed
        self.data = {}

        for split, count, offset in (
            ("train", train_count, 0),
            ("validation", validation_count, 1),
        ):
            rng = np.random.default_rng(
                np.random.SeedSequence([seed, offset])
            )

            labels = np.arange(count, dtype=np.int64) % CLASS_COUNT
            labels = labels[rng.permutation(count)]

            images = self._generate(rng, labels)

            onehot = np.eye(
                CLASS_COUNT, dtype=np.float32
            )[labels]

            self.data[split] = (images, onehot)

    @staticmethod
    def _generate(rng, labels):
        count = len(labels)

        yy, xx = np.mgrid[0:40, 0:40]
        xx = xx.astype(np.float32)
        yy = yy.astype(np.float32)

        horizontal_shift = rng.uniform(-3.0, 3.0, count)
        vertical_shift = rng.uniform(-3.0, 3.0, count)
        intensity = rng.uniform(0.55, 0.95, count)

        images = np.empty(
            (count, *IMAGE_SHAPE),
            dtype=np.float32,
        )

        for index, label in enumerate(labels):
            center_x = (
                12.0 if int(label) == 0 else 28.0
            ) + horizontal_shift[index]

            center_y = 20.0 + vertical_shift[index]

            radius = (
                (xx - center_x) ** 2
                + (yy - center_y) ** 2
            )

            pattern = intensity[index] * np.exp(
                -radius / (2.0 * 4.5**2)
            )

            images[index] = np.repeat(
                pattern[..., None],
                3,
                axis=2,
            ).astype(np.float32)

        if (
            not np.all(np.isfinite(images))
            or np.any(images < 0.0)
            or np.any(images > 1.0)
        ):
            raise RuntimeError("invalid synthetic images")

        return images

    def split_size(self, split: str) -> int:
        if split not in self.data:
            raise ValueError("unauthorized synthetic split")
        return int(self.data[split][0].shape[0])

    def canonical_batch(self, split, logical_positions):
        if split not in self.data:
            raise ValueError("unauthorized synthetic split")

        positions = np.asarray(logical_positions)

        if positions.ndim != 1:
            raise ValueError("positions must be one-dimensional")

        if positions.dtype.kind not in ("i", "u"):
            raise TypeError("positions must contain integers")

        if np.any(positions >= self.split_size(split)):
            raise IndexError("synthetic position out of bounds")

        if positions.dtype.kind == "i" and np.any(positions < 0):
            raise IndexError("negative synthetic position")

        images, labels = self.data[split]

        return (
            images[positions].copy(),
            labels[positions].copy(),
        )


__all__ = [
    "StructuredSyntheticVAEAdapter",
    "IMAGE_SHAPE",
    "CLASS_COUNT",
]
