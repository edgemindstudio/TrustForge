"""Candidate deterministic logical-position traversal for revised VAE.

Paper 1 revision 2026.

This module implements an engineering candidate, NOT an
approved scientific training schedule.

It never loads canonical images, accesses the test partition,
selects training budgets, or authorizes real-data training.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np


def _positive_integer(value, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a positive integer")

    if not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be a positive integer")

    result = int(value)

    if result <= 0:
        raise ValueError(f"{name} must be positive")

    return result


def _nonnegative_integer(value, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a nonnegative integer")

    if not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be a nonnegative integer")

    result = int(value)

    if result < 0:
        raise ValueError(f"{name} must be nonnegative")

    return result


def _require_split(split: str) -> str:
    if not isinstance(split, str):
        raise TypeError("split must be a string")

    if split not in ("train", "validation"):
        raise ValueError(
            "VAE traversal permits only train or validation"
        )

    return split


def _require_adapter(adapter):
    split_size = getattr(adapter, "split_size", None)

    if not callable(split_size):
        raise TypeError(
            "adapter must provide callable split_size(split)"
        )

    return adapter


def training_position_batches(
    *,
    adapter,
    batch_size: int,
    experimental_seed: int,
    epoch: int,
) -> Iterator[np.ndarray]:
    """Yield a deterministic complete permutation of train positions.

    Exactly one visit per logical training position per epoch.
    The final batch may be smaller than batch_size.

    No class reweighting or replacement sampling is applied.
    This is a candidate policy, pending scientific approval.
    """
    adapter = _require_adapter(adapter)

    size = _positive_integer(
        adapter.split_size("train"),
        name="training split size",
    )

    batch_size = _positive_integer(
        batch_size,
        name="batch_size",
    )

    seed = _nonnegative_integer(
        experimental_seed,
        name="experimental_seed",
    )

    epoch = _nonnegative_integer(
        epoch,
        name="epoch",
    )

    # Explicit two-word entropy distinguishes epochs without
    # relying on global NumPy random state.
    rng = np.random.default_rng(
        np.random.SeedSequence([seed, epoch])
    )

    positions = rng.permutation(size).astype(
        np.int64,
        copy=False,
    )

    for start in range(0, size, batch_size):
        yield positions[start:start + batch_size].copy()


def validation_position_batches(
    *,
    adapter,
    batch_size: int,
) -> Iterator[np.ndarray]:
    """Yield one complete, ordered pass over validation positions."""
    adapter = _require_adapter(adapter)

    size = _positive_integer(
        adapter.split_size("validation"),
        name="validation split size",
    )

    batch_size = _positive_integer(
        batch_size,
        name="batch_size",
    )

    for start in range(0, size, batch_size):
        stop = min(start + batch_size, size)

        yield np.arange(
            start,
            stop,
            dtype=np.int64,
        )


def position_batches(
    *,
    adapter,
    split: str,
    batch_size: int,
    experimental_seed: int | None = None,
    epoch: int | None = None,
) -> Iterator[np.ndarray]:
    """Dispatch exclusively to governed train/validation traversal."""
    split = _require_split(split)

    if split == "train":
        if experimental_seed is None or epoch is None:
            raise ValueError(
                "training traversal requires seed and epoch"
            )

        yield from training_position_batches(
            adapter=adapter,
            batch_size=batch_size,
            experimental_seed=experimental_seed,
            epoch=epoch,
        )
        return

    if experimental_seed is not None or epoch is not None:
        raise ValueError(
            "validation position traversal does not accept "
            "training seed or epoch"
        )

    yield from validation_position_batches(
        adapter=adapter,
        batch_size=batch_size,
    )


__all__ = [
    "training_position_batches",
    "validation_position_batches",
    "position_batches",
]
