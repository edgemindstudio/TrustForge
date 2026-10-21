"""Governed class-specific traversal for the Paper 1 revision-2026 GMM.

This module owns only deterministic training-position traversal and balanced
batch geometry.

Training-budget selection, optimizer behavior, checkpointing, and model
updates remain governed separately.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import numpy as np


MAX_BATCH_SIZE = 128
MAX_BATCH_SIZE_SPREAD = 1


def _validate_positive_integer(
    value: int,
    *,
    name: str,
) -> int:
    """Return one validated strictly positive integer."""

    if isinstance(
        value,
        bool,
    ):
        raise TypeError(
            f"{name} must be an integer, not bool"
        )

    try:
        converted = int(
            value
        )
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{name} must be an integer"
        ) from exc

    if converted != value:
        raise ValueError(
            f"{name} must be exactly integer-valued"
        )

    if converted <= 0:
        raise ValueError(
            f"{name} must be positive"
        )

    return converted


def _validate_seed(
    experimental_seed: int,
) -> int:
    """Validate a NumPy-compatible governed experimental seed."""

    if isinstance(
        experimental_seed,
        bool,
    ):
        raise TypeError(
            "experimental_seed must be an integer, not bool"
        )

    try:
        seed = int(
            experimental_seed
        )
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "experimental_seed must be an integer"
        ) from exc

    if seed != experimental_seed:
        raise ValueError(
            "experimental_seed must be exactly integer-valued"
        )

    if seed < 0 or seed > np.iinfo(
        np.uint32
    ).max:
        raise ValueError(
            "experimental_seed must lie in [0, 2**32 - 1]"
        )

    return seed


def batches_per_pass(
    class_count: int,
) -> int:
    """Return governed ``ceil(N / 128)`` batches per complete pass."""

    n = _validate_positive_integer(
        class_count,
        name="class_count",
    )

    return math.ceil(
        n
        / MAX_BATCH_SIZE
    )


def balanced_batch_sizes(
    class_count: int,
) -> tuple[int, ...]:
    """Return the governed balanced batch-size sequence for one pass.

    Let:

    ``B = ceil(N / 128)``

    and:

    ``q, r = divmod(N, B)``.

    The first ``r`` batches contain ``q + 1`` examples and the remaining
    ``B - r`` batches contain ``q`` examples.
    """

    n = _validate_positive_integer(
        class_count,
        name="class_count",
    )

    batch_count = batches_per_pass(
        n
    )

    q, r = divmod(
        n,
        batch_count,
    )

    sizes = (
        (q + 1,) * r
        + (q,) * (
            batch_count
            - r
        )
    )

    if len(
        sizes
    ) != batch_count:
        raise RuntimeError(
            "balanced batch count does not match governed batches per pass"
        )

    if sum(
        sizes
    ) != n:
        raise RuntimeError(
            "balanced batch sizes do not sum to class count"
        )

    if min(
        sizes
    ) < 1:
        raise RuntimeError(
            "balanced traversal produced an empty batch"
        )

    if max(
        sizes
    ) > MAX_BATCH_SIZE:
        raise RuntimeError(
            "balanced traversal batch exceeds governed maximum size"
        )

    if (
        max(
            sizes
        )
        - min(
            sizes
        )
        > MAX_BATCH_SIZE_SPREAD
    ):
        raise RuntimeError(
            "balanced traversal batch-size spread exceeds one"
        )

    return sizes


def _validate_class_positions(
    class_positions: np.ndarray,
) -> np.ndarray:
    """Validate one ordered class-specific logical-position sequence."""

    positions = np.asarray(
        class_positions
    )

    if positions.ndim != 1:
        raise ValueError(
            "class_positions must be one-dimensional"
        )

    if positions.size == 0:
        raise ValueError(
            "class_positions must not be empty"
        )

    if positions.dtype.kind not in {
        "i",
        "u",
    }:
        raise TypeError(
            "class_positions must contain integer values"
        )

    if np.any(
        positions < 0
    ):
        raise ValueError(
            "class_positions must be non-negative"
        )

    normalized = positions.astype(
        np.int64,
        copy=False,
    )

    if np.unique(
        normalized
    ).size != normalized.size:
        raise ValueError(
            "class_positions contain duplicates"
        )

    return normalized


def iter_training_batches(
    class_positions: np.ndarray,
    *,
    experimental_seed: int,
    max_updates: int,
) -> Iterator[np.ndarray]:
    """Yield governed deterministic class-training batches.

    A complete class permutation is generated at the beginning of every
    attempted pass. The permutation is sliced sequentially using the governed
    balanced batch-size sequence.

    Traversal stops exactly after ``max_updates`` batches, including when that
    limit is reached partway through a later pass.
    """

    positions = _validate_class_positions(
        class_positions
    )

    seed = _validate_seed(
        experimental_seed
    )

    update_limit = _validate_positive_integer(
        max_updates,
        name="max_updates",
    )

    sizes = balanced_batch_sizes(
        int(
            positions.size
        )
    )

    rng = np.random.default_rng(
        seed
    )

    updates = 0

    while updates < update_limit:
        permutation_indices = rng.permutation(
            positions.size
        )

        permuted_positions = positions[
            permutation_indices
        ]

        cursor = 0

        for size in sizes:
            if updates >= update_limit:
                return

            stop = (
                cursor
                + size
            )

            batch = permuted_positions[
                cursor:stop
            ]

            if batch.size != size:
                raise RuntimeError(
                    "traversal batch size disagrees with governed geometry"
                )

            if batch.size == 0:
                raise RuntimeError(
                    "traversal produced an empty batch"
                )

            if batch.size > MAX_BATCH_SIZE:
                raise RuntimeError(
                    "traversal produced an oversized batch"
                )

            yield batch.copy()

            updates += 1
            cursor = stop

        if cursor != positions.size:
            raise RuntimeError(
                "complete traversal pass did not consume every class position"
            )


__all__ = [
    "MAX_BATCH_SIZE",
    "MAX_BATCH_SIZE_SPREAD",
    "balanced_batch_sizes",
    "batches_per_pass",
    "iter_training_batches",
]
