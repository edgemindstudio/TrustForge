"""Governed traversal for the Paper 1 revision-2026 RBM.

Traversal semantics:

- class probability is proportional to n_c**0.85;
- sampling within the selected class is uniform;
- sampling is with replacement;
- batch size is exactly 256;
- one nominal epoch contains N_train sampled draws;
- every draw is deterministically addressable by:
      (dataset_id, experimental_seed, training_position)
- ambient process-global randomness is never authoritative.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import operator
from collections.abc import Mapping

import numpy as np


CLASS_BALANCE_EXPONENT = 0.85
BATCH_SIZE = 256


@dataclass(frozen=True)
class RBMTraversalSpec:
    """Validated dataset-specific traversal specification."""

    dataset_id: str
    class_ids: np.ndarray
    class_counts: np.ndarray
    class_probabilities: np.ndarray
    total_training_examples: int


@dataclass(frozen=True)
class RBMTraversalBatch:
    """One governed sampled RBM training batch."""

    training_positions: np.ndarray
    class_ids: np.ndarray
    logical_positions: np.ndarray


def _exact_nonnegative_integer(
    value: int,
    *,
    name: str,
) -> int:
    if isinstance(
        value,
        bool,
    ):
        raise TypeError(
            f"{name} must be an exact integer, not bool"
        )

    try:
        result = operator.index(
            value
        )
    except TypeError as exc:
        raise TypeError(
            f"{name} must be an exact integer"
        ) from exc

    if result < 0:
        raise ValueError(
            f"{name} must be non-negative"
        )

    return result


def _exact_positive_integer(
    value: int,
    *,
    name: str,
) -> int:
    result = _exact_nonnegative_integer(
        value,
        name=name,
    )

    if result == 0:
        raise ValueError(
            f"{name} must be greater than zero"
        )

    return result


def _validate_dataset_id(
    dataset_id: str,
) -> str:
    if not isinstance(
        dataset_id,
        str,
    ):
        raise TypeError(
            "dataset_id must be a string"
        )

    normalized = dataset_id.strip()

    if not normalized:
        raise ValueError(
            "dataset_id must not be empty"
        )

    return normalized


def _validate_seed(
    experimental_seed: int,
) -> int:
    seed = _exact_nonnegative_integer(
        experimental_seed,
        name="experimental_seed",
    )

    if seed > np.iinfo(
        np.uint32
    ).max:
        raise ValueError(
            "experimental_seed must lie in [0, 2**32 - 1]"
        )

    return seed


def _normalize_class_positions(
    class_positions: Mapping[
        int,
        np.ndarray,
    ],
) -> tuple[
    np.ndarray,
    tuple[np.ndarray, ...],
]:
    if not isinstance(
        class_positions,
        Mapping,
    ):
        raise TypeError(
            "class_positions must be a mapping"
        )

    if not class_positions:
        raise ValueError(
            "class_positions must not be empty"
        )

    normalized_items = []

    for raw_class_id, raw_positions in class_positions.items():
        class_id = _exact_nonnegative_integer(
            raw_class_id,
            name="class id",
        )

        positions = np.asarray(
            raw_positions
        )

        if positions.ndim != 1:
            raise ValueError(
                "each class position sequence must be one-dimensional"
            )

        if positions.size == 0:
            raise ValueError(
                "each class must contain at least one training position"
            )

        if positions.dtype.kind not in {
            "i",
            "u",
        }:
            raise TypeError(
                "class positions must contain integers"
            )

        if np.any(
            positions < 0
        ):
            raise ValueError(
                "class positions must be non-negative"
            )

        normalized = positions.astype(
            np.int64,
            copy=False,
        )

        if np.unique(
            normalized
        ).size != normalized.size:
            raise ValueError(
                "class positions contain duplicates within one class"
            )

        normalized_items.append(
            (
                class_id,
                normalized.copy(),
            )
        )

    normalized_items.sort(
        key=lambda item: item[0]
    )

    class_ids = np.asarray(
        [
            class_id
            for class_id, _ in normalized_items
        ],
        dtype=np.int64,
    )

    expected_ids = np.arange(
        class_ids.size,
        dtype=np.int64,
    )

    if not np.array_equal(
        class_ids,
        expected_ids,
    ):
        raise ValueError(
            "class ids must form the contiguous universe 0..K-1"
        )

    arrays = tuple(
        positions
        for _, positions in normalized_items
    )

    combined = np.concatenate(
        arrays
    )

    if np.unique(
        combined
    ).size != combined.size:
        raise ValueError(
            "training logical positions overlap across classes"
        )

    return (
        class_ids,
        arrays,
    )


def build_traversal_spec(
    *,
    dataset_id: str,
    class_positions: Mapping[
        int,
        np.ndarray,
    ],
) -> RBMTraversalSpec:
    """Build one governed dataset-specific RBM traversal specification."""

    governed_dataset_id = _validate_dataset_id(
        dataset_id
    )

    class_ids, position_arrays = _normalize_class_positions(
        class_positions
    )

    class_counts = np.asarray(
        [
            positions.size
            for positions in position_arrays
        ],
        dtype=np.int64,
    )

    weights = np.power(
        class_counts.astype(
            np.float64
        ),
        CLASS_BALANCE_EXPONENT,
    )

    if not np.all(
        np.isfinite(weights)
    ):
        raise RuntimeError(
            "class-balance weights are non-finite"
        )

    weight_sum = np.sum(
        weights,
        dtype=np.float64,
    )

    if not np.isfinite(
        weight_sum
    ) or weight_sum <= 0.0:
        raise RuntimeError(
            "class-balance weight sum is invalid"
        )

    probabilities = (
        weights
        / weight_sum
    ).astype(
        np.float64,
        copy=False,
    )

    if not np.all(
        np.isfinite(probabilities)
    ):
        raise RuntimeError(
            "class probabilities are non-finite"
        )

    if np.any(
        probabilities <= 0.0
    ):
        raise RuntimeError(
            "class probabilities must be strictly positive"
        )

    if not math.isclose(
        float(
            probabilities.sum(
                dtype=np.float64
            )
        ),
        1.0,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    ):
        raise RuntimeError(
            "class probabilities do not sum to one"
        )

    total_training_examples = int(
        class_counts.sum(
            dtype=np.int64
        )
    )

    return RBMTraversalSpec(
        dataset_id=governed_dataset_id,
        class_ids=class_ids,
        class_counts=class_counts,
        class_probabilities=probabilities,
        total_training_examples=total_training_examples,
    )


def nominal_epoch_draw_count(
    spec: RBMTraversalSpec,
) -> int:
    """Return governed N_train draws per nominal epoch."""

    if not isinstance(
        spec,
        RBMTraversalSpec,
    ):
        raise TypeError(
            "spec must be an RBMTraversalSpec"
        )

    return spec.total_training_examples


def batches_per_nominal_epoch(
    spec: RBMTraversalSpec,
) -> int:
    """Return ceil(N_train / 256) update batches per nominal epoch."""

    return math.ceil(
        nominal_epoch_draw_count(
            spec
        )
        / BATCH_SIZE
    )


def _dataset_entropy_words(
    dataset_id: str,
) -> tuple[
    int,
    int,
    int,
    int,
]:
    digest = hashlib.blake2b(
        dataset_id.encode(
            "utf-8"
        ),
        digest_size=16,
        person=b"TF-RBM-TRAV",
    ).digest()

    words = np.frombuffer(
        digest,
        dtype="<u4",
    )

    return tuple(
        int(word)
        for word in words
    )


def _rng_for_training_position(
    *,
    dataset_id: str,
    experimental_seed: int,
    training_position: int,
) -> np.random.Generator:
    """Create deterministic RNG for exactly one scientific training position."""

    dataset_words = _dataset_entropy_words(
        dataset_id
    )

    position = _exact_nonnegative_integer(
        training_position,
        name="training_position",
    )

    low = position & 0xFFFFFFFF
    high = (
        position >> 32
    ) & 0xFFFFFFFF

    seed_sequence = np.random.SeedSequence(
        [
            *dataset_words,
            experimental_seed,
            low,
            high,
        ]
    )

    return np.random.default_rng(
        seed_sequence
    )


def _validate_positions_against_spec(
    *,
    spec: RBMTraversalSpec,
    class_positions: Mapping[
        int,
        np.ndarray,
    ],
) -> tuple[np.ndarray, ...]:
    """Validate one class-position inventory against an existing spec."""

    if not isinstance(
        spec,
        RBMTraversalSpec,
    ):
        raise TypeError(
            "spec must be an RBMTraversalSpec"
        )

    class_ids, position_arrays = _normalize_class_positions(
        class_positions
    )

    if not np.array_equal(
        class_ids,
        spec.class_ids,
    ):
        raise ValueError(
            "class_positions do not match traversal specification class ids"
        )

    observed_counts = np.asarray(
        [
            positions.size
            for positions in position_arrays
        ],
        dtype=np.int64,
    )

    if not np.array_equal(
        observed_counts,
        spec.class_counts,
    ):
        raise ValueError(
            "class_positions do not match traversal specification counts"
        )

    return position_arrays


def _sample_from_validated_positions(
    *,
    spec: RBMTraversalSpec,
    position_arrays: tuple[np.ndarray, ...],
    experimental_seed: int,
    training_position: int,
) -> tuple[
    int,
    int,
]:
    """Sample one governed position from a prevalidated class inventory."""

    rng = _rng_for_training_position(
        dataset_id=spec.dataset_id,
        experimental_seed=experimental_seed,
        training_position=training_position,
    )

    class_index = int(
        rng.choice(
            spec.class_ids.size,
            p=spec.class_probabilities,
        )
    )

    selected_positions = position_arrays[
        class_index
    ]

    within_index = int(
        rng.integers(
            0,
            selected_positions.size,
        )
    )

    return (
        int(
            spec.class_ids[
                class_index
            ]
        ),
        int(
            selected_positions[
                within_index
            ]
        ),
    )


def sample_training_position(
    *,
    spec: RBMTraversalSpec,
    class_positions: Mapping[
        int,
        np.ndarray,
    ],
    experimental_seed: int,
    training_position: int,
) -> tuple[
    int,
    int,
]:
    """Return (class_id, logical_position) for one governed training position."""

    seed = _validate_seed(
        experimental_seed
    )

    position_arrays = _validate_positions_against_spec(
        spec=spec,
        class_positions=class_positions,
    )

    return _sample_from_validated_positions(
        spec=spec,
        position_arrays=position_arrays,
        experimental_seed=seed,
        training_position=training_position,
    )


def training_batch(
    *,
    spec: RBMTraversalSpec,
    class_positions: Mapping[
        int,
        np.ndarray,
    ],
    experimental_seed: int,
    start_training_position: int,
    batch_size: int = BATCH_SIZE,
) -> RBMTraversalBatch:
    """Construct one deterministic sampled batch from consecutive positions."""

    start = _exact_nonnegative_integer(
        start_training_position,
        name="start_training_position",
    )

    governed_batch_size = _exact_positive_integer(
        batch_size,
        name="batch_size",
    )

    if governed_batch_size > BATCH_SIZE:
        raise ValueError(
            "RBM batch_size must not exceed governed size 256"
        )

    seed = _validate_seed(
        experimental_seed
    )

    position_arrays = _validate_positions_against_spec(
        spec=spec,
        class_positions=class_positions,
    )

    training_positions = np.arange(
        start,
        start + governed_batch_size,
        dtype=np.int64,
    )

    class_ids = np.empty(
        governed_batch_size,
        dtype=np.int64,
    )

    logical_positions = np.empty(
        governed_batch_size,
        dtype=np.int64,
    )

    for offset, position in enumerate(
        training_positions
    ):
        class_id, logical_position = _sample_from_validated_positions(
            spec=spec,
            position_arrays=position_arrays,
            experimental_seed=seed,
            training_position=int(
                position
            ),
        )

        class_ids[
            offset
        ] = class_id

        logical_positions[
            offset
        ] = logical_position

    return RBMTraversalBatch(
        training_positions=training_positions,
        class_ids=class_ids,
        logical_positions=logical_positions,
    )


__all__ = [
    "BATCH_SIZE",
    "CLASS_BALANCE_EXPONENT",
    "RBMTraversalBatch",
    "RBMTraversalSpec",
    "batches_per_nominal_epoch",
    "build_traversal_spec",
    "nominal_epoch_draw_count",
    "sample_training_position",
    "training_batch",
]
