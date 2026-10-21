"""Governed validation evaluation for the Paper 1 revision-2026 GMM.

Validation is class-specific and uses the complete governed validation subset
for that class.

The reported objective is the example-count-weighted mean censored-mixture
negative log likelihood. Equivalently, this implementation sums per-example
negative log probabilities across every validation batch and divides exactly
once by the total number of validation examples.

Validation does not compute gradients and does not mutate model state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import tensorflow as tf

from trustforge.models.paper01_revision_2026.gaussian_mixture_model import (
    CensoredDiagonalGaussianMixture,
)


VALIDATION_BATCH_SIZE_MAX = 128


class CanonicalBatchAdapter(Protocol):
    """Minimum dataset-adapter interface required for GMM validation."""

    def canonical_batch(
        self,
        split: str,
        logical_positions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        ...


@dataclass(frozen=True)
class GMMValidationResult:
    """Detached diagnostics from one complete class-validation evaluation."""

    mean_nll: float
    example_count: int
    batch_count: int


def _validate_class_positions(
    logical_positions: np.ndarray,
) -> np.ndarray:
    """Validate exact class-specific split-global logical positions."""

    positions = np.asarray(
        logical_positions
    )

    if positions.ndim != 1:
        raise ValueError(
            "validation logical_positions must be one-dimensional"
        )

    if positions.size == 0:
        raise ValueError(
            "validation logical_positions must not be empty"
        )

    if positions.dtype.kind not in {
        "i",
        "u",
    }:
        raise TypeError(
            "validation logical_positions must contain integers"
        )

    if np.any(
        positions < 0
    ):
        raise ValueError(
            "validation logical_positions must be non-negative"
        )

    normalized = positions.astype(
        np.int64,
        copy=False,
    )

    if np.unique(
        normalized
    ).size != normalized.size:
        raise ValueError(
            "validation logical_positions contain duplicates"
        )

    return normalized


def _validate_one_hot_labels(
    labels: np.ndarray,
    *,
    expected_batch_size: int,
    class_id: int | None,
) -> None:
    """Validate adapter labels and optional class-specific membership."""

    array = np.asarray(
        labels
    )

    if array.ndim != 2:
        raise ValueError(
            "validation labels must be rank-2 one-hot arrays"
        )

    if array.shape[
        0
    ] != expected_batch_size:
        raise ValueError(
            "validation label count does not match image batch size"
        )

    if array.dtype != np.float32:
        raise TypeError(
            "validation one-hot labels must be float32"
        )

    if not np.isfinite(
        array
    ).all():
        raise ValueError(
            "validation one-hot labels contain non-finite values"
        )

    if np.any(
        array < 0.0
    ) or np.any(
        array > 1.0
    ):
        raise ValueError(
            "validation one-hot labels are outside [0,1]"
        )

    row_sums = np.sum(
        array,
        axis=1,
        dtype=np.float64,
    )

    if not np.array_equal(
        row_sums,
        np.ones(
            expected_batch_size,
            dtype=np.float64,
        ),
    ):
        raise ValueError(
            "validation labels are not exact one-hot rows"
        )

    if not np.all(
        np.count_nonzero(
            array,
            axis=1,
        )
        == 1
    ):
        raise ValueError(
            "validation labels are not exact one-hot rows"
        )

    if class_id is None:
        return

    if isinstance(
        class_id,
        bool,
    ):
        raise TypeError(
            "class_id must be an integer, not bool"
        )

    try:
        governed_class = int(
            class_id
        )
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "class_id must be an integer"
        ) from exc

    if governed_class != class_id:
        raise ValueError(
            "class_id must be exactly integer-valued"
        )

    if governed_class < 0 or governed_class >= array.shape[
        1
    ]:
        raise ValueError(
            "class_id is outside one-hot label width"
        )

    observed = np.argmax(
        array,
        axis=1,
    )

    if not np.all(
        observed
        == governed_class
    ):
        raise ValueError(
            "validation positions include examples outside governed class"
        )


def evaluate_full_class_validation_nll(
    model: CensoredDiagonalGaussianMixture,
    *,
    adapter: CanonicalBatchAdapter,
    logical_positions: np.ndarray,
    class_id: int | None = None,
    split: str = "validation",
    batch_size_max: int = VALIDATION_BATCH_SIZE_MAX,
) -> GMMValidationResult:
    """Evaluate full class-specific mean censored NLL without gradients."""

    if not isinstance(
        model,
        CensoredDiagonalGaussianMixture,
    ):
        raise TypeError(
            "model must be CensoredDiagonalGaussianMixture"
        )

    positions = _validate_class_positions(
        logical_positions
    )

    if isinstance(
        batch_size_max,
        bool,
    ):
        raise TypeError(
            "batch_size_max must be an integer, not bool"
        )

    try:
        batch_limit = int(
            batch_size_max
        )
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "batch_size_max must be an integer"
        ) from exc

    if batch_limit != batch_size_max:
        raise ValueError(
            "batch_size_max must be exactly integer-valued"
        )

    if batch_limit < 1:
        raise ValueError(
            "batch_size_max must be positive"
        )

    if batch_limit > VALIDATION_BATCH_SIZE_MAX:
        raise ValueError(
            "validation batch size exceeds governed maximum 128"
        )

    # Validation is deliberately accumulated as a float64 sum of per-example
    # NLL values. No batch-mean averaging is used.
    total_nll = np.float64(
        0.0
    )

    total_examples = 0
    batch_count = 0

    for start in range(
        0,
        positions.size,
        batch_limit,
    ):
        stop = min(
            start
            + batch_limit,
            positions.size,
        )

        batch_positions = positions[
            start:stop
        ]

        images, labels = adapter.canonical_batch(
            split,
            batch_positions,
        )

        images_array = np.asarray(
            images
        )

        if images_array.shape[
            0
        ] != batch_positions.size:
            raise ValueError(
                "validation image count does not match requested positions"
            )

        if images_array.dtype != np.float32:
            raise TypeError(
                "validation canonical images must be float32"
            )

        _validate_one_hot_labels(
            labels,
            expected_batch_size=int(
                batch_positions.size
            ),
            class_id=class_id,
        )

        with tf.GradientTape(
            watch_accessed_variables=False
        ) as tape:
            # Nothing is watched: validation must never produce gradients.
            del tape

            log_prob = model.log_prob(
                tf.convert_to_tensor(
                    images_array,
                    dtype=tf.float32,
                )
            )

        tf.debugging.assert_rank(
            log_prob,
            1,
            message="GMM validation log_prob must be rank 1",
        )

        tf.debugging.assert_equal(
            tf.shape(
                log_prob
            )[0],
            batch_positions.size,
            message=(
                "GMM validation log_prob count does not match batch size"
            ),
        )

        tf.debugging.assert_all_finite(
            log_prob,
            "GMM validation produced non-finite log probabilities",
        )

        batch_nll = -np.asarray(
            log_prob.numpy(),
            dtype=np.float64,
        )

        if not np.isfinite(
            batch_nll
        ).all():
            raise RuntimeError(
                "GMM validation produced non-finite per-example NLL"
            )

        total_nll = np.float64(
            total_nll
            + np.sum(
                batch_nll,
                dtype=np.float64,
            )
        )

        total_examples += int(
            batch_positions.size
        )

        batch_count += 1

    if total_examples != positions.size:
        raise RuntimeError(
            "GMM validation did not evaluate every governed example exactly once"
        )

    if batch_count < 1:
        raise RuntimeError(
            "GMM validation produced no batches"
        )

    mean_nll = np.float64(
        total_nll
        / np.float64(
            total_examples
        )
    )

    if not np.isfinite(
        mean_nll
    ):
        raise RuntimeError(
            "GMM full-class validation NLL is non-finite"
        )

    # Validation must leave authoritative model state finite.
    model.state_arrays()

    tf.debugging.assert_all_finite(
        model.scales(),
        "GMM validation observed non-finite transformed scales",
    )

    tf.debugging.assert_all_finite(
        model.mixture_weights(),
        "GMM validation observed non-finite mixture weights",
    )

    return GMMValidationResult(
        mean_nll=float(
            mean_nll
        ),
        example_count=total_examples,
        batch_count=batch_count,
    )


__all__ = [
    "CanonicalBatchAdapter",
    "GMMValidationResult",
    "VALIDATION_BATCH_SIZE_MAX",
    "evaluate_full_class_validation_nll",
]
