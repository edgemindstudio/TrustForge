"""Governed manual-SGD optimizer for the Paper 1 revision-2026 RBM.

Selected optimizer semantics:

- manual SGD;
- constant learning rate 5e-4;
- zero weight decay;
- zero momentum;
- no adaptive optimizer state;
- no update clipping;
- float64 arithmetic;
- W, a, b, and U all receive positive-minus-negative CD statistics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from trustforge.models.paper01_revision_2026.restricted_boltzmann_cd import (
    RBMCD1Result,
)
from trustforge.models.paper01_revision_2026.restricted_boltzmann_model import (
    HIDDEN_UNITS,
    STATE_DTYPE,
    VISIBLE_DIM,
    SharedConditionalRBM,
)


LEARNING_RATE = 5.0e-4
WEIGHT_DECAY = 0.0


@dataclass(frozen=True)
class RBMOptimizerStepDiagnostics:
    """Evidence describing one governed manual-SGD update."""

    learning_rate: float
    weight_decay: float
    delta_W_norm: float
    delta_a_norm: float
    delta_b_norm: float
    delta_U_norm: float


def _validated_statistic(
    value: np.ndarray,
    *,
    name: str,
    expected_shape: tuple[int, ...],
) -> tf.Tensor:
    """Validate one governed CD statistic and return float64 tensor."""

    array = np.asarray(
        value
    )

    if array.dtype != np.float64:
        raise TypeError(
            f"{name} must have dtype float64"
        )

    if array.shape != expected_shape:
        raise ValueError(
            f"{name} has wrong governed shape: "
            f"expected {expected_shape}, got {array.shape}"
        )

    if not np.all(
        np.isfinite(array)
    ):
        raise ValueError(
            f"{name} contains non-finite values"
        )

    tensor = tf.convert_to_tensor(
        array,
        dtype=STATE_DTYPE,
    )

    tf.debugging.assert_all_finite(
        tensor,
        f"{name} contains non-finite values",
    )

    return tensor


def apply_manual_sgd_update(
    *,
    model: SharedConditionalRBM,
    cd_result: RBMCD1Result,
) -> RBMOptimizerStepDiagnostics:
    """Apply one governed revision-2026 manual-SGD parameter update."""

    if not isinstance(
        model,
        SharedConditionalRBM,
    ):
        raise TypeError(
            "model must be a SharedConditionalRBM"
        )

    if not isinstance(
        cd_result,
        RBMCD1Result,
    ):
        raise TypeError(
            "cd_result must be an RBMCD1Result"
        )

    delta_W = _validated_statistic(
        cd_result.delta_W,
        name="delta_W",
        expected_shape=(
            VISIBLE_DIM,
            HIDDEN_UNITS,
        ),
    )

    delta_a = _validated_statistic(
        cd_result.delta_a,
        name="delta_a",
        expected_shape=(
            VISIBLE_DIM,
        ),
    )

    delta_b = _validated_statistic(
        cd_result.delta_b,
        name="delta_b",
        expected_shape=(
            HIDDEN_UNITS,
        ),
    )

    delta_U = _validated_statistic(
        cd_result.delta_U,
        name="delta_U",
        expected_shape=(
            model.class_count,
            HIDDEN_UNITS,
        ),
    )

    learning_rate = tf.constant(
        LEARNING_RATE,
        dtype=STATE_DTYPE,
    )

    candidate_W = (
        model.W
        + learning_rate
        * delta_W
    )
    candidate_a = (
        model.a
        + learning_rate
        * delta_a
    )
    candidate_b = (
        model.b
        + learning_rate
        * delta_b
    )
    candidate_U = (
        model.U
        + learning_rate
        * delta_U
    )

    for name, candidate in (
        (
            "W",
            candidate_W,
        ),
        (
            "a",
            candidate_a,
        ),
        (
            "b",
            candidate_b,
        ),
        (
            "U",
            candidate_U,
        ),
    ):
        tf.debugging.assert_type(
            candidate,
            STATE_DTYPE,
            message=f"updated RBM {name} must remain float64",
        )

        tf.debugging.assert_all_finite(
            candidate,
            f"updated RBM {name} contains non-finite values",
        )

    model.W.assign(
        candidate_W
    )
    model.a.assign(
        candidate_a
    )
    model.b.assign(
        candidate_b
    )
    model.U.assign(
        candidate_U
    )

    return RBMOptimizerStepDiagnostics(
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        delta_W_norm=float(
            np.linalg.norm(
                cd_result.delta_W
            )
        ),
        delta_a_norm=float(
            np.linalg.norm(
                cd_result.delta_a
            )
        ),
        delta_b_norm=float(
            np.linalg.norm(
                cd_result.delta_b
            )
        ),
        delta_U_norm=float(
            np.linalg.norm(
                cd_result.delta_U
            )
        ),
    )


__all__ = [
    "LEARNING_RATE",
    "WEIGHT_DECAY",
    "RBMOptimizerStepDiagnostics",
    "apply_manual_sgd_update",
]
