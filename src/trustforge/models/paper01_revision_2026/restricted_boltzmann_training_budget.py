"""Governed training budget for the Paper 1 revision-2026 RBM.

Selected semantics:

- batch size is fixed at 256;
- B = ceil(N_train / 256);
- max_updates = 2 * B;
- no minimum-update floor;
- validation occurs at steps 0, B, and 2B;
- no patience-based early stopping;
- reconstruction MSE is diagnostic only;
- the final governed update is authoritative.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import operator


BATCH_SIZE = 256


@dataclass(frozen=True)
class RBMTrainingBudget:
    """Deterministic governed RBM training-budget geometry."""

    training_example_count: int
    batch_size: int
    batches_per_nominal_epoch: int
    max_updates: int
    validation_steps: tuple[int, int, int]
    authoritative_step: int


def _exact_positive_integer(
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

    if result <= 0:
        raise ValueError(
            f"{name} must be greater than zero"
        )

    return result


def batches_per_nominal_epoch(
    training_example_count: int,
) -> int:
    """Return B = ceil(N_train / 256)."""

    n_train = _exact_positive_integer(
        training_example_count,
        name="training_example_count",
    )

    return math.ceil(
        n_train
        / BATCH_SIZE
    )


def max_updates(
    training_example_count: int,
) -> int:
    """Return the governed two-pass maximum update count."""

    B = batches_per_nominal_epoch(
        training_example_count
    )

    return 2 * B


def validation_steps(
    training_example_count: int,
) -> tuple[int, int, int]:
    """Return the exact governed validation schedule: 0, B, 2B."""

    B = batches_per_nominal_epoch(
        training_example_count
    )

    return (
        0,
        B,
        2 * B,
    )


def build_training_budget(
    training_example_count: int,
) -> RBMTrainingBudget:
    """Build one deterministic governed RBM training budget."""

    n_train = _exact_positive_integer(
        training_example_count,
        name="training_example_count",
    )

    B = batches_per_nominal_epoch(
        n_train
    )

    governed_max_updates = 2 * B

    governed_validation_steps = (
        0,
        B,
        governed_max_updates,
    )

    return RBMTrainingBudget(
        training_example_count=n_train,
        batch_size=BATCH_SIZE,
        batches_per_nominal_epoch=B,
        max_updates=governed_max_updates,
        validation_steps=governed_validation_steps,
        authoritative_step=governed_max_updates,
    )


__all__ = [
    "BATCH_SIZE",
    "RBMTrainingBudget",
    "batches_per_nominal_epoch",
    "build_training_budget",
    "max_updates",
    "validation_steps",
]
