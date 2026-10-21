"""Governed training-budget schedule for the Paper 1 revision-2026 GMM.

This module owns only deterministic budget arithmetic:

- balanced batches per pass;
- maximum optimizer updates;
- validation interval;
- exact validation checkpoint steps.

Traversal, optimization, validation evaluation, persistence, and sampling are
implemented separately.
"""

from __future__ import annotations

from dataclasses import dataclass

from trustforge.models.paper01_revision_2026.gaussian_mixture_traversal import (
    batches_per_pass,
)


MINIMUM_UPDATE_BUDGET = 100
MAXIMUM_PASS_MULTIPLIER = 5
MINIMUM_VALIDATION_INTERVAL_UPDATES = 20
POST_UPDATE_VALIDATION_EVALUATIONS = 5
TOTAL_VALIDATION_CHECKPOINT_OBSERVATIONS = 6


@dataclass(frozen=True)
class GMMTrainingBudget:
    """Exact governed budget for one class-specific GMM run."""

    class_count: int
    batches_per_pass: int
    max_updates: int
    validation_interval: int
    validation_steps: tuple[int, ...]


def training_budget(
    class_count: int,
) -> GMMTrainingBudget:
    """Return the exact governed budget for one class model."""

    batch_count = batches_per_pass(
        class_count
    )

    max_updates = max(
        MINIMUM_UPDATE_BUDGET,
        MAXIMUM_PASS_MULTIPLIER
        * batch_count,
    )

    validation_interval = max(
        MINIMUM_VALIDATION_INTERVAL_UPDATES,
        batch_count,
    )

    if (
        max_updates
        % validation_interval
        != 0
    ):
        raise RuntimeError(
            "governed max_updates is not divisible by validation_interval"
        )

    positive_steps = tuple(
        range(
            validation_interval,
            max_updates + 1,
            validation_interval,
        )
    )

    validation_steps = (
        0,
        *positive_steps,
    )

    if len(
        positive_steps
    ) != POST_UPDATE_VALIDATION_EVALUATIONS:
        raise RuntimeError(
            "governed budget does not produce exactly five "
            "post-update validation evaluations"
        )

    if len(
        validation_steps
    ) != TOTAL_VALIDATION_CHECKPOINT_OBSERVATIONS:
        raise RuntimeError(
            "governed budget does not produce exactly six "
            "validation checkpoint observations"
        )

    if validation_steps[
        -1
    ] != max_updates:
        raise RuntimeError(
            "governed maximum update is not the final scheduled "
            "validation checkpoint"
        )

    if len(
        set(
            validation_steps
        )
    ) != len(
        validation_steps
    ):
        raise RuntimeError(
            "governed validation schedule contains duplicate steps"
        )

    return GMMTrainingBudget(
        class_count=int(
            class_count
        ),
        batches_per_pass=batch_count,
        max_updates=max_updates,
        validation_interval=validation_interval,
        validation_steps=validation_steps,
    )


def is_validation_step(
    step: int,
    *,
    budget: GMMTrainingBudget,
) -> bool:
    """Return whether ``step`` is a governed validation checkpoint."""

    if isinstance(
        step,
        bool,
    ):
        raise TypeError(
            "step must be an integer, not bool"
        )

    try:
        value = int(
            step
        )
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "step must be an integer"
        ) from exc

    if value != step:
        raise ValueError(
            "step must be exactly integer-valued"
        )

    if value < 0:
        raise ValueError(
            "step must be non-negative"
        )

    if value > budget.max_updates:
        raise ValueError(
            "step exceeds governed maximum update"
        )

    return value in budget.validation_steps


__all__ = [
    "GMMTrainingBudget",
    "MAXIMUM_PASS_MULTIPLIER",
    "MINIMUM_UPDATE_BUDGET",
    "MINIMUM_VALIDATION_INTERVAL_UPDATES",
    "POST_UPDATE_VALIDATION_EVALUATIONS",
    "TOTAL_VALIDATION_CHECKPOINT_OBSERVATIONS",
    "is_validation_step",
    "training_budget",
]
