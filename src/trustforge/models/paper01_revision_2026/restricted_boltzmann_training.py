"""Governed in-memory training orchestration for the Paper 1 revision-2026 RBM.

This module composes, but does not redefine:

- full-training initialization census;
- governed class-balanced traversal;
- governed CD-1 statistics;
- governed manual SGD;
- governed fixed training budget;
- governed full empirical validation.

Checkpoint persistence is intentionally outside this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

import numpy as np

from trustforge.models.paper01_revision_2026.restricted_boltzmann_cd import (
    cd1_update_statistics,
)
from trustforge.models.paper01_revision_2026.restricted_boltzmann_initialization import (
    RBMInitializationDiagnostics,
    full_training_scalar_mean,
    initialize_shared_conditional_rbm,
)
from trustforge.models.paper01_revision_2026.restricted_boltzmann_model import (
    SharedConditionalRBM,
    canonical_images_to_vectors,
)
from trustforge.models.paper01_revision_2026.restricted_boltzmann_optimizer import (
    apply_manual_sgd_update,
)
from trustforge.models.paper01_revision_2026.restricted_boltzmann_training_budget import (
    BATCH_SIZE,
    RBMTrainingBudget,
    build_training_budget,
)
from trustforge.models.paper01_revision_2026.restricted_boltzmann_traversal import (
    RBMTraversalSpec,
    build_traversal_spec,
    training_batch,
)
from trustforge.models.paper01_revision_2026.restricted_boltzmann_validation import (
    RBMValidationDiagnostics,
    evaluate_full_validation_split,
)


class CanonicalTrainingAdapter(Protocol):
    """Minimum canonical adapter interface required by RBM training."""

    def split_size(
        self,
        split: str,
    ) -> int:
        ...

    def canonical_batch(
        self,
        split: str,
        logical_positions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        ...


@dataclass(frozen=True)
class RBMValidationObservation:
    """One governed validation observation at an optimizer-step boundary."""

    step: int
    diagnostics: RBMValidationDiagnostics


@dataclass(frozen=True)
class RBMTrainingResult:
    """Complete in-memory result of one governed dataset/seed RBM run."""

    model: SharedConditionalRBM
    initialization_diagnostics: RBMInitializationDiagnostics
    traversal_spec: RBMTraversalSpec
    budget: RBMTrainingBudget
    validation_history: tuple[RBMValidationObservation, ...]
    optimizer_steps_completed: int


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

    governed = dataset_id.strip()

    if not governed:
        raise ValueError(
            "dataset_id must not be empty"
        )

    return governed


def _validate_class_universe(
    spec: RBMTraversalSpec,
) -> int:
    """Require the canonical contiguous class universe 0..K-1."""

    class_ids = np.asarray(
        spec.class_ids,
        dtype=np.int64,
    )

    expected = np.arange(
        class_ids.size,
        dtype=np.int64,
    )

    if not np.array_equal(
        class_ids,
        expected,
    ):
        raise ValueError(
            "RBM training class ids must be contiguous and zero-based"
        )

    return int(
        class_ids.size
    )


def _fetch_training_batch(
    *,
    adapter: CanonicalTrainingAdapter,
    logical_positions: np.ndarray,
    expected_class_ids: np.ndarray,
    class_count: int,
):
    """Fetch and validate one governed canonical training batch."""

    positions = np.asarray(
        logical_positions,
        dtype=np.int64,
    )

    expected_labels = np.asarray(
        expected_class_ids,
        dtype=np.int64,
    )

    images, one_hot_labels = adapter.canonical_batch(
        "train",
        positions,
    )

    images_array = np.asarray(
        images
    )
    labels_array = np.asarray(
        one_hot_labels
    )

    batch_size = int(
        positions.size
    )

    if images_array.shape != (
        batch_size,
        40,
        40,
        3,
    ):
        raise ValueError(
            "RBM training canonical images must have shape (B,40,40,3)"
        )

    if images_array.dtype != np.float32:
        raise TypeError(
            "RBM training canonical images must be float32"
        )

    if not np.isfinite(
        images_array
    ).all():
        raise ValueError(
            "RBM training canonical images contain non-finite values"
        )

    if np.any(
        images_array < 0.0
    ) or np.any(
        images_array > 1.0
    ):
        raise ValueError(
            "RBM training canonical images are outside [0,1]"
        )

    if labels_array.shape != (
        batch_size,
        class_count,
    ):
        raise ValueError(
            "RBM training labels must have shape (B,class_count)"
        )

    if labels_array.dtype != np.float32:
        raise TypeError(
            "RBM training one-hot labels must be float32"
        )

    if not np.isfinite(
        labels_array
    ).all():
        raise ValueError(
            "RBM training one-hot labels contain non-finite values"
        )

    if not np.all(
        np.logical_or(
            labels_array == 0.0,
            labels_array == 1.0,
        )
    ):
        raise ValueError(
            "RBM training labels must contain exact 0/1 one-hot values"
        )

    if not np.all(
        np.sum(
            labels_array,
            axis=1,
            dtype=np.float64,
        )
        == 1.0
    ):
        raise ValueError(
            "RBM training labels must be exact one-hot rows"
        )

    observed_class_ids = np.argmax(
        labels_array,
        axis=1,
    ).astype(
        np.int64,
        copy=False,
    )

    if not np.array_equal(
        observed_class_ids,
        expected_labels,
    ):
        raise ValueError(
            "RBM traversal class ids do not match adapter training labels"
        )

    visible = canonical_images_to_vectors(
        images_array
    )

    return visible


def train_shared_conditional_rbm(
    *,
    adapter: CanonicalTrainingAdapter,
    dataset_id: str,
    experimental_seed: int,
    train_class_positions: Mapping[
        int,
        np.ndarray,
    ],
) -> RBMTrainingResult:
    """Train one governed shared class-conditional RBM entirely in memory.

    Scientific identity conventions:

    - optimizer step numbers are reported as 1..max_updates;
    - CD stochastic update identity is zero-based 0..max_updates-1;
    - training traversal positions are zero-based and consecutive;
    - update index ``u`` consumes training positions
      ``u*256 .. u*256+255``.

    Validation occurs exactly at reported steps 0, B, and 2B.
    """

    governed_dataset_id = _validate_dataset_id(
        dataset_id
    )

    if not hasattr(
        adapter,
        "split_size",
    ):
        raise TypeError(
            "adapter must provide split_size(split)"
        )

    if not hasattr(
        adapter,
        "canonical_batch",
    ):
        raise TypeError(
            "adapter must provide canonical_batch(split, logical_positions)"
        )

    traversal_spec = build_traversal_spec(
        dataset_id=governed_dataset_id,
        class_positions=train_class_positions,
    )

    class_count = _validate_class_universe(
        traversal_spec
    )

    adapter_training_count = int(
        adapter.split_size(
            "train"
        )
    )

    if (
        adapter_training_count
        != traversal_spec.total_training_examples
    ):
        raise ValueError(
            "RBM class-position inventory does not cover the exact training split"
        )

    training_mean = full_training_scalar_mean(
        adapter,
        batch_size=BATCH_SIZE,
    )

    model, initialization_diagnostics = (
        initialize_shared_conditional_rbm(
            class_count=class_count,
            experimental_seed=experimental_seed,
            training_scalar_mean=training_mean,
        )
    )

    budget = build_training_budget(
        traversal_spec.total_training_examples
    )

    validation_history: list[
        RBMValidationObservation
    ] = []

    initial_validation = evaluate_full_validation_split(
        model=model,
        adapter=adapter,
        batch_size=BATCH_SIZE,
    )

    validation_history.append(
        RBMValidationObservation(
            step=0,
            diagnostics=initial_validation,
        )
    )

    completed_steps = 0

    for update_index in range(
        budget.max_updates
    ):
        traversal = training_batch(
            spec=traversal_spec,
            class_positions=train_class_positions,
            experimental_seed=experimental_seed,
            start_training_position=(
                update_index
                * BATCH_SIZE
            ),
            batch_size=BATCH_SIZE,
        )

        visible = _fetch_training_batch(
            adapter=adapter,
            logical_positions=traversal.logical_positions,
            expected_class_ids=traversal.class_ids,
            class_count=class_count,
        )

        cd_result = cd1_update_statistics(
            model=model,
            visible_batch=visible,
            class_ids=traversal.class_ids,
            dataset_id=governed_dataset_id,
            experimental_seed=experimental_seed,
            update_index=update_index,
        )

        apply_manual_sgd_update(
            model=model,
            cd_result=cd_result,
        )

        completed_steps = (
            update_index + 1
        )

        if completed_steps in budget.validation_steps[
            1:
        ]:
            diagnostics = evaluate_full_validation_split(
                model=model,
                adapter=adapter,
                batch_size=BATCH_SIZE,
            )

            validation_history.append(
                RBMValidationObservation(
                    step=completed_steps,
                    diagnostics=diagnostics,
                )
            )

    if completed_steps != budget.max_updates:
        raise RuntimeError(
            "RBM trainer did not execute the governed maximum update budget"
        )

    observed_validation_steps = tuple(
        item.step
        for item in validation_history
    )

    if (
        observed_validation_steps
        != budget.validation_steps
    ):
        raise RuntimeError(
            "RBM trainer did not execute the exact governed validation schedule"
        )

    if len(
        validation_history
    ) != 3:
        raise RuntimeError(
            "RBM trainer must produce exactly three validation observations"
        )

    if (
        validation_history[-1].step
        != budget.authoritative_step
    ):
        raise RuntimeError(
            "RBM final validation does not correspond to authoritative update"
        )

    return RBMTrainingResult(
        model=model,
        initialization_diagnostics=initialization_diagnostics,
        traversal_spec=traversal_spec,
        budget=budget,
        validation_history=tuple(
            validation_history
        ),
        optimizer_steps_completed=completed_steps,
    )


__all__ = [
    "CanonicalTrainingAdapter",
    "RBMTrainingResult",
    "RBMValidationObservation",
    "train_shared_conditional_rbm",
]
