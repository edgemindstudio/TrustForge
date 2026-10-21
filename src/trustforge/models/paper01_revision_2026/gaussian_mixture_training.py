"""Governed in-memory class trainer for the Paper 1 revision-2026 GMM.

This module composes the already governed:

- class-model initialization;
- balanced class-specific traversal;
- training-budget schedule;
- momentum-free SGD update primitive;
- full-class validation evaluator;
- lowest-validation-NLL checkpoint selection.

It deliberately does not persist checkpoints. Persistence remains a separate
governed layer.

The returned model is reconstructed from the selected authoritative
in-memory validation checkpoint rather than from the final optimizer state.
"""

from __future__ import annotations

from dataclasses import dataclass
import operator
from typing import Protocol

import numpy as np

from trustforge.models.paper01_revision_2026.gaussian_mixture_initialization import (
    GMMInitializationDiagnostics,
    initialization_logical_positions,
    initialize_class_model_from_canonical_sample,
)
from trustforge.models.paper01_revision_2026.gaussian_mixture_model import (
    CensoredDiagonalGaussianMixture,
)
from trustforge.models.paper01_revision_2026.gaussian_mixture_optimizer import (
    sgd_train_step,
)
from trustforge.models.paper01_revision_2026.gaussian_mixture_training_budget import (
    GMMTrainingBudget,
    is_validation_step,
    training_budget,
)
from trustforge.models.paper01_revision_2026.gaussian_mixture_traversal import (
    iter_training_batches,
)
from trustforge.models.paper01_revision_2026.gaussian_mixture_validation import (
    GMMValidationResult,
    evaluate_full_class_validation_nll,
)


class CanonicalTrainingAdapter(Protocol):
    """Minimum dataset-adapter interface required by the class trainer."""

    def canonical_batch(
        self,
        split: str,
        logical_positions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        ...


@dataclass(frozen=True)
class GMMValidationCheckpoint:
    """One governed validation checkpoint observation."""

    step: int
    validation_nll: float


@dataclass(frozen=True)
class GMMClassTrainingResult:
    """Complete in-memory result of one governed class-model training run."""

    model: CensoredDiagonalGaussianMixture
    initialization_diagnostics: GMMInitializationDiagnostics
    budget: GMMTrainingBudget
    validation_history: tuple[GMMValidationCheckpoint, ...]
    selected_step: int
    selected_validation_nll: float
    optimizer_steps_completed: int


def _validate_exact_nonnegative_integer(
    value: int,
    *,
    name: str,
) -> int:
    """Validate a governed non-negative integer identity."""

    if isinstance(
        value,
        bool,
    ):
        raise TypeError(
            f"{name} must be an integer, not bool"
        )

    try:
        converted = operator.index(
            value
        )
    except TypeError as exc:
        raise TypeError(
            f"{name} must be an exact integer"
        ) from exc

    if converted < 0:
        raise ValueError(
            f"{name} must be non-negative"
        )

    return converted


def _validate_positions(
    positions: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    """Validate one exact ordered split-global position sequence."""

    array = np.asarray(
        positions
    )

    if array.ndim != 1:
        raise ValueError(
            f"{name} must be one-dimensional"
        )

    if array.size == 0:
        raise ValueError(
            f"{name} must not be empty"
        )

    if array.dtype.kind not in {
        "i",
        "u",
    }:
        raise TypeError(
            f"{name} must contain integers"
        )

    if np.any(
        array < 0
    ):
        raise ValueError(
            f"{name} must be non-negative"
        )

    normalized = array.astype(
        np.int64,
        copy=False,
    )

    if np.unique(
        normalized
    ).size != normalized.size:
        raise ValueError(
            f"{name} contain duplicates"
        )

    return normalized


def _validate_class_labels(
    labels: np.ndarray,
    *,
    class_id: int,
    expected_count: int,
    context: str,
) -> None:
    """Fail closed unless all returned one-hot labels match the class."""

    array = np.asarray(
        labels
    )

    if array.ndim != 2:
        raise ValueError(
            f"{context} labels must be rank-2 one-hot arrays"
        )

    if array.shape[
        0
    ] != expected_count:
        raise ValueError(
            f"{context} label count does not match requested positions"
        )

    if array.dtype != np.float32:
        raise TypeError(
            f"{context} labels must be float32"
        )

    if not np.isfinite(
        array
    ).all():
        raise ValueError(
            f"{context} labels contain non-finite values"
        )

    if class_id >= array.shape[
        1
    ]:
        raise ValueError(
            f"{context} class_id is outside one-hot label width"
        )

    row_sums = np.sum(
        array,
        axis=1,
        dtype=np.float64,
    )

    if not np.array_equal(
        row_sums,
        np.ones(
            expected_count,
            dtype=np.float64,
        ),
    ):
        raise ValueError(
            f"{context} labels are not exact one-hot rows"
        )

    if not np.all(
        np.count_nonzero(
            array,
            axis=1,
        )
        == 1
    ):
        raise ValueError(
            f"{context} labels are not exact one-hot rows"
        )

    observed = np.argmax(
        array,
        axis=1,
    )

    if not np.all(
        observed
        == class_id
    ):
        raise ValueError(
            f"{context} positions include examples outside governed class"
        )


def _fetch_class_batch(
    *,
    adapter: CanonicalTrainingAdapter,
    split: str,
    positions: np.ndarray,
    class_id: int,
    context: str,
) -> np.ndarray:
    """Fetch and validate one canonical class-specific image batch."""

    images, labels = adapter.canonical_batch(
        split,
        positions,
    )

    image_array = np.asarray(
        images
    )

    if image_array.shape[
        0
    ] != positions.size:
        raise ValueError(
            f"{context} image count does not match requested positions"
        )

    if image_array.dtype != np.float32:
        raise TypeError(
            f"{context} canonical images must be float32"
        )

    _validate_class_labels(
        labels,
        class_id=class_id,
        expected_count=int(
            positions.size
        ),
        context=context,
    )

    return image_array


def _selected_model_from_state(
    state: dict[str, np.ndarray],
) -> CensoredDiagonalGaussianMixture:
    """Construct a fresh authoritative model from one detached state."""

    if set(
        state
    ) != {
        "logits",
        "means",
        "raw_scales",
    }:
        raise RuntimeError(
            "selected in-memory GMM state has unexpected keys"
        )

    return CensoredDiagonalGaussianMixture(
        logits=state[
            "logits"
        ],
        means=state[
            "means"
        ],
        raw_scales=state[
            "raw_scales"
        ],
    )


def train_class_model(
    *,
    adapter: CanonicalTrainingAdapter,
    class_id: int,
    experimental_seed: int,
    train_class_positions: np.ndarray,
    validation_class_positions: np.ndarray,
) -> GMMClassTrainingResult:
    """Train one governed class-specific GMM entirely in memory.

    Checkpoint selection uses the lowest finite full-class validation NLL.
    Exact ties retain the earliest checkpoint because the incumbent is
    replaced only by a strictly lower NLL.
    """

    governed_class_id = _validate_exact_nonnegative_integer(
        class_id,
        name="class_id",
    )

    train_positions = _validate_positions(
        train_class_positions,
        name="train_class_positions",
    )

    validation_positions = _validate_positions(
        validation_class_positions,
        name="validation_class_positions",
    )

    class_count = int(
        train_positions.size
    )

    init_relative_positions = initialization_logical_positions(
        class_count
    )

    init_split_positions = train_positions[
        init_relative_positions
    ]

    initialization_images = _fetch_class_batch(
        adapter=adapter,
        split="train",
        positions=init_split_positions,
        class_id=governed_class_id,
        context="GMM initialization",
    )

    model, initialization_diagnostics = (
        initialize_class_model_from_canonical_sample(
            initialization_images,
            class_count=class_count,
            experimental_seed=experimental_seed,
        )
    )

    budget = training_budget(
        class_count
    )

    history: list[
        GMMValidationCheckpoint
    ] = []

    initial_validation = evaluate_full_class_validation_nll(
        model,
        adapter=adapter,
        logical_positions=validation_positions,
        class_id=governed_class_id,
    )

    history.append(
        GMMValidationCheckpoint(
            step=0,
            validation_nll=initial_validation.mean_nll,
        )
    )

    best_step = 0
    best_validation_nll = initial_validation.mean_nll
    best_state = model.state_arrays()

    completed_steps = 0

    batches = iter_training_batches(
        train_positions,
        experimental_seed=experimental_seed,
        max_updates=budget.max_updates,
    )

    for step, batch_positions in enumerate(
        batches,
        start=1,
    ):
        training_images = _fetch_class_batch(
            adapter=adapter,
            split="train",
            positions=batch_positions,
            class_id=governed_class_id,
            context="GMM training",
        )

        sgd_train_step(
            model,
            training_images,
        )

        completed_steps = step

        if is_validation_step(
            step,
            budget=budget,
        ):
            validation_result: GMMValidationResult = (
                evaluate_full_class_validation_nll(
                    model,
                    adapter=adapter,
                    logical_positions=validation_positions,
                    class_id=governed_class_id,
                )
            )

            validation_nll = validation_result.mean_nll

            if not np.isfinite(
                validation_nll
            ):
                raise RuntimeError(
                    "GMM validation checkpoint NLL is non-finite"
                )

            history.append(
                GMMValidationCheckpoint(
                    step=step,
                    validation_nll=validation_nll,
                )
            )

            # Strict inequality implements the governed earliest-step tie break.
            if validation_nll < best_validation_nll:
                best_step = step
                best_validation_nll = validation_nll
                best_state = model.state_arrays()

    if completed_steps != budget.max_updates:
        raise RuntimeError(
            "GMM trainer did not execute the governed maximum update budget"
        )

    observed_steps = tuple(
        checkpoint.step
        for checkpoint in history
    )

    if observed_steps != budget.validation_steps:
        raise RuntimeError(
            "GMM validation history does not match governed schedule"
        )

    if len(
        history
    ) != len(
        budget.validation_steps
    ):
        raise RuntimeError(
            "GMM validation history has unexpected checkpoint count"
        )

    if not any(
        (
            checkpoint.step
            == best_step
            and checkpoint.validation_nll
            == best_validation_nll
        )
        for checkpoint in history
    ):
        raise RuntimeError(
            "selected GMM checkpoint is absent from validation history"
        )

    selected_model = _selected_model_from_state(
        best_state
    )

    # Validate detached authoritative state once more after reconstruction.
    selected_model.state_arrays()
    selected_model.scales()
    selected_model.mixture_weights()

    return GMMClassTrainingResult(
        model=selected_model,
        initialization_diagnostics=initialization_diagnostics,
        budget=budget,
        validation_history=tuple(
            history
        ),
        selected_step=best_step,
        selected_validation_nll=best_validation_nll,
        optimizer_steps_completed=completed_steps,
    )


__all__ = [
    "CanonicalTrainingAdapter",
    "GMMClassTrainingResult",
    "GMMValidationCheckpoint",
    "train_class_model",
]
