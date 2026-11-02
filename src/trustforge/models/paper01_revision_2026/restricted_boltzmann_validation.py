"""Governed validation diagnostics for the Paper 1 revision-2026 RBM.

Validation semantics:

- governed empirical validation data only;
- no training class rebalance;
- no gradients or optimizer updates;
- deterministic one-step reconstruction from conditional expectations;
- reconstruction MSE is diagnostic only;
- no exact likelihood or exact NLL claim;
- validation failures fail closed.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import tensorflow as tf

from trustforge.models.paper01_revision_2026.restricted_boltzmann_model import (
    STATE_DTYPE,
    VISIBLE_DIM,
    SharedConditionalRBM,
    canonical_images_to_vectors,
    validate_class_ids,
)
from trustforge.models.paper01_revision_2026.restricted_boltzmann_observation import (
    validate_visible_values,
)


VISIBLE_TOLERANCE = 1.0e-12


@dataclass(frozen=True)
class RBMValidationDiagnostics:
    """One governed deterministic RBM validation observation."""

    reconstruction_mse: float
    example_count: int
    visible_min: float
    visible_max: float
    reconstructed_min: float
    reconstructed_max: float
    hidden_probability_min: float
    hidden_probability_max: float


def _validate_model_state(
    model: SharedConditionalRBM,
) -> None:
    """Fail closed unless all authoritative model state is finite float64."""

    if not isinstance(
        model,
        SharedConditionalRBM,
    ):
        raise TypeError(
            "model must be a SharedConditionalRBM"
        )

    for name, variable in (
        ("W", model.W),
        ("a", model.a),
        ("b", model.b),
        ("U", model.U),
    ):
        tf.debugging.assert_type(
            variable,
            STATE_DTYPE,
            message=f"RBM validation state {name} must be float64",
        )

        tf.debugging.assert_all_finite(
            variable,
            f"RBM validation state {name} contains non-finite values",
        )


def validate_reconstruction_batch(
    *,
    model: SharedConditionalRBM,
    visible_batch: np.ndarray | tf.Tensor,
    class_ids: np.ndarray | tf.Tensor,
) -> RBMValidationDiagnostics:
    """Compute deterministic one-step reconstruction diagnostics.

    The diagnostic path is:

    visible input
        -> hidden conditional probabilities
        -> visible natural parameters using hidden expectations
        -> analytic bounded-exponential visible mean
        -> reconstruction MSE

    No stochastic sampling, gradient update, or optimizer mutation occurs.
    """

    _validate_model_state(
        model
    )

    visible = validate_visible_values(
        visible_batch
    )

    tf.debugging.assert_rank(
        visible,
        2,
        message="RBM validation visible batch must be rank two",
    )

    tf.debugging.assert_equal(
        tf.shape(visible)[1],
        VISIBLE_DIM,
        message="RBM validation visible width must equal 4800",
    )

    tf.debugging.assert_greater_equal(
        tf.shape(visible)[0],
        1,
        message="RBM validation batch must not be empty",
    )

    labels = validate_class_ids(
        class_ids,
        class_count=model.class_count,
    )

    tf.debugging.assert_equal(
        tf.shape(labels)[0],
        tf.shape(visible)[0],
        message="RBM validation class batch must match visible batch",
    )

    hidden_probabilities = model.hidden_probabilities(
        visible,
        labels,
    )

    tf.debugging.assert_all_finite(
        hidden_probabilities,
        "RBM validation hidden probabilities are non-finite",
    )

    visible_natural_parameters = model.visible_natural_parameter(
        hidden_probabilities
    )

    tf.debugging.assert_all_finite(
        visible_natural_parameters,
        "RBM validation visible natural parameters are non-finite",
    )

    reconstructed_visible = model.visible_mean_given_hidden(
        hidden_probabilities
    )

    tf.debugging.assert_all_finite(
        reconstructed_visible,
        "RBM validation reconstructed visible means are non-finite",
    )

    tolerance = tf.constant(
        VISIBLE_TOLERANCE,
        dtype=STATE_DTYPE,
    )

    tf.debugging.assert_greater_equal(
        reconstructed_visible,
        -tolerance,
        message="RBM validation reconstructed mean lies below zero",
    )

    tf.debugging.assert_less_equal(
        reconstructed_visible,
        tf.constant(
            1.0,
            dtype=STATE_DTYPE,
        )
        + tolerance,
        message="RBM validation reconstructed mean lies above one",
    )

    squared_error = tf.math.squared_difference(
        reconstructed_visible,
        visible,
    )

    reconstruction_mse = tf.reduce_mean(
        squared_error
    )

    tf.debugging.assert_all_finite(
        reconstruction_mse,
        "RBM validation reconstruction MSE is non-finite",
    )

    mse_value = float(
        reconstruction_mse.numpy()
    )

    if not math.isfinite(
        mse_value
    ):
        raise ValueError(
            "RBM validation reconstruction MSE is non-finite"
        )

    visible_np = np.asarray(
        visible.numpy(),
        dtype=np.float64,
    )

    reconstructed_np = np.asarray(
        reconstructed_visible.numpy(),
        dtype=np.float64,
    )

    hidden_np = np.asarray(
        hidden_probabilities.numpy(),
        dtype=np.float64,
    )

    return RBMValidationDiagnostics(
        reconstruction_mse=mse_value,
        example_count=int(
            visible_np.shape[0]
        ),
        visible_min=float(
            visible_np.min()
        ),
        visible_max=float(
            visible_np.max()
        ),
        reconstructed_min=float(
            reconstructed_np.min()
        ),
        reconstructed_max=float(
            reconstructed_np.max()
        ),
        hidden_probability_min=float(
            hidden_np.min()
        ),
        hidden_probability_max=float(
            hidden_np.max()
        ),
    )


def evaluate_full_validation_split(
    *,
    model: SharedConditionalRBM,
    adapter,
    batch_size: int = 256,
) -> RBMValidationDiagnostics:
    """Evaluate the complete governed empirical validation split.

    Validation is streamed batchwise from split="validation". Every logical
    validation position is visited exactly once, in ascending split-global
    order. Adapter labels must be exact float32 one-hot rows.

    Reconstruction MSE is accumulated as a float64 sum of per-element squared
    error and divided exactly once by the total visible scalar count.
    """

    if isinstance(batch_size, bool):
        raise TypeError(
            "validation batch_size must be an integer, not bool"
        )

    try:
        governed_batch_size = int(batch_size)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "validation batch_size must be an integer"
        ) from exc

    if governed_batch_size != batch_size:
        raise ValueError(
            "validation batch_size must be exactly integer-valued"
        )

    if governed_batch_size < 1:
        raise ValueError(
            "validation batch_size must be positive"
        )

    if not hasattr(adapter, "split_size"):
        raise TypeError(
            "adapter must provide split_size(split)"
        )

    if not hasattr(adapter, "canonical_batch"):
        raise TypeError(
            "adapter must provide canonical_batch(split, logical_positions)"
        )

    validation_count = int(
        adapter.split_size(
            "validation"
        )
    )

    if validation_count < 1:
        raise ValueError(
            "validation split must contain at least one example"
        )

    total_squared_error = np.float64(0.0)
    total_visible_scalars = 0
    total_examples = 0

    visible_min = np.inf
    visible_max = -np.inf
    reconstructed_min = np.inf
    reconstructed_max = -np.inf
    hidden_probability_min = np.inf
    hidden_probability_max = -np.inf

    for start in range(
        0,
        validation_count,
        governed_batch_size,
    ):
        stop = min(
            start + governed_batch_size,
            validation_count,
        )

        logical_positions = np.arange(
            start,
            stop,
            dtype=np.int64,
        )

        images, one_hot_labels = adapter.canonical_batch(
            "validation",
            logical_positions,
        )

        images_array = np.asarray(
            images
        )
        labels_array = np.asarray(
            one_hot_labels
        )

        expected_batch_size = stop - start

        if images_array.shape != (
            expected_batch_size,
            40,
            40,
            3,
        ):
            raise ValueError(
                "validation canonical images must have shape (B,40,40,3)"
            )

        if images_array.dtype != np.float32:
            raise TypeError(
                "validation canonical images must be float32"
            )

        if not np.isfinite(
            images_array
        ).all():
            raise ValueError(
                "validation canonical images contain non-finite values"
            )

        if np.any(
            images_array < 0.0
        ) or np.any(
            images_array > 1.0
        ):
            raise ValueError(
                "validation canonical images are outside [0,1]"
            )

        if labels_array.ndim != 2:
            raise ValueError(
                "validation labels must be rank-2 one-hot arrays"
            )

        if labels_array.shape[0] != expected_batch_size:
            raise ValueError(
                "validation label count does not match image count"
            )

        if labels_array.shape[1] != model.class_count:
            raise ValueError(
                "validation one-hot width does not match model class count"
            )

        if labels_array.dtype != np.float32:
            raise TypeError(
                "validation one-hot labels must be float32"
            )

        if not np.isfinite(
            labels_array
        ).all():
            raise ValueError(
                "validation one-hot labels contain non-finite values"
            )

        if not np.all(
            np.logical_or(
                labels_array == 0.0,
                labels_array == 1.0,
            )
        ):
            raise ValueError(
                "validation labels must contain exact 0/1 one-hot values"
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
                "validation labels must be exact one-hot rows"
            )

        class_ids = np.argmax(
            labels_array,
            axis=1,
        ).astype(
            np.int64,
            copy=False,
        )

        visible = canonical_images_to_vectors(
            images_array
        )

        diagnostics = validate_reconstruction_batch(
            model=model,
            visible_batch=visible,
            class_ids=class_ids,
        )

        hidden_probabilities = model.hidden_probabilities(
            visible,
            class_ids,
        )

        reconstructed = model.visible_mean_given_hidden(
            hidden_probabilities
        )

        squared_error = np.square(
            np.asarray(
                reconstructed.numpy(),
                dtype=np.float64,
            )
            - np.asarray(
                visible.numpy(),
                dtype=np.float64,
            )
        )

        batch_squared_error = np.sum(
            squared_error,
            dtype=np.float64,
        )

        if not np.isfinite(
            batch_squared_error
        ):
            raise RuntimeError(
                "full validation squared-error sum is non-finite"
            )

        total_squared_error += batch_squared_error
        total_visible_scalars += int(
            squared_error.size
        )
        total_examples += diagnostics.example_count

        visible_min = min(
            visible_min,
            diagnostics.visible_min,
        )
        visible_max = max(
            visible_max,
            diagnostics.visible_max,
        )
        reconstructed_min = min(
            reconstructed_min,
            diagnostics.reconstructed_min,
        )
        reconstructed_max = max(
            reconstructed_max,
            diagnostics.reconstructed_max,
        )
        hidden_probability_min = min(
            hidden_probability_min,
            diagnostics.hidden_probability_min,
        )
        hidden_probability_max = max(
            hidden_probability_max,
            diagnostics.hidden_probability_max,
        )

    if total_examples != validation_count:
        raise RuntimeError(
            "full validation did not visit the complete validation split"
        )

    if total_visible_scalars != validation_count * VISIBLE_DIM:
        raise RuntimeError(
            "full validation visible-scalar census is inconsistent"
        )

    reconstruction_mse = float(
        total_squared_error
        / np.float64(
            total_visible_scalars
        )
    )

    if not math.isfinite(
        reconstruction_mse
    ):
        raise RuntimeError(
            "full validation reconstruction MSE is non-finite"
        )

    return RBMValidationDiagnostics(
        reconstruction_mse=reconstruction_mse,
        example_count=total_examples,
        visible_min=float(visible_min),
        visible_max=float(visible_max),
        reconstructed_min=float(reconstructed_min),
        reconstructed_max=float(reconstructed_max),
        hidden_probability_min=float(hidden_probability_min),
        hidden_probability_max=float(hidden_probability_max),
    )


__all__ = [
    "VISIBLE_TOLERANCE",
    "RBMValidationDiagnostics",
    "evaluate_full_validation_split",
    "validate_reconstruction_batch",
]
