"""Governed optimizer primitive for the Paper 1 revision-2026 GMM.

The selected optimizer is exact momentum-free SGD with learning rate 1e-3.

Each update:

1. evaluates the governed censored-mixture NLL;
2. requires one finite gradient for every governed trainable parameter;
3. forms candidate float64 SGD parameter values without mutating the model;
4. validates the complete candidate state;
5. commits the complete candidate state only after validation succeeds.

There is no gradient clipping, latent-mean projection, optimizer substitution,
learning-rate substitution, NaN replacement, or component deletion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from trustforge.models.paper01_revision_2026.gaussian_mixture_model import (
    CensoredDiagonalGaussianMixture,
)
from trustforge.models.paper01_revision_2026.gaussian_mixture_observation import (
    DTYPE,
    transformed_scale,
)


LEARNING_RATE = 0.001
SGD_MOMENTUM = 0.0
GRADIENT_CLIPPING = False
MAX_TRAIN_BATCH_SIZE = 128


@dataclass(frozen=True)
class GMMOptimizerStepResult:
    """Detached diagnostics for one successfully committed SGD update."""

    loss: float
    gradient_global_norm: float


def _validate_training_batch(
    images: tf.Tensor,
) -> tf.Tensor:
    """Validate only optimizer-specific batch constraints.

    Full canonical shape, dtype, range, and finiteness validation remains
    authoritative in the model's ``nll`` path.
    """

    tensor = tf.convert_to_tensor(
        images
    )

    tf.debugging.assert_rank(
        tensor,
        4,
        message="GMM training batch must be rank 4",
    )

    batch_size = int(
        tf.shape(
            tensor
        )[0].numpy()
    )

    if batch_size < 1:
        raise ValueError(
            "GMM training batch must contain at least one example"
        )

    if batch_size > MAX_TRAIN_BATCH_SIZE:
        raise ValueError(
            "GMM training batch exceeds governed maximum size 128"
        )

    return tensor


def _validate_gradients(
    gradients: tuple[tf.Tensor | None, ...],
    parameters: tuple[tf.Variable, ...],
) -> tuple[tf.Tensor, ...]:
    """Require one finite float64 gradient for every governed parameter."""

    if len(
        gradients
    ) != len(
        parameters
    ):
        raise RuntimeError(
            "GMM gradient count does not match trainable parameter count"
        )

    validated: list[tf.Tensor] = []

    for index, (
        gradient,
        parameter,
    ) in enumerate(
        zip(
            gradients,
            parameters,
        )
    ):
        if gradient is None:
            raise RuntimeError(
                f"GMM gradient {index} is missing"
            )

        dense = tf.convert_to_tensor(
            gradient
        )

        if dense.dtype != DTYPE:
            raise RuntimeError(
                f"GMM gradient {index} is not float64"
            )

        tf.debugging.assert_equal(
            tf.shape(
                dense
            ),
            tf.shape(
                parameter
            ),
            message=(
                f"GMM gradient {index} shape does not match parameter"
            ),
        )

        tf.debugging.assert_all_finite(
            dense,
            f"GMM gradient {index} contains non-finite values",
        )

        validated.append(
            dense
        )

    return tuple(
        validated
    )


def _candidate_state(
    parameters: tuple[tf.Variable, ...],
    gradients: tuple[tf.Tensor, ...],
) -> tuple[tf.Tensor, ...]:
    """Form uncommitted exact momentum-free SGD candidate parameters."""

    learning_rate = tf.constant(
        LEARNING_RATE,
        dtype=DTYPE,
    )

    candidates = tuple(
        tf.identity(
            parameter
            - learning_rate
            * gradient
        )
        for parameter, gradient in zip(
            parameters,
            gradients,
        )
    )

    for index, candidate in enumerate(
        candidates
    ):
        tf.debugging.assert_all_finite(
            candidate,
            f"GMM candidate parameter {index} contains non-finite values",
        )

    return candidates


def _validate_candidate_model_state(
    *,
    logits: tf.Tensor,
    means: tf.Tensor,
    raw_scales: tf.Tensor,
) -> None:
    """Fail closed unless the complete candidate state is numerically valid."""

    CensoredDiagonalGaussianMixture._validate_initial_state(
        logits,
        means,
        raw_scales,
    )

    scales = transformed_scale(
        raw_scales
    )

    tf.debugging.assert_all_finite(
        scales,
        "GMM candidate transformed scales contain non-finite values",
    )

    weights = tf.nn.softmax(
        logits
    )

    tf.debugging.assert_all_finite(
        weights,
        "GMM candidate mixture weights contain non-finite values",
    )

    tf.debugging.assert_positive(
        weights,
        message="GMM candidate mixture weights must remain positive",
    )

    tf.debugging.assert_near(
        tf.reduce_sum(
            weights
        ),
        tf.constant(
            1.0,
            dtype=DTYPE,
        ),
        atol=tf.constant(
            1e-12,
            dtype=DTYPE,
        ),
        rtol=tf.constant(
            0.0,
            dtype=DTYPE,
        ),
        message="GMM candidate mixture weights must sum to one",
    )


def _gradient_global_norm(
    gradients: tuple[tf.Tensor, ...],
) -> float:
    """Return detached float64 Euclidean norm without clipping."""

    squared_norm = tf.add_n(
        [
            tf.reduce_sum(
                tf.square(
                    gradient
                )
            )
            for gradient in gradients
        ]
    )

    norm = tf.sqrt(
        squared_norm
    )

    tf.debugging.assert_all_finite(
        norm,
        "GMM gradient global norm is non-finite",
    )

    return float(
        norm.numpy()
    )


def sgd_train_step(
    model: CensoredDiagonalGaussianMixture,
    images: tf.Tensor,
) -> GMMOptimizerStepResult:
    """Perform one governed momentum-free SGD update.

    The model is not mutated unless the loss, every gradient, and the entire
    candidate model state have passed validation.
    """

    if not isinstance(
        model,
        CensoredDiagonalGaussianMixture,
    ):
        raise TypeError(
            "model must be CensoredDiagonalGaussianMixture"
        )

    batch = _validate_training_batch(
        images
    )

    parameters = model.trainable_parameters

    before = tuple(
        tf.identity(
            parameter
        )
        for parameter in parameters
    )

    with tf.GradientTape() as tape:
        loss = model.nll(
            batch
        )

    tf.debugging.assert_all_finite(
        loss,
        "GMM training loss is non-finite",
    )

    gradients = tape.gradient(
        loss,
        parameters,
    )

    validated_gradients = _validate_gradients(
        tuple(
            gradients
        ),
        parameters,
    )

    candidates = _candidate_state(
        parameters,
        validated_gradients,
    )

    _validate_candidate_model_state(
        logits=candidates[0],
        means=candidates[1],
        raw_scales=candidates[2],
    )

    gradient_norm = _gradient_global_norm(
        validated_gradients
    )

    # All numerical checks have completed before the first assignment.
    try:
        for parameter, candidate in zip(
            parameters,
            candidates,
        ):
            parameter.assign(
                candidate
            )
    except Exception:
        # Preserve fail-closed state semantics even if an unexpected assignment
        # failure occurs after an earlier variable has already been assigned.
        for parameter, original in zip(
            parameters,
            before,
        ):
            parameter.assign(
                original
            )
        raise

    # Verify the committed state through the model's public governed accessors.
    model.state_arrays()

    scales = model.scales()

    tf.debugging.assert_all_finite(
        scales,
        "GMM committed transformed scales contain non-finite values",
    )

    weights = model.mixture_weights()

    tf.debugging.assert_all_finite(
        weights,
        "GMM committed mixture weights contain non-finite values",
    )

    return GMMOptimizerStepResult(
        loss=float(
            loss.numpy()
        ),
        gradient_global_norm=gradient_norm,
    )


__all__ = [
    "GRADIENT_CLIPPING",
    "GMMOptimizerStepResult",
    "LEARNING_RATE",
    "MAX_TRAIN_BATCH_SIZE",
    "SGD_MOMENTUM",
    "sgd_train_step",
]
