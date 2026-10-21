"""Training primitive for the Paper 1 revision-2026 conditional MAF.

This module implements only the governed conditional negative-log-likelihood
training step.

It does not select the final optimizer, learning rate, batch size, epoch count,
or experiment schedule.

Canonical images are consumed directly in float32 [0,1].

No endpoint epsilon, universal dequantization, post-hoc clipping, or historical
MAF preprocessing is used.
"""

from __future__ import annotations

from typing import Dict

import tensorflow as tf

from trustforge.models.paper01_revision_2026.masked_autoflow_model import (
    ConditionalBlockMAF,
)


def conditional_nll(
    model: ConditionalBlockMAF,
    images: tf.Tensor,
    labels: tf.Tensor,
    *,
    training: bool,
) -> tf.Tensor:
    """Return mean conditional negative log likelihood."""
    images = tf.convert_to_tensor(
        images,
        dtype=tf.float32,
    )

    labels = tf.convert_to_tensor(
        labels,
        dtype=tf.int32,
    )

    log_prob = model.log_prob(
        images,
        labels,
        training=training,
    )

    tf.debugging.assert_all_finite(
        log_prob,
        "non-finite MAF conditional log probability",
    )

    loss = -tf.reduce_mean(
        log_prob
    )

    tf.debugging.assert_all_finite(
        loss,
        "non-finite MAF conditional NLL",
    )

    return loss


def train_step(
    model: ConditionalBlockMAF,
    optimizer: tf.keras.optimizers.Optimizer,
    images: tf.Tensor,
    labels: tf.Tensor,
) -> Dict[str, tf.Tensor]:
    """Execute one fail-closed conditional-NLL optimizer step."""
    images = tf.convert_to_tensor(
        images,
        dtype=tf.float32,
    )

    labels = tf.convert_to_tensor(
        labels,
        dtype=tf.int32,
    )

    tf.debugging.assert_rank(
        images,
        4,
        message="images must have shape (B,40,40,3)",
    )

    tf.debugging.assert_greater_equal(
        images,
        tf.constant(
            0.0,
            dtype=tf.float32,
        ),
        message="training input below canonical lower bound",
    )

    tf.debugging.assert_less_equal(
        images,
        tf.constant(
            1.0,
            dtype=tf.float32,
        ),
        message="training input above canonical upper bound",
    )

    with tf.GradientTape() as tape:
        loss = conditional_nll(
            model,
            images,
            labels,
            training=True,
        )

    trainable_variables = (
        model.trainable_variables
    )

    gradients = tape.gradient(
        loss,
        trainable_variables,
    )

    if not trainable_variables:
        raise RuntimeError(
            "MAF model has no trainable variables"
        )

    if any(
        gradient is None
        for gradient in gradients
    ):
        missing = [
            variable.name
            for variable, gradient in zip(
                trainable_variables,
                gradients,
            )
            if gradient is None
        ]

        raise RuntimeError(
            "missing gradients for trainable variables: "
            + ", ".join(missing)
        )

    for gradient in gradients:
        tf.debugging.assert_all_finite(
            gradient,
            "non-finite MAF training gradient",
        )

    global_gradient_norm = (
        tf.linalg.global_norm(
            gradients
        )
    )

    tf.debugging.assert_all_finite(
        global_gradient_norm,
        "non-finite MAF global gradient norm",
    )

    optimizer.apply_gradients(
        zip(
            gradients,
            trainable_variables,
        )
    )

    return {
        "loss": loss,
        "global_gradient_norm": (
            global_gradient_norm
        ),
        "trainable_variable_count": (
            tf.constant(
                len(trainable_variables),
                dtype=tf.int32,
            )
        ),
    }


__all__ = [
    "conditional_nll",
    "train_step",
]
