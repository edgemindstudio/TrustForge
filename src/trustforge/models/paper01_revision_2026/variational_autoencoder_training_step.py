"""Governed single-step conditional VAE optimization.

Paper 1 revision 2026. Eager TensorFlow execution only.

This module does not load datasets, create checkpoints, select
training budgets, or authorize real-dataset training.
"""

from __future__ import annotations

import math

import tensorflow as tf

from trustforge.models.paper01_revision_2026 import (
    variational_autoencoder_model as model_module,
)
from trustforge.models.paper01_revision_2026 import (
    variational_autoencoder_observation as observation,
)


def _require_one_hot(labels, *, batch_size, num_classes):
    """Validate class conditioning without implicit conversion."""
    labels = tf.convert_to_tensor(labels)

    if labels.dtype != tf.float32:
        raise TypeError("class labels must be float32")

    if labels.shape.rank != 2:
        raise ValueError("class labels must have rank two")

    tf.debugging.assert_equal(
        tf.shape(labels),
        tf.stack([batch_size, num_classes]),
        message="class label shape mismatch",
    )
    tf.debugging.assert_all_finite(
        labels, "nonfinite class labels"
    )

    valid_entries = tf.logical_or(
        tf.equal(labels, 0.0),
        tf.equal(labels, 1.0),
    )
    tf.debugging.assert_equal(
        tf.reduce_all(valid_entries),
        True,
        message="class labels must be binary one-hot",
    )
    tf.debugging.assert_equal(
        tf.reduce_all(
            tf.equal(tf.reduce_sum(labels, axis=1), 1.0)
        ),
        True,
        message="each row must contain exactly one class",
    )
    return labels


def vae_training_step(
    *,
    encoder,
    decoder,
    optimizer,
    canonical_images,
    class_labels,
    seed,
):
    """Perform one explicitly seeded VAE optimization step.

    Returns scalar total, reconstruction, and KL losses.

    The governed objective is Gaussian NLL plus KL with
    observation variance 1.0 and KL coefficient 1.0.
    No KL annealing, posterior clipping, or gradient
    clipping is performed.

    Input validation happens before model execution.
    All loss and gradient checks happen before the
    optimizer update.

    Training-mode BatchNormalization state may change
    during the forward pass, even if a later check fails.
    Therefore this function is not transactional.
    A higher-level runner must reject interrupted runs
    and prohibit publication of partial checkpoints.
    """
    if not tf.executing_eagerly():
        raise RuntimeError(
            "vae_training_step requires eager execution"
        )

    if not isinstance(encoder, tf.keras.Model):
        raise TypeError("encoder must be a Keras model")
    if not isinstance(decoder, tf.keras.Model):
        raise TypeError("decoder must be a Keras model")
    if not isinstance(optimizer, tf.keras.optimizers.Optimizer):
        raise TypeError("optimizer must be a Keras optimizer")

    if len(encoder.inputs) != 2 or len(decoder.inputs) != 2:
        raise ValueError("conditional models require two inputs")

    if len(encoder.outputs) != 2:
        raise ValueError("encoder must output mean and log variance")

    if tuple(encoder.input_shape[1][1:]) != (
        int(encoder.input_shape[1][-1]),
    ):
        raise ValueError("invalid encoder class input")

    num_classes = int(encoder.input_shape[1][-1])

    if num_classes < 2:
        raise ValueError("invalid class count")

    if int(decoder.input_shape[1][-1]) != num_classes:
        raise ValueError("encoder/decoder class mismatch")

    if tuple(encoder.input_shape[0][1:]) != (
        model_module.IMAGE_SHAPE
    ):
        raise ValueError("invalid encoder image shape")

    if tuple(decoder.output_shape[1:]) != (
        model_module.IMAGE_SHAPE
    ):
        raise ValueError("invalid decoder image shape")

    if int(decoder.input_shape[0][-1]) != (
        model_module.LATENT_DIM
    ):
        raise ValueError("invalid decoder latent dimension")

    if any(
        int(shape[-1]) != model_module.LATENT_DIM
        for shape in encoder.output_shape
    ):
        raise ValueError("invalid encoder latent dimension")

    images = tf.convert_to_tensor(canonical_images)

    if images.dtype != tf.float32:
        raise TypeError("canonical images must be float32")

    # The existing observation module enforces shape,
    # canonical range, nonempty batch, and finiteness.
    internal_images = observation.canonical_to_internal(images)

    labels = _require_one_hot(
        class_labels,
        batch_size=tf.shape(internal_images)[0],
        num_classes=tf.constant(num_classes, dtype=tf.int32),
    )

    # Check random-seed validity before entering the
    # stateful training-mode forward pass.
    seed = tf.convert_to_tensor(seed)
    if seed.dtype not in (tf.int32, tf.int64):
        raise TypeError("seed must be integer typed")
    tf.debugging.assert_equal(
        tf.shape(seed),
        tf.constant([2], dtype=tf.int32),
        message="seed must have shape (2,)",
    )

    variables = (
        list(encoder.trainable_variables)
        + list(decoder.trainable_variables)
    )

    if not variables:
        raise ValueError("VAE has no trainable variables")

    if len({id(v) for v in variables}) != len(variables):
        raise ValueError("encoder/decoder share trainable variables")

    with tf.GradientTape() as tape:
        posterior_mean, posterior_log_variance = encoder(
            [internal_images, labels],
            training=True,
        )

        z = model_module.reparameterize(
            posterior_mean,
            posterior_log_variance,
            seed=seed,
        )

        decoder_means = decoder(
            [z, labels],
            training=True,
        )

        tf.debugging.assert_all_finite(
            decoder_means, "nonfinite decoder means"
        )

        total, reconstruction, kl = observation.vae_objective(
            internal_images,
            decoder_means,
            posterior_mean,
            posterior_log_variance,
        )

    for name, loss in (
        ("total", total),
        ("reconstruction", reconstruction),
        ("kl", kl),
    ):
        tf.debugging.assert_all_finite(
            loss, f"nonfinite VAE {name} loss"
        )

    gradients = tape.gradient(total, variables)

    if len(gradients) != len(variables):
        raise RuntimeError("VAE gradient count mismatch")

    if any(gradient is None for gradient in gradients):
        raise RuntimeError("VAE disconnected gradient")

    for gradient in gradients:
        tf.debugging.assert_all_finite(
            tf.convert_to_tensor(gradient),
            "nonfinite VAE gradient",
        )

    optimizer.apply_gradients(zip(gradients, variables))

    for variable in variables:
        tf.debugging.assert_all_finite(
            variable, "nonfinite updated VAE weight"
        )

    result = {
        "total_loss": float(total.numpy()),
        "reconstruction_loss": float(reconstruction.numpy()),
        "kl_loss": float(kl.numpy()),
    }

    if not all(math.isfinite(value) for value in result.values()):
        raise RuntimeError("nonfinite VAE loss record")

    return result


__all__ = ["vae_training_step"]
