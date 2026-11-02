"""Governed conditional VAE validation primitives.

Paper 1 revision 2026.

Batch evaluation uses inference-mode networks and explicit
stateless posterior sampling. Aggregation is sample-weighted.

Dataset traversal, validation seed scheduling, checkpoint
selection, and production authorization belong to the
higher-level governed runner.
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


def _validate_labels(labels, batch_size, num_classes):
    labels = tf.convert_to_tensor(labels)

    if labels.dtype != tf.float32:
        raise TypeError("validation labels must be float32")

    tf.debugging.assert_rank(
        labels, 2, message="validation labels must have rank two"
    )
    tf.debugging.assert_equal(
        tf.shape(labels),
        tf.stack([batch_size, num_classes]),
        message="validation label shape mismatch",
    )
    tf.debugging.assert_all_finite(
        labels, "nonfinite validation labels"
    )

    binary = tf.logical_or(
        tf.equal(labels, 0.0),
        tf.equal(labels, 1.0),
    )
    tf.debugging.assert_equal(
        tf.reduce_all(binary),
        True,
        message="validation labels must be binary one-hot",
    )
    tf.debugging.assert_equal(
        tf.reduce_all(
            tf.equal(tf.reduce_sum(labels, axis=1), 1.0)
        ),
        True,
        message="exactly one class required per example",
    )

    return labels


def vae_validation_batch(
    *,
    encoder,
    decoder,
    canonical_images,
    class_labels,
    seed,
):
    """Evaluate one batch without optimizer updates.

    A single stateless posterior draw estimates the expected
    reconstruction NLL. The exact diagonal-Gaussian KL is
    added without annealing.

    Returns detached Python scalars and the example count.
    """
    if not tf.executing_eagerly():
        raise RuntimeError(
            "vae_validation_batch requires eager execution"
        )

    if not isinstance(encoder, tf.keras.Model):
        raise TypeError("encoder must be a Keras model")
    if not isinstance(decoder, tf.keras.Model):
        raise TypeError("decoder must be a Keras model")

    if len(encoder.inputs) != 2 or len(decoder.inputs) != 2:
        raise ValueError("conditional models require two inputs")

    if len(encoder.outputs) != 2:
        raise ValueError("encoder must output mean and log variance")

    if tuple(encoder.input_shape[0][1:]) != model_module.IMAGE_SHAPE:
        raise ValueError("invalid encoder image shape")

    if tuple(decoder.output_shape[1:]) != model_module.IMAGE_SHAPE:
        raise ValueError("invalid decoder image shape")

    num_classes = int(encoder.input_shape[1][-1])

    if num_classes < 2:
        raise ValueError("invalid class count")

    if int(decoder.input_shape[1][-1]) != num_classes:
        raise ValueError("encoder/decoder class mismatch")

    if any(
        int(shape[-1]) != model_module.LATENT_DIM
        for shape in encoder.output_shape
    ):
        raise ValueError("invalid encoder latent dimension")

    if int(decoder.input_shape[0][-1]) != model_module.LATENT_DIM:
        raise ValueError("invalid decoder latent dimension")

    images = tf.convert_to_tensor(canonical_images)

    if images.dtype != tf.float32:
        raise TypeError("canonical images must be float32")

    internal_images = observation.canonical_to_internal(images)

    labels = _validate_labels(
        class_labels,
        tf.shape(internal_images)[0],
        tf.constant(num_classes, dtype=tf.int32),
    )

    seed = tf.convert_to_tensor(seed)

    if seed.dtype not in (tf.int32, tf.int64):
        raise TypeError("seed must have integer dtype")

    tf.debugging.assert_equal(
        tf.shape(seed),
        tf.constant([2], dtype=tf.int32),
        message="validation seed must have shape (2,)",
    )

    # Inference-mode calls do not update BatchNormalization
    # moving statistics. No optimizer is used.
    mean, logvar = encoder(
        [internal_images, labels],
        training=False,
    )

    latent = model_module.reparameterize(
        mean,
        logvar,
        seed=seed,
    )

    reconstructed = decoder(
        [latent, labels],
        training=False,
    )

    total, reconstruction, kl = observation.vae_objective(
        internal_images,
        reconstructed,
        mean,
        logvar,
    )

    for name, value in (
        ("total", total),
        ("reconstruction", reconstruction),
        ("kl", kl),
    ):
        tf.debugging.assert_all_finite(
            value, f"nonfinite validation {name}"
        )

    result = {
        "count": int(tf.shape(internal_images)[0].numpy()),
        "total_loss": float(total.numpy()),
        "reconstruction_loss": float(reconstruction.numpy()),
        "kl_loss": float(kl.numpy()),
    }

    if result["count"] <= 0:
        raise ValueError("validation batch must be nonempty")

    if not all(
        math.isfinite(result[key])
        for key in (
            "total_loss",
            "reconstruction_loss",
            "kl_loss",
        )
    ):
        raise ValueError("nonfinite validation result")

    return result


def aggregate_validation_batches(batch_results):
    """Combine batch means using their example counts.

    The caller is responsible for complete, nonduplicated
    validation-partition traversal and governed seed scheduling.
    """
    results = list(batch_results)

    if not results:
        raise ValueError("validation results must not be empty")

    keys = (
        "total_loss",
        "reconstruction_loss",
        "kl_loss",
    )

    total_count = 0
    weighted = {key: 0.0 for key in keys}

    for result in results:
        if not isinstance(result, dict):
            raise TypeError("each batch result must be a dictionary")

        if set(result) != {"count", *keys}:
            raise ValueError("unexpected validation result fields")

        count = result["count"]

        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError("validation count must be an integer")

        if count <= 0:
            raise ValueError("validation count must be positive")

        total_count += count

        for key in keys:
            value = result[key]

            if isinstance(value, bool) or not isinstance(
                value, (int, float)
            ):
                raise TypeError(f"invalid validation metric: {key}")

            if not math.isfinite(value):
                raise ValueError(
                    f"nonfinite validation metric: {key}"
                )

            weighted[key] += count * float(value)

            if not math.isfinite(weighted[key]):
                raise ValueError(
                    f"nonfinite weighted validation sum: {key}"
                )

    aggregated = {
        "count": total_count,
        **{
            key: weighted[key] / total_count
            for key in keys
        },
    }

    for key in keys:
        if not math.isfinite(aggregated[key]):
            raise ValueError(
                f"nonfinite aggregate validation metric: {key}"
            )

    return aggregated


__all__ = [
    "vae_validation_batch",
    "aggregate_validation_batches",
]
