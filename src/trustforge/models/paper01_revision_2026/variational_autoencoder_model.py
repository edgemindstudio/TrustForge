"""Conditional VAE architecture for Paper 1 revision 2026.

Implements the proposed architecture baseline without training,
dataset loading, checkpoint publication, or production synthesis.

All images use internal float32 NHWC coordinates in [-1, 1].
Observation likelihood mathematics lives in the separate
variational_autoencoder_observation module.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Model, layers


IMAGE_SHAPE = (40, 40, 3)
LATENT_DIM = 100
ENCODER_DENSE_UNITS = 256
DECODER_FEATURE_CHANNELS = 64


def _validate_architecture(
    latent_dim: int,
    num_classes: int,
) -> tuple[int, int]:
    """Reject unsupported architecture settings explicitly."""
    if isinstance(latent_dim, bool) or not isinstance(latent_dim, int):
        raise TypeError("latent_dim must be an integer")
    if isinstance(num_classes, bool) or not isinstance(num_classes, int):
        raise TypeError("num_classes must be an integer")

    if latent_dim != LATENT_DIM:
        raise ValueError(
            f"revision-2026 VAE requires latent_dim={LATENT_DIM}"
        )
    if num_classes < 2:
        raise ValueError("num_classes must be at least two")

    return latent_dim, num_classes


def _spatial_condition(
    labels: tf.Tensor,
    num_classes: int,
) -> tf.Tensor:
    """Broadcast one-hot label vectors over the 40x40 image plane."""
    labels = layers.RepeatVector(40 * 40)(
        labels
    )
    return layers.Reshape(
        (40, 40, num_classes)
    )(labels)


def build_encoder(
    *,
    latent_dim: int = LATENT_DIM,
    num_classes: int,
) -> Model:
    """Build the class-conditional diagonal-Gaussian encoder.

    Returns posterior mean and log-variance, not an implicit
    random latent draw.
    """
    latent_dim, num_classes = _validate_architecture(
        latent_dim, num_classes
    )

    image = layers.Input(
        shape=IMAGE_SHAPE,
        dtype=tf.float32,
        name="internal_image",
    )
    condition = layers.Input(
        shape=(num_classes,),
        dtype=tf.float32,
        name="class_condition",
    )

    label_map = _spatial_condition(condition, num_classes)

    x = layers.Concatenate(name="encoder_condition_concat")(
        [image, label_map]
    )

    x = layers.Conv2D(
        32, 3, strides=2, padding="same",
        activation="relu", name="encoder_conv1",
    )(x)
    x = layers.Conv2D(
        64, 3, strides=2, padding="same",
        activation="relu", name="encoder_conv2",
    )(x)
    x = layers.BatchNormalization(
        name="encoder_batchnorm"
    )(x)
    x = layers.Conv2D(
        128, 3, strides=1, padding="same",
        activation="relu", name="encoder_conv3",
    )(x)

    x = layers.Flatten(name="encoder_flatten")(x)
    x = layers.Dense(
        ENCODER_DENSE_UNITS,
        activation="relu",
        name="encoder_dense",
    )(x)

    posterior_mean = layers.Dense(
        latent_dim, name="posterior_mean"
    )(x)
    posterior_log_variance = layers.Dense(
        latent_dim, name="posterior_log_variance"
    )(x)

    return Model(
        [image, condition],
        [posterior_mean, posterior_log_variance],
        name="revision2026_conditional_vae_encoder",
    )


def build_decoder(
    *,
    latent_dim: int = LATENT_DIM,
    num_classes: int,
) -> Model:
    """Build the class-conditional, tanh-bounded decoder."""
    latent_dim, num_classes = _validate_architecture(
        latent_dim, num_classes
    )

    latent = layers.Input(
        shape=(latent_dim,),
        dtype=tf.float32,
        name="latent",
    )
    condition = layers.Input(
        shape=(num_classes,),
        dtype=tf.float32,
        name="class_condition",
    )

    x = layers.Concatenate(
        name="decoder_condition_concat"
    )([latent, condition])

    x = layers.Dense(
        10 * 10 * DECODER_FEATURE_CHANNELS,
        activation="relu",
        name="decoder_dense",
    )(x)
    x = layers.Reshape(
        (10, 10, DECODER_FEATURE_CHANNELS),
        name="decoder_reshape",
    )(x)
    x = layers.Conv2DTranspose(
        64, 3, strides=2, padding="same",
        activation="relu", name="decoder_deconv1",
    )(x)
    x = layers.Conv2DTranspose(
        32, 3, strides=2, padding="same",
        activation="relu", name="decoder_deconv2",
    )(x)
    output = layers.Conv2D(
        3, 3, padding="same",
        activation="tanh",
        name="decoder_mean",
    )(x)

    return Model(
        [latent, condition],
        output,
        name="revision2026_conditional_vae_decoder",
    )


def reparameterize(
    posterior_mean: tf.Tensor,
    posterior_log_variance: tf.Tensor,
    *,
    seed: tf.Tensor,
) -> tf.Tensor:
    """Draw posterior latents using an explicit stateless RNG seed.

    The seed must have shape (2,) and integer dtype.
    No posterior clipping or implicit random fallback occurs.
    """
    mean = tf.convert_to_tensor(posterior_mean)
    logvar = tf.convert_to_tensor(posterior_log_variance)
    seed = tf.convert_to_tensor(seed)

    if mean.dtype != tf.float32 or logvar.dtype != tf.float32:
        raise TypeError("posterior parameters must be float32")
    if seed.dtype not in (tf.int32, tf.int64):
        raise TypeError("stateless seed must be int32 or int64")

    tf.debugging.assert_rank(mean, 2)
    tf.debugging.assert_rank(logvar, 2)
    tf.debugging.assert_equal(tf.shape(mean), tf.shape(logvar))
    tf.debugging.assert_positive(tf.shape(mean)[0])
    tf.debugging.assert_equal(tf.shape(mean)[1], LATENT_DIM)

    tf.debugging.assert_equal(
        tf.shape(seed),
        tf.constant([2], dtype=tf.int32),
        message="stateless seed must have shape (2,)",
    )
    tf.debugging.assert_all_finite(
        mean, "posterior mean contains non-finite values"
    )
    tf.debugging.assert_all_finite(
        logvar, "posterior log variance contains non-finite values"
    )

    scale = tf.exp(0.5 * logvar)
    tf.debugging.assert_all_finite(
        scale, "posterior standard deviation is non-finite"
    )

    epsilon = tf.random.stateless_normal(
        shape=tf.shape(mean),
        seed=seed,
        dtype=tf.float32,
    )
    latent = mean + scale * epsilon

    tf.debugging.assert_all_finite(
        latent, "reparameterized latent is non-finite"
    )
    return latent


def build_models(
    *,
    latent_dim: int = LATENT_DIM,
    num_classes: int,
) -> tuple[Model, Model]:
    """Build the governed encoder and decoder without side effects."""
    _validate_architecture(latent_dim, num_classes)

    encoder = build_encoder(
        latent_dim=latent_dim,
        num_classes=num_classes,
    )
    decoder = build_decoder(
        latent_dim=latent_dim,
        num_classes=num_classes,
    )
    return encoder, decoder


__all__ = [
    "IMAGE_SHAPE",
    "LATENT_DIM",
    "ENCODER_DENSE_UNITS",
    "DECODER_FEATURE_CHANNELS",
    "build_encoder",
    "build_decoder",
    "build_models",
    "reparameterize",
]
