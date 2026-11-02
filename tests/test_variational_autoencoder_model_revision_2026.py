"""Conditional VAE architecture regression tests for Paper 1 revision 2026."""

import numpy as np
import pytest
import tensorflow as tf

from trustforge.models.paper01_revision_2026 import (
    variational_autoencoder_model as vae,
)


def make_models(num_classes=9):
    tf.keras.utils.set_random_seed(42)
    return vae.build_models(
        latent_dim=100,
        num_classes=num_classes,
    )


@pytest.mark.parametrize("num_classes", [2, 5, 9, 47])
def test_shapes_for_governed_class_counts(num_classes):
    encoder, decoder = make_models(num_classes)

    images = tf.zeros((2, 40, 40, 3), tf.float32)
    labels = tf.one_hot([0, 1], num_classes)

    mean, logvar = encoder([images, labels], training=False)

    assert tuple(mean.shape) == (2, 100)
    assert tuple(logvar.shape) == (2, 100)

    z = vae.reparameterize(
        mean, logvar,
        seed=tf.constant([42, 0], tf.int32),
    )

    decoded = decoder([z, labels], training=False)

    assert tuple(decoded.shape) == (2, 40, 40, 3)
    assert bool(tf.reduce_all(tf.math.is_finite(decoded)))
    assert bool(tf.reduce_all(decoded >= -1.0))
    assert bool(tf.reduce_all(decoded <= 1.0))


def test_fixed_seed_reproducibility():
    mean = tf.zeros((2, 100), tf.float32)
    logvar = tf.zeros_like(mean)
    seed = tf.constant([42, 0], tf.int32)

    first = vae.reparameterize(mean, logvar, seed=seed)
    second = vae.reparameterize(mean, logvar, seed=seed)

    np.testing.assert_array_equal(first.numpy(), second.numpy())


def test_changed_seed_changes_samples():
    mean = tf.zeros((2, 100), tf.float32)
    logvar = tf.zeros_like(mean)

    first = vae.reparameterize(mean, logvar, seed=[42, 0])
    second = vae.reparameterize(mean, logvar, seed=[43, 0])

    assert bool(tf.reduce_any(tf.not_equal(first, second)))


def test_eager_and_graph_samples_identical():
    mean = tf.zeros((2, 100), tf.float32)
    logvar = tf.zeros_like(mean)
    seed = tf.constant([42, 0], tf.int32)

    @tf.function(autograph=False)
    def graph_sample(mu, lv, s):
        return vae.reparameterize(mu, lv, seed=s)

    eager = vae.reparameterize(mean, logvar, seed=seed)
    graph = graph_sample(mean, logvar, seed)

    np.testing.assert_array_equal(eager.numpy(), graph.numpy())


def test_encoder_condition_changes_posterior():
    encoder, _ = make_models()

    images = tf.ones((2, 40, 40, 3), tf.float32) * 0.25
    label_a = tf.one_hot([0, 0], 9)
    label_b = tf.one_hot([1, 1], 9)

    mean_a, logvar_a = encoder(
        [images, label_a], training=False
    )
    mean_b, logvar_b = encoder(
        [images, label_b], training=False
    )

    difference = tf.reduce_sum(
        tf.abs(mean_a - mean_b)
        + tf.abs(logvar_a - logvar_b)
    )

    assert float(difference.numpy()) > 0.0


def test_decoder_condition_changes_output():
    _, decoder = make_models()

    z = tf.zeros((2, 100), tf.float32)
    label_a = tf.one_hot([0, 0], 9)
    label_b = tf.one_hot([1, 1], 9)

    output_a = decoder([z, label_a], training=False)
    output_b = decoder([z, label_b], training=False)

    assert bool(tf.reduce_any(tf.not_equal(output_a, output_b)))


def test_encoder_and_decoder_have_finite_gradients():
    encoder, decoder = make_models()

    images = tf.ones((2, 40, 40, 3), tf.float32) * 0.25
    labels = tf.one_hot([0, 1], 9)

    with tf.GradientTape() as tape:
        mean, logvar = encoder(
            [images, labels], training=True
        )
        z = vae.reparameterize(
            mean, logvar, seed=[42, 0]
        )
        reconstruction = decoder(
            [z, labels], training=True
        )
        loss = tf.reduce_mean(
            tf.square(reconstruction - images)
        ) + tf.reduce_mean(tf.square(mean))

    variables = (
        encoder.trainable_variables
        + decoder.trainable_variables
    )
    gradients = tape.gradient(loss, variables)

    assert len(gradients) == len(variables)
    assert all(g is not None for g in gradients)

    for gradient in gradients:
        tf.debugging.assert_all_finite(
            gradient, "nonfinite VAE architecture gradient"
        )


@pytest.mark.parametrize(
    "latent_dim",
    [0, 32, 128, True],
)
def test_invalid_latent_dimension_rejected(latent_dim):
    with pytest.raises((TypeError, ValueError)):
        vae.build_models(latent_dim=latent_dim, num_classes=9)


@pytest.mark.parametrize(
    "num_classes",
    [0, 1, -1, True],
)
def test_invalid_class_count_rejected(num_classes):
    with pytest.raises((TypeError, ValueError)):
        vae.build_models(num_classes=num_classes)


def test_noninteger_seed_rejected():
    mean = tf.zeros((2, 100), tf.float32)
    logvar = tf.zeros_like(mean)

    with pytest.raises(TypeError):
        vae.reparameterize(
            mean, logvar,
            seed=tf.constant([42.0, 0.0], tf.float32),
        )


def test_wrong_seed_shape_rejected():
    mean = tf.zeros((2, 100), tf.float32)
    logvar = tf.zeros_like(mean)

    with pytest.raises(tf.errors.InvalidArgumentError):
        vae.reparameterize(
            mean, logvar,
            seed=tf.constant([42], tf.int32),
        )


def test_nonfinite_posterior_rejected():
    mean = tf.zeros((2, 100), tf.float32)
    logvar = tf.fill((2, 100), float("nan"))

    with pytest.raises(tf.errors.InvalidArgumentError):
        vae.reparameterize(
            mean, logvar,
            seed=[42, 0],
        )


def test_overflowing_posterior_scale_rejected():
    mean = tf.zeros((2, 100), tf.float32)
    logvar = tf.fill((2, 100), 200.0)

    with pytest.raises(tf.errors.InvalidArgumentError):
        vae.reparameterize(
            mean, logvar,
            seed=[42, 0],
        )
