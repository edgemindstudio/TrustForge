"""Governed VAE training-step regression tests, Paper 1 revision 2026."""

import numpy as np
import pytest

# Explicit initialization avoids the observed Keras import-order failure.
import optree
import keras
import tensorflow as tf

from trustforge.models.paper01_revision_2026 import (
    variational_autoencoder_model as model,
    variational_autoencoder_training_step as training,
)


def make_case():
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(42)

    encoder, decoder = model.build_models(
        latent_dim=100,
        num_classes=9,
    )
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=1e-4,
    )
    images = tf.fill((2, 40, 40, 3), 0.5)
    labels = tf.one_hot([0, 1], 9, dtype=tf.float32)

    return encoder, decoder, optimizer, images, labels


def run_step(encoder, decoder, optimizer, images, labels, seed=None):
    if seed is None:
        seed = tf.constant([42, 0], tf.int32)

    return training.vae_training_step(
        encoder=encoder,
        decoder=decoder,
        optimizer=optimizer,
        canonical_images=images,
        class_labels=labels,
        seed=seed,
    )


def snapshot(encoder, decoder):
    return [
        v.numpy().copy()
        for v in (
            encoder.trainable_variables
            + decoder.trainable_variables
        )
    ]


def test_one_step_updates_both_networks_once():
    encoder, decoder, optimizer, images, labels = make_case()

    before_encoder = [
        v.numpy().copy() for v in encoder.trainable_variables
    ]
    before_decoder = [
        v.numpy().copy() for v in decoder.trainable_variables
    ]

    assert int(optimizer.iterations.numpy()) == 0

    result = run_step(
        encoder, decoder, optimizer, images, labels
    )

    assert set(result) == {
        "total_loss",
        "reconstruction_loss",
        "kl_loss",
    }
    assert all(np.isfinite(v) for v in result.values())

    np.testing.assert_allclose(
        result["total_loss"],
        result["reconstruction_loss"] + result["kl_loss"],
        rtol=1e-5,
        atol=1e-4,
    )

    assert result["reconstruction_loss"] > 0.0
    assert result["kl_loss"] >= 0.0

    assert any(
        not np.array_equal(old, new.numpy())
        for old, new in zip(
            before_encoder, encoder.trainable_variables
        )
    )
    assert any(
        not np.array_equal(old, new.numpy())
        for old, new in zip(
            before_decoder, decoder.trainable_variables
        )
    )

    assert int(optimizer.iterations.numpy()) == 1


def test_identical_initialization_and_seed_are_reproducible():
    results = []
    final_weights = []

    for _ in range(2):
        encoder, decoder, optimizer, images, labels = make_case()

        results.append(
            run_step(
                encoder, decoder, optimizer, images, labels
            )
        )
        final_weights.append(snapshot(encoder, decoder))

    for key in results[0]:
        np.testing.assert_allclose(
            results[0][key],
            results[1][key],
            rtol=1e-5,
            atol=1e-4,
        )

    for left, right in zip(*final_weights):
        np.testing.assert_allclose(
            left, right, rtol=1e-5, atol=1e-6
        )


@pytest.mark.parametrize(
    "invalid_labels",
    [
        [[0.0] * 9, [0.0] * 9],
        [[1.0] * 9, [0.0] * 9],
        [[0.5, 0.5] + [0.0] * 7, [0.0, 1.0] + [0.0] * 7],
    ],
)
def test_invalid_one_hot_rejected_before_update(invalid_labels):
    encoder, decoder, optimizer, images, _ = make_case()
    before = snapshot(encoder, decoder)

    with pytest.raises((ValueError, tf.errors.OpError)):
        run_step(
            encoder,
            decoder,
            optimizer,
            images,
            tf.constant(invalid_labels, tf.float32),
        )

    assert int(optimizer.iterations.numpy()) == 0

    for old, new in zip(before, snapshot(encoder, decoder)):
        np.testing.assert_array_equal(old, new)


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan")])
def test_invalid_canonical_images_rejected_before_update(value):
    encoder, decoder, optimizer, _, labels = make_case()
    images = tf.fill((2, 40, 40, 3), value)
    before = snapshot(encoder, decoder)

    with pytest.raises((ValueError, tf.errors.OpError)):
        run_step(
            encoder, decoder, optimizer, images, labels
        )

    assert int(optimizer.iterations.numpy()) == 0

    for old, new in zip(before, snapshot(encoder, decoder)):
        np.testing.assert_array_equal(old, new)


def test_noninteger_seed_rejected_before_update():
    encoder, decoder, optimizer, images, labels = make_case()
    before = snapshot(encoder, decoder)

    with pytest.raises(TypeError):
        run_step(
            encoder,
            decoder,
            optimizer,
            images,
            labels,
            seed=tf.constant([42.0, 0.0], tf.float32),
        )

    assert int(optimizer.iterations.numpy()) == 0

    for old, new in zip(before, snapshot(encoder, decoder)):
        np.testing.assert_array_equal(old, new)


def test_invalid_seed_shape_rejected_before_update():
    encoder, decoder, optimizer, images, labels = make_case()
    before = snapshot(encoder, decoder)

    with pytest.raises(tf.errors.OpError):
        run_step(
            encoder,
            decoder,
            optimizer,
            images,
            labels,
            seed=tf.constant([42], tf.int32),
        )

    assert int(optimizer.iterations.numpy()) == 0

    for old, new in zip(before, snapshot(encoder, decoder)):
        np.testing.assert_array_equal(old, new)


def test_wrong_image_dtype_rejected_before_update():
    encoder, decoder, optimizer, images, labels = make_case()
    before = snapshot(encoder, decoder)

    with pytest.raises(TypeError):
        run_step(
            encoder,
            decoder,
            optimizer,
            tf.cast(images, tf.float64),
            labels,
        )

    assert int(optimizer.iterations.numpy()) == 0

    for old, new in zip(before, snapshot(encoder, decoder)):
        np.testing.assert_array_equal(old, new)
