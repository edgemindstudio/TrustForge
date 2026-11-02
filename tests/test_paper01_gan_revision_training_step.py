"""Focused TensorFlow regression tests for GAN training updates."""

import math

import numpy as np
import pytest
import tensorflow as tf

from trustforge.models.paper01_revision_2026.gan_model_factory import (
    GANModelConfig,
    build_revision_gan,
)
from trustforge.models.paper01_revision_2026.gan_training_step import (
    gan_training_step,
)


def make_case():
    tf.keras.utils.set_random_seed(42)
    models = build_revision_gan(GANModelConfig(num_classes=9))
    labels = tf.one_hot([0, 1], depth=9, dtype=tf.float32)
    real = tf.zeros((2, 40, 40, 3), dtype=tf.float32)
    d_noise = tf.random.stateless_normal((2, 100), seed=[42, 1])
    g_noise = tf.random.stateless_normal((2, 100), seed=[42, 2])
    return models, {
        "real_images": real,
        "real_labels": labels,
        "fake_labels": labels,
        "generator_labels": labels,
        "discriminator_noise": d_noise,
        "generator_noise": g_noise,
    }


def snapshot(variables):
    return [v.numpy().copy() for v in variables]


def changed(before, variables):
    return any(
        np.any(old != var.numpy())
        for old, var in zip(before, variables)
    )


def test_real_tensorflow_iteration_updates_both_networks():
    models, batch = make_case()
    generator = models["generator"]
    discriminator = models["discriminator"]

    g_before = snapshot(generator.trainable_variables)
    d_before = snapshot(discriminator.trainable_variables)

    losses = gan_training_step(**models, **batch)

    assert set(losses) == {
        "discriminator_real_loss",
        "discriminator_fake_loss",
        "generator_loss",
    }
    assert all(math.isfinite(float(v)) for v in losses.values())

    assert int(models["generator_optimizer"].iterations) == 1
    assert int(models["discriminator_optimizer"].iterations) == 2

    assert changed(g_before, generator.trainable_variables)
    assert changed(d_before, discriminator.trainable_variables)

    for model in (generator, discriminator):
        for var in model.variables:
            array = var.numpy()
            if np.issubdtype(array.dtype, np.floating):
                assert np.all(np.isfinite(array))
            else:
                assert np.issubdtype(array.dtype, np.integer)


def test_invalid_class_labels_fail_before_updates():
    models, batch = make_case()

    invalid = dict(batch)
    invalid["real_labels"] = tf.zeros((2, 9), dtype=tf.float32)

    with pytest.raises(ValueError, match="one-hot"):
        gan_training_step(**models, **invalid)

    assert int(models["generator_optimizer"].iterations) == 0
    assert int(models["discriminator_optimizer"].iterations) == 0


def test_nonfinite_real_input_fails_before_updates():
    models, batch = make_case()

    invalid = dict(batch)
    invalid["real_images"] = tf.fill(
        (2, 40, 40, 3), float("nan")
    )

    with pytest.raises(tf.errors.InvalidArgumentError):
        gan_training_step(**models, **invalid)

    assert int(models["generator_optimizer"].iterations) == 0
    assert int(models["discriminator_optimizer"].iterations) == 0


def test_discriminator_state_unchanged_during_generator_only_update():
    models, batch = make_case()
    generator = models["generator"]
    discriminator = models["discriminator"]
    optimizer = models["generator_optimizer"]

    # Initialize all state before taking snapshots.
    generator(
        [batch["generator_noise"], batch["generator_labels"]],
        training=False,
    )
    discriminator(
        [batch["real_images"], batch["real_labels"]],
        training=False,
    )

    d_before = snapshot(discriminator.variables)
    g_before = snapshot(generator.trainable_variables)

    with tf.GradientTape() as tape:
        images = generator(
            [batch["generator_noise"], batch["generator_labels"]],
            training=True,
        )
        scores = discriminator(
            [images, batch["generator_labels"]],
            training=False,
        )
        loss = tf.keras.losses.BinaryCrossentropy()(
            tf.ones_like(scores), scores
        )

    gradients = tape.gradient(
        loss, generator.trainable_variables
    )
    assert all(g is not None for g in gradients)
    for grad in gradients:
        tf.debugging.assert_all_finite(
            grad, "nonfinite generator gradient"
        )

    optimizer.apply_gradients(
        zip(gradients, generator.trainable_variables)
    )

    assert int(optimizer.iterations) == 1
    assert changed(g_before, generator.trainable_variables)

    for before, variable in zip(
        d_before, discriminator.variables
    ):
        np.testing.assert_array_equal(
            before,
            variable.numpy(),
            err_msg=f"discriminator state changed: {variable.name}",
        )
