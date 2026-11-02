"""Compute-node correctness smoke test for revision-2026 GAN updates."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from trustforge.models.paper01_revision_2026.gan_model_factory import (
    GANModelConfig,
    build_revision_gan,
)
from trustforge.models.paper01_revision_2026.gan_training_step import (
    gan_training_step,
)


def snapshot(variables):
    return [np.asarray(v.numpy()).copy() for v in variables]


def changed(before, variables):
    return any(
        not np.array_equal(old, np.asarray(new.numpy()))
        for old, new in zip(before, variables)
    )


def batchnorm_state(model):
    return [
        variable
        for layer in model.layers
        if isinstance(layer, tf.keras.layers.BatchNormalization)
        for variable in (layer.moving_mean, layer.moving_variance)
    ]


def main():
    tf.keras.utils.set_random_seed(42)

    models = build_revision_gan(GANModelConfig(num_classes=2))
    generator = models["generator"]
    discriminator = models["discriminator"]

    g_optimizer = models["generator_optimizer"]
    d_optimizer = models["discriminator_optimizer"]

    batch = 4
    labels = tf.one_hot([0, 1, 0, 1], depth=2)

    rng = np.random.default_rng(42)

    inputs = {
        "generator": generator,
        "discriminator": discriminator,
        "generator_optimizer": g_optimizer,
        "discriminator_optimizer": d_optimizer,
        "real_images": rng.uniform(
            -1.0, 1.0, size=(batch, 40, 40, 3)
        ).astype(np.float32),
        "real_labels": labels,
        "fake_labels": labels,
        "generator_labels": labels,
        "discriminator_noise": rng.normal(
            size=(batch, 100)
        ).astype(np.float32),
        "generator_noise": rng.normal(
            size=(batch, 100)
        ).astype(np.float32),
    }

    # Invalid input must fail before any optimizer update.
    bad = dict(inputs)
    bad["real_images"] = np.full(
        (batch, 40, 40, 3), 2.0, dtype=np.float32
    )

    try:
        gan_training_step(**bad)
    except ValueError as error:
        assert "outside internal" in str(error)
    else:
        raise AssertionError("invalid image range was accepted")

    assert int(g_optimizer.iterations.numpy()) == 0
    assert int(d_optimizer.iterations.numpy()) == 0
    print("INVALID_INPUT_PRE_UPDATE_REJECTION=PASS", flush=True)

    g_before = snapshot(generator.trainable_variables)
    d_before = snapshot(discriminator.trainable_variables)

    losses = gan_training_step(**inputs)

    assert set(losses) == {
        "discriminator_real_loss",
        "discriminator_fake_loss",
        "generator_loss",
    }
    assert all(np.isfinite(value) for value in losses.values())

    assert int(d_optimizer.iterations.numpy()) == 2
    assert int(g_optimizer.iterations.numpy()) == 1
    print("OPTIMIZER_UPDATE_COUNTS=PASS", flush=True)

    assert changed(g_before, generator.trainable_variables)
    assert changed(d_before, discriminator.trainable_variables)
    print("GENERATOR_AND_DISCRIMINATOR_WEIGHTS_UPDATED=PASS", flush=True)

    for model in (generator, discriminator):
        for variable in model.variables:
            assert np.isfinite(variable.numpy()).all()

    g_bn = batchnorm_state(generator)
    d_bn = batchnorm_state(discriminator)
    assert g_bn and d_bn
    assert all(np.isfinite(v.numpy()).all() for v in g_bn + d_bn)
    print("MODEL_AND_BATCHNORM_STATE_FINITE=PASS", flush=True)

    for name, value in losses.items():
        print(f"{name.upper()}={value:.6f}", flush=True)

    print("GAN_TRAINING_STEP_SMOKE_TEST=PASS", flush=True)


if __name__ == "__main__":
    main()
