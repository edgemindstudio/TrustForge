"""Compute-node check of GAN BatchNormalization training modes."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from trustforge.models.paper01_revision_2026.gan_model_factory import (
    GANModelConfig,
    build_revision_gan,
)


def bn_variables(model):
    return [
        variable
        for layer in model.layers
        if isinstance(layer, tf.keras.layers.BatchNormalization)
        for variable in (layer.moving_mean, layer.moving_variance)
    ]


def snapshot(variables):
    return [np.asarray(variable.numpy()).copy() for variable in variables]


def identical(before, variables):
    return len(before) == len(variables) and all(
        np.array_equal(old, np.asarray(v.numpy()))
        for old, v in zip(before, variables)
    )


def main():
    tf.keras.utils.set_random_seed(42)

    models = build_revision_gan(GANModelConfig(num_classes=2))
    generator = models["generator"]
    discriminator = models["discriminator"]

    g_bn = bn_variables(generator)
    d_bn = bn_variables(discriminator)
    assert g_bn and d_bn

    labels = tf.one_hot([0, 1, 0, 1], depth=2)
    noise = tf.random.normal((4, 100))
    images = tf.random.uniform((4, 40, 40, 3), -1.0, 1.0)

    # Discriminator real phase.
    g_before = snapshot(g_bn)
    d_before = snapshot(d_bn)
    discriminator([images, labels], training=True)

    assert identical(g_before, g_bn)
    assert not identical(d_before, d_bn)
    print("REAL_D_PHASE_BATCHNORM=PASS", flush=True)

    # Discriminator fake phase: G runs in inference mode.
    g_before = snapshot(g_bn)
    d_before = snapshot(d_bn)
    fake = generator([noise, labels], training=False)
    discriminator([tf.stop_gradient(fake), labels], training=True)

    assert identical(g_before, g_bn)
    assert not identical(d_before, d_bn)
    print("FAKE_D_PHASE_BATCHNORM=PASS", flush=True)

    # Generator phase: G trains, D is inference-only.
    g_before = snapshot(g_bn)
    d_before = snapshot(d_bn)
    generated = generator([noise, labels], training=True)
    discriminator([generated, labels], training=False)

    assert not identical(g_before, g_bn)
    assert identical(d_before, d_bn)
    print("GENERATOR_PHASE_BATCHNORM=PASS", flush=True)

    print("GAN_BATCHNORM_PHASE_TEST=PASS", flush=True)


if __name__ == "__main__":
    main()
