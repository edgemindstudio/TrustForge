"""Compute-node test of GAN class-conditioning paths."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from trustforge.models.paper01_revision_2026.gan_model_factory import (
    GANModelConfig,
    build_revision_gan,
)


def main():
    tf.keras.utils.set_random_seed(42)
    rng = np.random.default_rng(42)

    for classes in (2, 47):
        tf.keras.backend.clear_session()

        models = build_revision_gan(
            GANModelConfig(num_classes=classes)
        )

        generator = models["generator"]
        discriminator = models["discriminator"]

        batch = 4

        noise = tf.convert_to_tensor(
            rng.normal(size=(batch, 100)).astype(np.float32)
        )
        images = tf.convert_to_tensor(
            rng.uniform(
                -1.0, 1.0, size=(batch, 40, 40, 3)
            ).astype(np.float32)
        )

        labels_a = tf.one_hot(
            np.zeros(batch, dtype=np.int32),
            depth=classes,
            dtype=tf.float32,
        )
        labels_b = tf.one_hot(
            np.full(batch, classes - 1, dtype=np.int32),
            depth=classes,
            dtype=tf.float32,
        )

        generated_a = generator(
            [noise, labels_a], training=False
        ).numpy()
        generated_b = generator(
            [noise, labels_b], training=False
        ).numpy()

        generator_difference = float(
            np.max(np.abs(generated_a - generated_b))
        )

        assert np.isfinite(generator_difference)
        assert generator_difference > 1e-8, (
            f"Generator conditioning inactive: {classes} classes"
        )

        scores_a = discriminator(
            [images, labels_a], training=False
        ).numpy()
        scores_b = discriminator(
            [images, labels_b], training=False
        ).numpy()

        discriminator_difference = float(
            np.max(np.abs(scores_a - scores_b))
        )

        assert np.isfinite(discriminator_difference)
        assert discriminator_difference > 1e-8, (
            f"Discriminator conditioning inactive: {classes} classes"
        )

        print(
            f"CLASSES={classes} "
            f"GENERATOR_LABEL_DELTA={generator_difference:.9g} "
            f"DISCRIMINATOR_LABEL_DELTA={discriminator_difference:.9g}",
            flush=True,
        )
        print(
            f"CLASSES={classes} CONDITIONING_PATHS=PASS",
            flush=True,
        )

    print("GAN_CLASS_CONDITIONING_SMOKE_TEST=PASS", flush=True)


if __name__ == "__main__":
    main()
