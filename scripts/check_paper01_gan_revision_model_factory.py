"""Compute-node smoke test for Paper 1 revision-2026 GAN models."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from trustforge.models.paper01_revision_2026.gan_model_factory import (
    GANModelConfig,
    build_revision_gan,
)


def main() -> None:
    tf.keras.utils.set_random_seed(42)

    for classes in (2, 5, 9, 47):
        tf.keras.backend.clear_session()

        config = GANModelConfig(num_classes=classes)
        models = build_revision_gan(config)

        generator = models["generator"]
        discriminator = models["discriminator"]

        assert generator.output_shape == (
            None, 40, 40, 3
        )
        assert discriminator.output_shape == (
            None, 1
        )

        assert len(generator.inputs) == 2
        assert len(discriminator.inputs) == 2

        assert (
            models["generator_optimizer"]
            is not models["discriminator_optimizer"]
        )

        z = tf.zeros((2, config.latent_dim), dtype=tf.float32)
        labels = tf.one_hot(
            [0, classes - 1],
            depth=classes,
            dtype=tf.float32,
        )

        generated = generator(
            [z, labels], training=False
        )

        scores = discriminator(
            [generated, labels], training=False
        )

        images = np.asarray(generated)
        validity = np.asarray(scores)

        assert images.shape == (2, 40, 40, 3)
        assert validity.shape == (2, 1)

        assert np.isfinite(images).all()
        assert np.isfinite(validity).all()

        assert np.all(images >= -1.0)
        assert np.all(images <= 1.0)

        assert np.all(validity >= 0.0)
        assert np.all(validity <= 1.0)

        print(
            f"CLASS_COUNT={classes} "
            "MODEL_CONSTRUCTION=PASS "
            "FORWARD_PASS=PASS",
            flush=True,
        )

    print("GAN_MODEL_FACTORY_SMOKE_TEST=PASS")


if __name__ == "__main__":
    main()
