"""Compute-node GAN optimizer and complete model-state isolation test."""

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


def snapshot(model):
    return [np.asarray(v.numpy()).copy() for v in model.variables]


def identical(before, model):
    current = model.variables
    return len(before) == len(current) and all(
        np.array_equal(old, np.asarray(new.numpy()))
        for old, new in zip(before, current)
    )


class GuardedOptimizer:
    """Verify that an optimizer update cannot mutate the opposing model."""

    def __init__(self, optimizer, owner, opposing, name):
        self.optimizer = optimizer
        self.owner = owner
        self.opposing = opposing
        self.name = name
        self.calls = 0

    def apply_gradients(self, gradients_and_variables):
        opposing_before = snapshot(self.opposing)
        owner_before = snapshot(self.owner)

        result = self.optimizer.apply_gradients(
            gradients_and_variables
        )

        if not identical(opposing_before, self.opposing):
            raise AssertionError(
                f"{self.name}: opposing model changed during update"
            )

        if identical(owner_before, self.owner):
            raise AssertionError(
                f"{self.name}: owner model did not change"
            )

        self.calls += 1
        print(
            f"{self.name}_UPDATE_{self.calls}_ISOLATION=PASS",
            flush=True,
        )
        return result


def main():
    tf.keras.utils.set_random_seed(42)
    rng = np.random.default_rng(42)

    models = build_revision_gan(GANModelConfig(num_classes=2))
    generator = models["generator"]
    discriminator = models["discriminator"]

    guarded_g = GuardedOptimizer(
        models["generator_optimizer"],
        generator,
        discriminator,
        "GENERATOR",
    )
    guarded_d = GuardedOptimizer(
        models["discriminator_optimizer"],
        discriminator,
        generator,
        "DISCRIMINATOR",
    )

    batch = 4
    labels = tf.one_hot([0, 1, 0, 1], depth=2)

    inputs = {
        "generator": generator,
        "discriminator": discriminator,
        "generator_optimizer": guarded_g,
        "discriminator_optimizer": guarded_d,
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

    losses = gan_training_step(**inputs)

    assert guarded_d.calls == 2
    assert guarded_g.calls == 1

    assert int(
        models["discriminator_optimizer"].iterations.numpy()
    ) == 2
    assert int(
        models["generator_optimizer"].iterations.numpy()
    ) == 1

    assert all(np.isfinite(v) for v in losses.values())

    print("ALL_OPTIMIZER_BOUNDARIES_ISOLATED=PASS", flush=True)
    print("GAN_STATE_ISOLATION_SMOKE_TEST=PASS", flush=True)


if __name__ == "__main__":
    main()
