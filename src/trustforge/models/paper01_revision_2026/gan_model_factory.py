"""Revision-2026 conditional GAN model construction.

Preserves historical generator and discriminator architectures.
Models are not compiled; explicit gradient tapes will govern updates.
"""

from __future__ import annotations

from dataclasses import dataclass
import math



GAN_IMAGE_SHAPE = (40, 40, 3)
GAN_LATENT_DIM = 100
GAN_LEARNING_RATE = 2e-4
GAN_BETA_1 = 0.5


@dataclass(frozen=True)
class GANModelConfig:
    num_classes: int
    latent_dim: int = GAN_LATENT_DIM
    learning_rate: float = GAN_LEARNING_RATE
    beta_1: float = GAN_BETA_1

    def __post_init__(self) -> None:
        for name in ("num_classes", "latent_dim"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")

        if self.latent_dim != GAN_LATENT_DIM:
            raise ValueError("revision GAN latent dimension must be 100")

        if (
            isinstance(self.learning_rate, bool)
            or not isinstance(self.learning_rate, (int, float))
            or not math.isfinite(self.learning_rate)
            or self.learning_rate != GAN_LEARNING_RATE
        ):
            raise ValueError("revision GAN learning rate must be 0.0002")

        if (
            isinstance(self.beta_1, bool)
            or not isinstance(self.beta_1, (int, float))
            or not math.isfinite(self.beta_1)
            or self.beta_1 != GAN_BETA_1
        ):
            raise ValueError("revision GAN beta_1 must be 0.5")


def build_revision_gan(config: GANModelConfig):
    """Build historical architectures with independent Adam optimizers.

    This function constructs models but performs no training.
    """
    import tensorflow as tf
    from gan.models import build_generator, build_discriminator

    generator = build_generator(
        latent_dim=config.latent_dim,
        num_classes=config.num_classes,
        img_shape=GAN_IMAGE_SHAPE,
    )

    discriminator = build_discriminator(
        img_shape=GAN_IMAGE_SHAPE,
        num_classes=config.num_classes,
    )

    if tuple(generator.output_shape[1:]) != GAN_IMAGE_SHAPE:
        raise RuntimeError("unexpected generator output shape")

    if tuple(discriminator.output_shape[1:]) != (1,):
        raise RuntimeError("unexpected discriminator output shape")

    if len(generator.inputs) != 2 or len(discriminator.inputs) != 2:
        raise RuntimeError("GAN class-conditioning inputs are missing")

    generator_optimizer = tf.keras.optimizers.Adam(
        learning_rate=config.learning_rate,
        beta_1=config.beta_1,
    )

    discriminator_optimizer = tf.keras.optimizers.Adam(
        learning_rate=config.learning_rate,
        beta_1=config.beta_1,
    )

    return {
        "generator": generator,
        "discriminator": discriminator,
        "generator_optimizer": generator_optimizer,
        "discriminator_optimizer": discriminator_optimizer,
    }
