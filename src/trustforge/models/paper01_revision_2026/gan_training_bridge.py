"""Governed batch preparation for Paper 1 revision-2026 GAN.

Uses the existing train-only dataset sampler and matched class
conditioning probabilities. No model or training execution here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trustforge.models.paper01_revision_2026.gan_image_domain import (
    canonical_to_gan,
)


@dataclass(frozen=True)
class GANPreparedBatch:
    real_images: np.ndarray
    real_labels: np.ndarray
    fake_labels: np.ndarray
    generator_labels: np.ndarray
    discriminator_noise: np.ndarray
    generator_noise: np.ndarray
    train_positions: np.ndarray

    def training_step_inputs(self) -> dict:
        """Return only batch arguments accepted by gan_training_step."""
        return {
            "real_images": self.real_images,
            "real_labels": self.real_labels,
            "fake_labels": self.fake_labels,
            "generator_labels": self.generator_labels,
            "discriminator_noise": self.discriminator_noise,
            "generator_noise": self.generator_noise,
        }


class GANTrainingBridge:
    """Prepare conditional GAN inputs from governed training data."""

    def __init__(self, training_data, *, seed: int):
        if isinstance(seed, (bool, np.bool_)) or not isinstance(
            seed, (int, np.integer)
        ) or not 0 <= int(seed) < 2**64:
            raise ValueError("seed must be an unsigned 64-bit integer")

        if not hasattr(training_data, "draw") or not hasattr(
            training_data, "sampler"
        ):
            raise TypeError("expected initialized GANTrainingData")

        self.training_data = training_data
        self.class_count = int(training_data.class_count)
        self.latent_dim = 100

        probabilities = np.asarray(
            training_data.sampler.probabilities, dtype=np.float64
        )

        if (
            probabilities.shape != (self.class_count,)
            or not np.all(np.isfinite(probabilities))
            or np.any(probabilities <= 0)
            or not np.isclose(probabilities.sum(), 1.0, atol=1e-12)
        ):
            raise ValueError("invalid governed class probabilities")

        self.probabilities = probabilities.copy()
        self.probabilities.flags.writeable = False

        streams = np.random.SeedSequence(int(seed)).spawn(3)
        self.fake_label_rng = np.random.default_rng(streams[0])
        self.generator_label_rng = np.random.default_rng(streams[1])
        self.noise_rng = np.random.default_rng(streams[2])

    def _labels(self, batch_size: int, rng) -> np.ndarray:
        ids = rng.choice(
            self.class_count,
            size=batch_size,
            p=self.probabilities,
        )
        labels = np.eye(self.class_count, dtype=np.float32)[ids]
        return labels

    def draw(self, batch_size: int) -> GANPreparedBatch:
        """Draw one real batch and independently conditioned GAN inputs."""
        if isinstance(batch_size, (bool, np.bool_)) or not isinstance(
            batch_size, (int, np.integer)
        ) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        images, real_labels, positions = self.training_data.draw(
            int(batch_size)
        )

        real_images = canonical_to_gan(images)
        n = int(batch_size)

        fake_labels = self._labels(n, self.fake_label_rng)
        generator_labels = self._labels(n, self.generator_label_rng)

        noises = self.noise_rng.standard_normal(
            size=(2, n, self.latent_dim)
        ).astype(np.float32)

        discriminator_noise = noises[0].copy()
        generator_noise = noises[1].copy()

        if (
            not np.isfinite(discriminator_noise).all()
            or not np.isfinite(generator_noise).all()
        ):
            raise ValueError("nonfinite GAN noise")

        return GANPreparedBatch(
            real_images=real_images,
            real_labels=real_labels,
            fake_labels=fake_labels,
            generator_labels=generator_labels,
            discriminator_noise=discriminator_noise,
            generator_noise=generator_noise,
            train_positions=np.asarray(positions, dtype=np.int64).copy(),
        )
