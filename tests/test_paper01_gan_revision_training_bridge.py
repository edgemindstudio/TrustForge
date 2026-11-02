"""Regression tests for revision-2026 GAN training-batch bridge."""

import numpy as np
import pytest

from trustforge.models.paper01_revision_2026.gan_training_bridge import (
    GANTrainingBridge,
)


class FakeSampler:
    probabilities = np.asarray([0.25, 0.75], dtype=np.float64)


class FakeTrainingData:
    class_count = 2

    def __init__(self):
        self.sampler = FakeSampler()
        self.calls = 0

    def draw(self, batch_size):
        self.calls += 1
        positions = np.arange(batch_size, dtype=np.int64)
        classes = positions % 2

        images = np.full(
            (batch_size, 40, 40, 3),
            0.25,
            dtype=np.float32,
        )
        labels = np.eye(2, dtype=np.float32)[classes]
        return images, labels, positions


def test_shapes_and_domains():
    data = FakeTrainingData()
    batch = GANTrainingBridge(data, seed=42).draw(8)

    assert batch.real_images.shape == (8, 40, 40, 3)
    assert batch.real_images.dtype == np.float32
    assert np.all(batch.real_images == -0.5)

    for labels in (
        batch.real_labels,
        batch.fake_labels,
        batch.generator_labels,
    ):
        assert labels.shape == (8, 2)
        assert labels.dtype == np.float32
        np.testing.assert_array_equal(labels.sum(axis=1), 1.0)

    for noise in (
        batch.discriminator_noise,
        batch.generator_noise,
    ):
        assert noise.shape == (8, 100)
        assert noise.dtype == np.float32
        assert np.isfinite(noise).all()

    np.testing.assert_array_equal(
        batch.train_positions,
        np.arange(8),
    )
    assert data.calls == 1


def test_reproducible_random_streams():
    left = GANTrainingBridge(FakeTrainingData(), seed=2026)
    right = GANTrainingBridge(FakeTrainingData(), seed=2026)

    for _ in range(3):
        a = left.draw(16)
        b = right.draw(16)

        for field in (
            "fake_labels",
            "generator_labels",
            "discriminator_noise",
            "generator_noise",
        ):
            np.testing.assert_array_equal(
                getattr(a, field),
                getattr(b, field),
            )


def test_different_seeds_change_noise():
    left = GANTrainingBridge(FakeTrainingData(), seed=11).draw(8)
    right = GANTrainingBridge(FakeTrainingData(), seed=12).draw(8)

    assert not np.array_equal(
        left.discriminator_noise,
        right.discriminator_noise,
    )


def test_noise_streams_are_independent():
    batch = GANTrainingBridge(FakeTrainingData(), seed=99).draw(8)

    assert not np.array_equal(
        batch.discriminator_noise,
        batch.generator_noise,
    )


def test_training_step_inputs_exclude_positions():
    batch = GANTrainingBridge(FakeTrainingData(), seed=3).draw(4)

    assert set(batch.training_step_inputs()) == {
        "real_images",
        "real_labels",
        "fake_labels",
        "generator_labels",
        "discriminator_noise",
        "generator_noise",
    }


@pytest.mark.parametrize("seed", [-1, True, 2**64, 1.5])
def test_invalid_seed_rejected(seed):
    with pytest.raises(ValueError):
        GANTrainingBridge(FakeTrainingData(), seed=seed)


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
def test_invalid_batch_size_rejected(batch_size):
    bridge = GANTrainingBridge(FakeTrainingData(), seed=5)

    with pytest.raises(ValueError):
        bridge.draw(batch_size)


def test_invalid_probability_vector_rejected():
    data = FakeTrainingData()
    data.sampler = FakeSampler()
    data.sampler.probabilities = np.array([0.3, 0.3])

    with pytest.raises(ValueError):
        GANTrainingBridge(data, seed=5)


def test_no_extra_real_image_loading_for_conditioning():
    data = FakeTrainingData()
    bridge = GANTrainingBridge(data, seed=7)

    bridge.draw(32)
    bridge.draw(32)

    assert data.calls == 2
