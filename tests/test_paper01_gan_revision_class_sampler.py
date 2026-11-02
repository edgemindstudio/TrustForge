"""Regression tests for the revision-2026 GAN class sampler."""

import numpy as np
import pytest

from trustforge.models.paper01_revision_2026.gan_class_sampler import (
    GANTrainingClassSampler,
)


def _positions():
    return (
        np.arange(0, 10, dtype=np.int64),
        np.arange(10, 30, dtype=np.int64),
        np.arange(30, 70, dtype=np.int64),
    )


def test_probabilities_follow_governed_power_law():
    sampler = GANTrainingClassSampler(_positions(), seed=42)

    counts = np.array([10, 20, 40], dtype=np.float64)
    weights = counts ** 0.85
    expected = weights / weights.sum()

    np.testing.assert_allclose(
        sampler.probabilities,
        expected,
        rtol=1e-12,
        atol=1e-12,
    )
    assert np.isclose(sampler.probabilities.sum(), 1.0)


def test_drawn_positions_match_selected_classes():
    sampler = GANTrainingClassSampler(_positions(), seed=42)

    positions, classes = sampler.draw(4096)

    assert positions.shape == (4096,)
    assert classes.shape == (4096,)
    assert positions.dtype == np.int64
    assert classes.dtype == np.int32

    allowed = _positions()

    for class_id, available in enumerate(allowed):
        selected = positions[classes == class_id]
        assert np.isin(selected, available).all()


def test_identical_seed_reproduces_draws():
    first = GANTrainingClassSampler(_positions(), seed=123)
    second = GANTrainingClassSampler(_positions(), seed=123)

    for size in (1, 17, 256):
        a_positions, a_classes = first.draw(size)
        b_positions, b_classes = second.draw(size)

        np.testing.assert_array_equal(a_positions, b_positions)
        np.testing.assert_array_equal(a_classes, b_classes)


def test_different_seeds_change_sampling_sequence():
    first = GANTrainingClassSampler(_positions(), seed=123)
    second = GANTrainingClassSampler(_positions(), seed=124)

    a_positions, a_classes = first.draw(256)
    b_positions, b_classes = second.draw(256)

    assert not (
        np.array_equal(a_positions, b_positions)
        and np.array_equal(a_classes, b_classes)
    )


def test_invalid_seed_and_alpha_rejected():
    for invalid_seed in (-1, True, 2**64, 1.5):
        with pytest.raises(ValueError):
            GANTrainingClassSampler(
                _positions(), seed=invalid_seed
            )

    for invalid_alpha in (0.7, 1.0, float("nan")):
        with pytest.raises(ValueError):
            GANTrainingClassSampler(
                _positions(), seed=42, alpha=invalid_alpha
            )


def test_invalid_position_arrays_rejected():
    cases = [
        (),
        (np.array([], dtype=np.int64),),
        (np.array([0.5, 1.5]),),
        (np.array([-1, 0]),),
        (np.array([0, 1]), np.array([1, 2])),
        (np.array([[0, 1]]),),
    ]

    for positions in cases:
        with pytest.raises(ValueError):
            GANTrainingClassSampler(
                positions, seed=42
            )


def test_invalid_batch_sizes_rejected():
    sampler = GANTrainingClassSampler(_positions(), seed=42)

    for invalid in (0, -1, True, 2.5):
        with pytest.raises(ValueError):
            sampler.draw(invalid)


def test_source_arrays_remain_unchanged():
    sources = _positions()
    originals = tuple(x.copy() for x in sources)

    sampler = GANTrainingClassSampler(sources, seed=42)
    sampler.draw(1024)

    for source, original in zip(sources, originals):
        np.testing.assert_array_equal(source, original)
