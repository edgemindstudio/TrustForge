"""Behavioral tests for the structured synthetic VAE adapter."""

import numpy as np
import pytest

from trustforge.models.paper01_revision_2026.variational_autoencoder_structured_synthetic_adapter import (
    StructuredSyntheticVAEAdapter,
)


def test_deterministic_reproduction():
    first = StructuredSyntheticVAEAdapter(
        train_count=32, validation_count=16, seed=2026
    )
    second = StructuredSyntheticVAEAdapter(
        train_count=32, validation_count=16, seed=2026
    )

    for split in ("train", "validation"):
        positions = np.arange(first.split_size(split))
        images_a, labels_a = first.canonical_batch(split, positions)
        images_b, labels_b = second.canonical_batch(split, positions)

        np.testing.assert_array_equal(images_a, images_b)
        np.testing.assert_array_equal(labels_a, labels_b)


def test_different_seeds_change_images():
    first = StructuredSyntheticVAEAdapter(
        train_count=32, validation_count=16, seed=2026
    )
    second = StructuredSyntheticVAEAdapter(
        train_count=32, validation_count=16, seed=2027
    )

    indices = np.arange(32)

    images_a, _ = first.canonical_batch("train", indices)
    images_b, _ = second.canonical_batch("train", indices)

    assert not np.array_equal(images_a, images_b)


def test_shapes_ranges_and_class_balance():
    adapter = StructuredSyntheticVAEAdapter(
        train_count=32, validation_count=16
    )

    for split, count in (("train", 32), ("validation", 16)):
        images, labels = adapter.canonical_batch(
            split, np.arange(count)
        )

        assert images.shape == (count, 40, 40, 3)
        assert labels.shape == (count, 2)
        assert images.dtype == np.float32
        assert labels.dtype == np.float32
        assert np.isfinite(images).all()
        assert np.min(images) >= 0.0
        assert np.max(images) <= 1.0

        np.testing.assert_array_equal(
            labels.sum(axis=0),
            [count // 2, count // 2],
        )


def test_class_specific_horizontal_centers():
    adapter = StructuredSyntheticVAEAdapter(train_count=64)

    images, labels = adapter.canonical_batch(
        "train", np.arange(64)
    )

    x_coordinates = np.arange(40, dtype=np.float64)

    for class_id in (0, 1):
        selected = images[labels[:, class_id] == 1.0]
        intensity = selected.mean(axis=(1, 3))
        centers = (
            intensity @ x_coordinates
        ) / intensity.sum(axis=1)

        if class_id == 0:
            assert np.all(centers < 20.0)
        else:
            assert np.all(centers > 20.0)


def test_within_class_continuous_variation():
    adapter = StructuredSyntheticVAEAdapter(train_count=64)

    images, labels = adapter.canonical_batch(
        "train", np.arange(64)
    )

    for class_id in (0, 1):
        selected = images[labels[:, class_id] == 1.0]
        assert np.var(selected.mean(axis=(1, 2, 3))) > 0.0
        assert not np.array_equal(selected[0], selected[1])


def test_train_validation_are_distinct():
    adapter = StructuredSyntheticVAEAdapter(
        train_count=32, validation_count=16
    )

    train, _ = adapter.canonical_batch(
        "train", np.arange(16)
    )
    validation, _ = adapter.canonical_batch(
        "validation", np.arange(16)
    )

    assert not np.array_equal(train, validation)


def test_batches_are_independent_copies():
    adapter = StructuredSyntheticVAEAdapter(train_count=16)

    positions = np.array([0, 1], dtype=np.int64)

    original_images, original_labels = adapter.canonical_batch(
        "train", positions
    )

    modified_images, modified_labels = adapter.canonical_batch(
        "train", positions
    )

    modified_images[:] = 0.0
    modified_labels[:] = 0.0

    recovered_images, recovered_labels = adapter.canonical_batch(
        "train", positions
    )

    np.testing.assert_array_equal(
        original_images, recovered_images
    )
    np.testing.assert_array_equal(
        original_labels, recovered_labels
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"train_count": 0},
        {"validation_count": 1},
        {"train_count": True},
        {"seed": -1},
        {"seed": 2**31},
        {"seed": 1.5},
    ],
)
def test_invalid_configuration_rejected(kwargs):
    with pytest.raises((TypeError, ValueError)):
        StructuredSyntheticVAEAdapter(**kwargs)


def test_test_partition_rejected():
    adapter = StructuredSyntheticVAEAdapter()

    with pytest.raises(ValueError):
        adapter.split_size("test")

    with pytest.raises(ValueError):
        adapter.canonical_batch(
            "test", np.array([0], dtype=np.int64)
        )


@pytest.mark.parametrize(
    "positions",
    [
        np.array([-1], dtype=np.int64),
        np.array([999999], dtype=np.int64),
        np.array([0.0], dtype=np.float64),
        np.array([[0]], dtype=np.int64),
    ],
)
def test_invalid_positions_rejected(positions):
    adapter = StructuredSyntheticVAEAdapter()

    with pytest.raises((TypeError, ValueError, IndexError)):
        adapter.canonical_batch("train", positions)
