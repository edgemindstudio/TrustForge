"""Canonical batch integration tests for Paper 1 revised VAE."""

import numpy as np
import pytest

from trustforge.models.paper01_revision_2026 import (
    variational_autoencoder_data_integration as integration,
)


class SyntheticAdapter:
    def __init__(self, *, num_classes=9, train_size=12,
                 validation_size=5, image_value=0.5):
        self.num_classes = num_classes
        self.sizes = {
            "train": train_size,
            "validation": validation_size,
        }
        self.image_value = image_value
        self.calls = []

    def split_size(self, split):
        self.calls.append(("split_size", split))
        if split not in self.sizes:
            raise AssertionError("unauthorized split access")
        return self.sizes[split]

    def canonical_batch(self, split, positions):
        self.calls.append(
            ("canonical_batch", split, positions.copy())
        )
        if split not in self.sizes:
            raise AssertionError("unauthorized split access")

        count = len(positions)
        images = np.full(
            (count, 40, 40, 3),
            self.image_value,
            dtype=np.float32,
        )
        labels = np.zeros(
            (count, self.num_classes),
            dtype=np.float32,
        )
        labels[
            np.arange(count),
            positions % self.num_classes,
        ] = 1.0

        return images, labels


def fetch(adapter, split="train", positions=None, num_classes=9):
    if positions is None:
        positions = np.array([0, 2, 4], dtype=np.int64)

    return integration.canonical_vae_batch(
        adapter=adapter,
        split=split,
        logical_positions=positions,
        num_classes=num_classes,
    )


@pytest.mark.parametrize("num_classes", [2, 5, 9, 47])
def test_valid_batch_for_four_class_counts(num_classes):
    adapter = SyntheticAdapter(num_classes=num_classes)

    images, labels = fetch(
        adapter, num_classes=num_classes
    )

    assert images.shape == (3, 40, 40, 3)
    assert labels.shape == (3, num_classes)
    assert images.dtype == np.float32
    assert labels.dtype == np.float32
    assert np.all(images == 0.5)
    assert np.all(labels.sum(axis=1) == 1.0)


def test_validation_partition_access():
    adapter = SyntheticAdapter()

    fetch(
        adapter,
        split="validation",
        positions=np.array([0, 1], dtype=np.int64),
    )

    assert adapter.calls[0] == ("split_size", "validation")
    assert adapter.calls[1][0:2] == (
        "canonical_batch", "validation"
    )


def test_test_partition_rejected_without_adapter_access():
    adapter = SyntheticAdapter()

    with pytest.raises(ValueError):
        fetch(adapter, split="test")

    assert adapter.calls == []


@pytest.mark.parametrize(
    "positions",
    [
        np.array([-1], dtype=np.int64),
        np.array([12], dtype=np.int64),
        np.array([2, 2], dtype=np.int64),
        np.array([], dtype=np.int64),
        np.array([[0, 1]], dtype=np.int64),
    ],
)
def test_invalid_positions_rejected_before_batch_access(positions):
    adapter = SyntheticAdapter()

    with pytest.raises(ValueError):
        fetch(adapter, positions=positions)

    assert all(call[0] != "canonical_batch" for call in adapter.calls)


def test_wrong_position_dtype_rejected():
    adapter = SyntheticAdapter()

    with pytest.raises(TypeError):
        fetch(
            adapter,
            positions=np.array([0, 1], dtype=np.int32),
        )


def test_wrong_class_count_rejected():
    adapter = SyntheticAdapter(num_classes=9)

    with pytest.raises(ValueError):
        fetch(adapter, num_classes=5)


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan")])
def test_invalid_image_values_rejected(value):
    adapter = SyntheticAdapter(image_value=value)

    with pytest.raises(ValueError):
        fetch(adapter)


def test_mutating_adapter_cannot_change_caller_positions():
    class MutatingAdapter(SyntheticAdapter):
        def canonical_batch(self, split, positions):
            original = positions.copy()
            result = super().canonical_batch(split, positions)
            positions[:] = 0
            assert not np.array_equal(original, positions)
            return result

    adapter = MutatingAdapter()
    original = np.array([0, 2, 4], dtype=np.int64)
    before = original.copy()

    fetch(adapter, positions=original)

    np.testing.assert_array_equal(original, before)


def test_invalid_image_dtype_rejected():
    class InvalidAdapter(SyntheticAdapter):
        def canonical_batch(self, split, positions):
            images, labels = super().canonical_batch(
                split, positions
            )
            return images.astype(np.float64), labels

    with pytest.raises(TypeError):
        fetch(InvalidAdapter())


def test_invalid_label_dtype_rejected():
    class InvalidAdapter(SyntheticAdapter):
        def canonical_batch(self, split, positions):
            images, labels = super().canonical_batch(
                split, positions
            )
            return images, labels.astype(np.float64)

    with pytest.raises(TypeError):
        fetch(InvalidAdapter())


@pytest.mark.parametrize("mode", ["all_zero", "multi_hot", "fractional"])
def test_invalid_one_hot_labels_rejected(mode):
    class InvalidAdapter(SyntheticAdapter):
        def canonical_batch(self, split, positions):
            images, labels = super().canonical_batch(
                split, positions
            )

            if mode == "all_zero":
                labels[0, :] = 0.0
            elif mode == "multi_hot":
                labels[0, 0:2] = 1.0
            else:
                labels[0, :] = 0.0
                labels[0, 0:2] = 0.5

            return images, labels

    with pytest.raises(ValueError):
        fetch(InvalidAdapter())


def test_invalid_adapter_rejected():
    with pytest.raises(TypeError):
        fetch(object())


def test_invalid_num_classes_rejected():
    with pytest.raises(ValueError):
        fetch(SyntheticAdapter(), num_classes=1)


def test_empty_partition_rejected():
    with pytest.raises(ValueError):
        fetch(SyntheticAdapter(train_size=0))
