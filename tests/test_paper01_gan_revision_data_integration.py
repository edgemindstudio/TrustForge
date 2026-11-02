"""Focused GAN revision-2026 data-integration tests."""

import numpy as np
import pytest

from trustforge.models.paper01_revision_2026.gan_data_integration import (
    GANTrainingData,
    training_class_positions,
)


class _DatasetBase:
    def __init__(self):
        self.y = np.array([0, 1, 1, 2, 0, 2, 1, 2], dtype=np.int32)
        self.calls = []
        self.corrupt_labels = False
        self.corrupt_images = False

    def split_size(self, split):
        assert split == "train"
        self.calls.append(("size", split))
        return len(self.y)

    def canonical_batch(self, split, positions):
        assert split == "train"
        self.calls.append(("batch", split))
        labels = np.eye(3, dtype=np.float32)[self.y[positions]]
        images = np.zeros(
            (len(positions), 40, 40, 3), dtype=np.float32
        )

        if self.corrupt_labels:
            labels = np.roll(labels, 1, axis=1)

        if self.corrupt_images:
            images[0, 0, 0, 0] = 2.0

        return images, labels


def _dataset(name, positioned):
    def labels(self, split, positions=None):
        assert split == "train"
        self.calls.append(("labels", split))
        if positioned:
            assert positions is not None
            return self.y[positions]
        assert positions is None
        return self.y.copy()

    return type(
        name,
        (_DatasetBase,),
        {"labels": labels},
    )()


@pytest.mark.parametrize(
    "name,positioned",
    [
        ("USTCRevision2026", False),
        ("CICMalDroidRevision2026", False),
        ("MalNetImageRevision2026", True),
        ("MaleXRevision2026", True),
    ],
)
def test_index_and_batch_for_each_interface(name, positioned):
    ds = _dataset(name, positioned)

    positions = training_class_positions(ds, class_count=3)

    np.testing.assert_array_equal(positions[0], [0, 4])
    np.testing.assert_array_equal(positions[1], [1, 2, 6])
    np.testing.assert_array_equal(positions[2], [3, 5, 7])

    data = GANTrainingData(ds, class_count=3, seed=42)
    images, labels, sampled = data.draw(32)

    assert images.shape == (32, 40, 40, 3)
    assert labels.shape == (32, 3)
    np.testing.assert_array_equal(
        np.argmax(labels, axis=1), ds.y[sampled]
    )

    assert all(call[1] == "train" for call in ds.calls)


def test_identical_seeds_repeat_sampled_positions():
    first = GANTrainingData(
        _dataset("MalNetImageRevision2026", True), 3, 123
    )
    second = GANTrainingData(
        _dataset("MalNetImageRevision2026", True), 3, 123
    )

    _, _, a = first.draw(64)
    _, _, b = second.draw(64)
    np.testing.assert_array_equal(a, b)


def test_rejects_incorrect_conditioning():
    ds = _dataset("USTCRevision2026", False)
    data = GANTrainingData(ds, 3, 42)
    ds.corrupt_labels = True

    with pytest.raises(ValueError, match="disagree"):
        data.draw(16)


def test_rejects_invalid_image_range():
    ds = _dataset("CICMalDroidRevision2026", False)
    data = GANTrainingData(ds, 3, 42)
    ds.corrupt_images = True

    with pytest.raises(ValueError, match="outside"):
        data.draw(16)


def test_rejects_missing_training_class():
    ds = _dataset("MaleXRevision2026", True)

    with pytest.raises(ValueError, match="every governed class"):
        training_class_positions(ds, class_count=4)


def test_rejects_unknown_dataset_adapter():
    class UnknownDataset(_DatasetBase):
        def labels(self, split):
            return self.y

    with pytest.raises(TypeError, match="unsupported"):
        training_class_positions(UnknownDataset(), class_count=3)


def test_rejects_invalid_class_count():
    ds = _dataset("USTCRevision2026", False)

    for invalid in (0, -1, True, 2.5):
        with pytest.raises(ValueError):
            training_class_positions(ds, class_count=invalid)
