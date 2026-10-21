"""Regression tests for the governed Paper 1 revision-2026 USTC adapter."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from trustforge.data.paper01_revision_2026.ustc import (
    USTCRevision2026,
)


def _write_tiny_ustc_source(root: Path) -> Path:
    """Create a tiny provider-style USTC NCHW source tree."""
    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 10 provider-training samples/class:
    # governed Case 2 with 10% validation => 9 train + 1 val/class.
    #
    # 2 provider-test samples/class:
    # provider test must be preserved unchanged.
    train_per_class = 10
    test_per_class = 2
    num_classes = 9

    train_images = []
    train_labels = []

    test_images = []
    test_labels = []

    for class_id in range(num_classes):
        for sample_id in range(train_per_class):
            value = (
                class_id * train_per_class
                + sample_id
            ) / 100.0

            image = np.full(
                (1, 40, 40),
                value,
                dtype=np.float64,
            )

            train_images.append(image)
            train_labels.append(class_id)

        for sample_id in range(test_per_class):
            value = (
                class_id * test_per_class
                + sample_id
            ) / 100.0

            image = np.full(
                (1, 40, 40),
                value,
                dtype=np.float64,
            )

            test_images.append(image)
            test_labels.append(class_id)

    np.save(
        root / "train_data.npy",
        np.stack(
            train_images,
            axis=0,
        ),
    )

    np.save(
        root / "train_labels.npy",
        np.asarray(
            train_labels,
            dtype=np.int32,
        ),
    )

    np.save(
        root / "test_data.npy",
        np.stack(
            test_images,
            axis=0,
        ),
    )

    np.save(
        root / "test_labels.npy",
        np.asarray(
            test_labels,
            dtype=np.int32,
        ),
    )

    return root


def test_ustc_revision_case2_partition_counts(tmp_path):
    source = _write_tiny_ustc_source(
        tmp_path / "ustc"
    )

    dataset = USTCRevision2026(
        source,
    )

    assert dataset.split_size("train") == 81
    assert dataset.split_size("validation") == 9
    assert dataset.split_size("test") == 18

    expected = {
        "train": 9,
        "validation": 1,
        "test": 2,
    }

    for split, per_class in expected.items():
        labels = dataset.labels(split)

        values, counts = np.unique(
            labels,
            return_counts=True,
        )

        assert np.array_equal(
            values,
            np.arange(
                9,
                dtype=np.int32,
            ),
        )

        assert np.all(
            counts == per_class
        )


def test_ustc_revision_preserves_provider_test(tmp_path):
    source = _write_tiny_ustc_source(
        tmp_path / "ustc"
    )

    dataset = USTCRevision2026(
        source,
    )

    assert np.array_equal(
        dataset.partitions.test,
        np.arange(
            18,
            dtype=np.int64,
        ),
    )

    assert np.array_equal(
        dataset.labels("test"),
        dataset.arrays.y_test,
    )


def test_ustc_revision_partition_is_deterministic(tmp_path):
    source = _write_tiny_ustc_source(
        tmp_path / "ustc"
    )

    a = USTCRevision2026(
        source,
        partition_seed=42,
    )

    b = USTCRevision2026(
        source,
        partition_seed=42,
    )

    assert np.array_equal(
        a.partitions.train,
        b.partitions.train,
    )

    assert np.array_equal(
        a.partitions.validation,
        b.partitions.validation,
    )

    assert np.array_equal(
        a.partitions.test,
        b.partitions.test,
    )


def test_ustc_revision_source_arrays_are_memory_mapped(tmp_path):
    source = _write_tiny_ustc_source(
        tmp_path / "ustc"
    )

    dataset = USTCRevision2026(
        source,
    )

    assert isinstance(
        dataset.arrays.x_train,
        np.memmap,
    )

    assert isinstance(
        dataset.arrays.y_train,
        np.memmap,
    )

    assert isinstance(
        dataset.arrays.x_test,
        np.memmap,
    )

    assert isinstance(
        dataset.arrays.y_test,
        np.memmap,
    )


def test_ustc_revision_canonical_batch_contract(tmp_path):
    source = _write_tiny_ustc_source(
        tmp_path / "ustc"
    )

    dataset = USTCRevision2026(
        source,
    )

    positions = np.array(
        [0, 1, 2, 3],
        dtype=np.int64,
    )

    images, labels = dataset.canonical_batch(
        "train",
        positions,
    )

    assert images.shape == (4, 40, 40, 3)
    assert images.dtype == np.float32
    assert np.isfinite(images).all()
    assert images.min() >= 0.0
    assert images.max() <= 1.0

    assert labels.shape == (4, 9)
    assert labels.dtype == np.float32

    # USTC grayscale must be replicated exactly across RGB channels.
    assert np.array_equal(
        images[..., 0],
        images[..., 1],
    )

    assert np.array_equal(
        images[..., 1],
        images[..., 2],
    )


def test_ustc_revision_rejects_out_of_bounds_batch_position(tmp_path):
    source = _write_tiny_ustc_source(
        tmp_path / "ustc"
    )

    dataset = USTCRevision2026(
        source,
    )

    invalid = np.array(
        [dataset.split_size("train")],
        dtype=np.int64,
    )

    try:
        dataset.canonical_batch(
            "train",
            invalid,
        )
    except IndexError:
        return

    raise AssertionError(
        "out-of-range logical batch position was not rejected"
    )
