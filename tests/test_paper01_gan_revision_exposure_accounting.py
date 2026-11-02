"""Tests for governed GAN train-position exposure accounting."""

import numpy as np
import pytest

from trustforge.models.paper01_revision_2026.gan_exposure_accounting import (
    GANExposureAccounting,
)


def recorder():
    return GANExposureAccounting(
        train_labels=np.array([0, 1, 0, 1, 1], dtype=np.int32),
        class_count=2,
    )


def record(accounting, positions):
    positions = np.asarray(positions, dtype=np.int64)
    labels = accounting.train_labels[positions]
    one_hot = np.eye(2, dtype=np.float32)[labels]
    accounting.record_batch(
        positions=positions, one_hot_labels=one_hot
    )


def test_exact_exposure_and_duplicate_counts():
    accounting = recorder()
    record(accounting, [0, 1, 1, 4])
    record(accounting, [0, 2])
    report = accounting.report(expected_batches=2, expected_draws=6)

    assert report["total_real_draws"] == 6
    assert report["total_unique_positions"] == 4
    assert report["per_class"][0]["draws"] == 3
    assert report["per_class"][0]["unique_positions"] == 2
    assert report["per_class"][1]["draws"] == 3
    assert report["per_class"][1]["unique_positions"] == 2
    assert report["per_class"][1]["repeated_draws"] == 1


def test_missing_batches_rejected():
    accounting = recorder()
    record(accounting, [0, 1])
    with pytest.raises(ValueError):
        accounting.report(expected_batches=2, expected_draws=2)


def test_wrong_total_rejected():
    accounting = recorder()
    record(accounting, [0, 1])
    with pytest.raises(ValueError):
        accounting.report(expected_batches=1, expected_draws=3)


@pytest.mark.parametrize("positions", [[-1], [5], []])
def test_invalid_positions_rejected(positions):
    accounting = recorder()
    positions = np.asarray(positions, dtype=np.int64)
    labels = np.zeros((len(positions), 2), dtype=np.float32)
    with pytest.raises(ValueError):
        accounting.record_batch(
            positions=positions, one_hot_labels=labels
        )


def test_mismatched_class_rejected():
    accounting = recorder()
    with pytest.raises(ValueError):
        accounting.record_batch(
            positions=np.array([0], dtype=np.int64),
            one_hot_labels=np.array([[0.0, 1.0]], dtype=np.float32),
        )


def test_non_one_hot_labels_rejected():
    accounting = recorder()
    with pytest.raises(ValueError):
        accounting.record_batch(
            positions=np.array([0], dtype=np.int64),
            one_hot_labels=np.array([[0.5, 0.5]], dtype=np.float32),
        )


def test_no_initial_exposures():
    accounting = recorder()
    result = accounting.report(expected_batches=0, expected_draws=0)
    assert result["total_unique_positions"] == 0
    assert all(row["draws"] == 0 for row in result["per_class"])


def test_position_lookup_complete_and_ordered():
    from trustforge.models.paper01_revision_2026.gan_exposure_accounting import (
        training_labels_from_class_positions,
    )

    labels = training_labels_from_class_positions(
        (
            np.array([0, 2], dtype=np.int64),
            np.array([1, 3, 4], dtype=np.int64),
        ),
        train_size=5,
        class_count=2,
    )

    np.testing.assert_array_equal(labels, [0, 1, 0, 1, 1])
    assert not labels.flags.writeable


@pytest.mark.parametrize(
    "positions",
    [
        ([0, 2], [1, 3]),
        ([0, 2], [2, 3, 4]),
        ([0, 5], [1, 3, 4]),
        ([0, 0, 2], [1, 3, 4]),
        ([], [1, 3, 4]),
    ],
)
def test_incomplete_or_invalid_position_lookup_rejected(positions):
    from trustforge.models.paper01_revision_2026.gan_exposure_accounting import (
        training_labels_from_class_positions,
    )

    with pytest.raises(ValueError):
        training_labels_from_class_positions(
            tuple(np.asarray(p, dtype=np.int64) for p in positions),
            train_size=5,
            class_count=2,
        )
