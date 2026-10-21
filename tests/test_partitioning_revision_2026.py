"""Regression tests for the TrustForge revision-2026 partition contract."""

import numpy as np

from common.partitioning import (
    DEFAULT_CASE2_VAL_FRACTION,
    DEFAULT_CASE3_RATIOS,
    DEFAULT_PARTITION_SEED,
    derive_validation_from_train,
    preserve_provider_partitions,
    split_unsplit_corpus,
)


def _counts(y):
    values, counts = np.unique(y, return_counts=True)
    return dict(zip(values.tolist(), counts.tolist()))


def test_governed_defaults():
    assert DEFAULT_PARTITION_SEED == 42
    assert DEFAULT_CASE2_VAL_FRACTION == 0.10
    assert DEFAULT_CASE3_RATIOS == (0.70, 0.10, 0.20)


def test_case1_preserves_provider_partitions_exactly():
    x_train = np.arange(12).reshape(6, 2)
    y_train = np.array([0, 0, 0, 1, 1, 1])

    x_val = np.arange(8).reshape(4, 2)
    y_val = np.array([0, 0, 1, 1])

    x_test = np.arange(8).reshape(4, 2)
    y_test = np.array([0, 0, 1, 1])

    result = preserve_provider_partitions(
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
    )

    assert np.array_equal(result.x_train, x_train)
    assert np.array_equal(result.y_train, y_train)
    assert np.array_equal(result.x_val, x_val)
    assert np.array_equal(result.y_val, y_val)
    assert np.array_equal(result.x_test, x_test)
    assert np.array_equal(result.y_test, y_test)


def test_case2_derives_validation_from_train_and_preserves_test():
    x_train = np.arange(200).reshape(100, 2)
    y_train = np.array([0] * 50 + [1] * 50)

    x_test = np.arange(40).reshape(20, 2)
    y_test = np.array([0] * 10 + [1] * 10)

    result = derive_validation_from_train(
        x_train,
        y_train,
        x_test,
        y_test,
    )

    assert len(result.x_train) == 90
    assert len(result.x_val) == 10
    assert len(result.x_test) == 20

    assert _counts(result.y_train) == {0: 45, 1: 45}
    assert _counts(result.y_val) == {0: 5, 1: 5}

    assert np.array_equal(result.x_test, x_test)
    assert np.array_equal(result.y_test, y_test)


def test_case2_is_deterministic():
    x_train = np.arange(200).reshape(100, 2)
    y_train = np.array([0] * 50 + [1] * 50)

    x_test = np.arange(40).reshape(20, 2)
    y_test = np.array([0] * 10 + [1] * 10)

    first = derive_validation_from_train(
        x_train,
        y_train,
        x_test,
        y_test,
    )
    second = derive_validation_from_train(
        x_train,
        y_train,
        x_test,
        y_test,
    )

    assert np.array_equal(first.x_train, second.x_train)
    assert np.array_equal(first.y_train, second.y_train)
    assert np.array_equal(first.x_val, second.x_val)
    assert np.array_equal(first.y_val, second.y_val)


def test_case3_creates_stratified_70_10_20_partitions():
    x = np.arange(200).reshape(100, 2)
    y = np.array([0] * 50 + [1] * 50)

    result = split_unsplit_corpus(x, y)

    assert len(result.x_train) == 70
    assert len(result.x_val) == 10
    assert len(result.x_test) == 20

    assert _counts(result.y_train) == {0: 35, 1: 35}
    assert _counts(result.y_val) == {0: 5, 1: 5}
    assert _counts(result.y_test) == {0: 10, 1: 10}


def test_case3_is_deterministic_and_non_overlapping():
    # First column provides a unique sample identity for this test.
    x = np.column_stack((np.arange(100), np.arange(100) + 1000))
    y = np.array([0] * 50 + [1] * 50)

    first = split_unsplit_corpus(x, y)
    second = split_unsplit_corpus(x, y)

    assert np.array_equal(first.x_train, second.x_train)
    assert np.array_equal(first.y_train, second.y_train)
    assert np.array_equal(first.x_val, second.x_val)
    assert np.array_equal(first.y_val, second.y_val)
    assert np.array_equal(first.x_test, second.x_test)
    assert np.array_equal(first.y_test, second.y_test)

    train_ids = set(first.x_train[:, 0].tolist())
    val_ids = set(first.x_val[:, 0].tolist())
    test_ids = set(first.x_test[:, 0].tolist())

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    assert train_ids | val_ids | test_ids == set(range(100))
