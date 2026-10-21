"""Regression tests for Paper 1 revision-2026 GMM traversal."""

from __future__ import annotations

import math

import numpy as np
import pytest

from trustforge.models.paper01_revision_2026.gaussian_mixture_traversal import (
    MAX_BATCH_SIZE,
    MAX_BATCH_SIZE_SPREAD,
    balanced_batch_sizes,
    batches_per_pass,
    iter_training_batches,
)


@pytest.mark.parametrize(
    "class_count,expected_batches",
    [
        (1, 1),
        (79, 1),
        (106, 1),
        (128, 1),
        (129, 2),
        (206, 2),
        (877, 7),
        (8100, 64),
        (12221, 96),
        (125794, 983),
        (605255, 4729),
        (619118, 4837),
    ],
)
def test_batches_per_pass_matches_ceil_rule(
    class_count,
    expected_batches,
):
    assert (
        batches_per_pass(
            class_count
        )
        == expected_batches
    )

    assert expected_batches == math.ceil(
        class_count
        / 128
    )


@pytest.mark.parametrize(
    "class_count,expected_min,expected_max",
    [
        (79, 79, 79),
        (106, 106, 106),
        (206, 103, 103),
        (877, 125, 126),
        (8100, 126, 127),
        (12221, 127, 128),
        (125794, 127, 128),
        (605255, 127, 128),
        (619118, 127, 128),
    ],
)
def test_balanced_batch_geometry(
    class_count,
    expected_min,
    expected_max,
):
    sizes = balanced_batch_sizes(
        class_count
    )

    assert len(
        sizes
    ) == batches_per_pass(
        class_count
    )

    assert sum(
        sizes
    ) == class_count

    assert min(
        sizes
    ) == expected_min

    assert max(
        sizes
    ) == expected_max

    assert (
        max(
            sizes
        )
        - min(
            sizes
        )
        <= MAX_BATCH_SIZE_SPREAD
    )

    assert max(
        sizes
    ) <= MAX_BATCH_SIZE


def test_ustc_exact_balanced_geometry():
    sizes = balanced_batch_sizes(
        8100
    )

    assert len(
        sizes
    ) == 64

    assert sizes.count(
        127
    ) == 36

    assert sizes.count(
        126
    ) == 28

    assert sizes[:36] == (
        127,
    ) * 36

    assert sizes[36:] == (
        126,
    ) * 28


def test_206_class_has_two_equal_batches():
    assert balanced_batch_sizes(
        206
    ) == (
        103,
        103,
    )


def test_complete_pass_has_exact_once_coverage():
    positions = np.arange(
        877,
        dtype=np.int64,
    )

    batches = list(
        iter_training_batches(
            positions,
            experimental_seed=42,
            max_updates=batches_per_pass(
                len(
                    positions
                )
            ),
        )
    )

    concatenated = np.concatenate(
        batches
    )

    assert len(
        batches
    ) == 7

    assert concatenated.size == positions.size

    np.testing.assert_array_equal(
        np.sort(
            concatenated
        ),
        positions,
    )

    assert np.unique(
        concatenated
    ).size == positions.size


def test_same_seed_produces_identical_batches():
    positions = np.arange(
        206,
        dtype=np.int64,
    )

    batches_a = list(
        iter_training_batches(
            positions,
            experimental_seed=42,
            max_updates=8,
        )
    )

    batches_b = list(
        iter_training_batches(
            positions,
            experimental_seed=42,
            max_updates=8,
        )
    )

    assert len(
        batches_a
    ) == len(
        batches_b
    )

    for batch_a, batch_b in zip(
        batches_a,
        batches_b,
    ):
        np.testing.assert_array_equal(
            batch_a,
            batch_b,
        )


def test_different_seed_changes_traversal_order():
    positions = np.arange(
        206,
        dtype=np.int64,
    )

    first_a = next(
        iter_training_batches(
            positions,
            experimental_seed=42,
            max_updates=1,
        )
    )

    first_b = next(
        iter_training_batches(
            positions,
            experimental_seed=43,
            max_updates=1,
        )
    )

    assert not np.array_equal(
        first_a,
        first_b,
    )


def test_reshuffle_occurs_between_passes():
    positions = np.arange(
        206,
        dtype=np.int64,
    )

    batches = list(
        iter_training_batches(
            positions,
            experimental_seed=42,
            max_updates=4,
        )
    )

    first_pass = np.concatenate(
        batches[:2]
    )

    second_pass = np.concatenate(
        batches[2:4]
    )

    np.testing.assert_array_equal(
        np.sort(
            first_pass
        ),
        positions,
    )

    np.testing.assert_array_equal(
        np.sort(
            second_pass
        ),
        positions,
    )

    assert not np.array_equal(
        first_pass,
        second_pass,
    )


def test_stops_exactly_inside_later_pass():
    positions = np.arange(
        877,
        dtype=np.int64,
    )

    batches = list(
        iter_training_batches(
            positions,
            experimental_seed=42,
            max_updates=100,
        )
    )

    assert len(
        batches
    ) == 100

    sizes = balanced_batch_sizes(
        877
    )

    expected_first_two_sizes = (
        sizes[
            0
        ],
        sizes[
            1
        ],
    )

    assert (
        batches[
            98
        ].size,
        batches[
            99
        ].size,
    ) == expected_first_two_sizes


def test_exposure_spread_is_at_most_one_when_stopping_midpass():
    positions = np.arange(
        877,
        dtype=np.int64,
    )

    exposures = np.zeros(
        positions.size,
        dtype=np.int64,
    )

    for batch in iter_training_batches(
        positions,
        experimental_seed=42,
        max_updates=100,
    ):
        exposures[
            batch
        ] += 1

    assert exposures.min() == 14
    assert exposures.max() == 15

    assert (
        exposures.max()
        - exposures.min()
    ) == 1

    # With balanced 877-example geometry, the first two batches of the
    # fifteenth pass contain 126 + 126 examples.
    assert np.count_nonzero(
        exposures
        == 15
    ) == 252

    assert np.count_nonzero(
        exposures
        == 14
    ) == 625


def test_five_pass_geometry_preserves_equal_exposure():
    positions = np.arange(
        8100,
        dtype=np.int64,
    )

    updates = (
        5
        * batches_per_pass(
            positions.size
        )
    )

    exposures = np.zeros(
        positions.size,
        dtype=np.int64,
    )

    for batch in iter_training_batches(
        positions,
        experimental_seed=42,
        max_updates=updates,
    ):
        exposures[
            batch
        ] += 1

    assert updates == 320

    assert np.all(
        exposures
        == 5
    )


@pytest.mark.parametrize(
    "positions",
    [
        np.array(
            [],
            dtype=np.int64,
        ),
        np.array(
            [
                0,
                1,
                1,
            ],
            dtype=np.int64,
        ),
        np.array(
            [
                -1,
                0,
                1,
            ],
            dtype=np.int64,
        ),
    ],
)
def test_invalid_class_positions_fail_closed(
    positions,
):
    with pytest.raises(
        ValueError
    ):
        list(
            iter_training_batches(
                positions,
                experimental_seed=42,
                max_updates=1,
            )
        )


def test_noninteger_positions_fail_closed():
    with pytest.raises(
        TypeError
    ):
        list(
            iter_training_batches(
                np.array(
                    [
                        0.0,
                        1.0,
                        2.0,
                    ],
                    dtype=np.float64,
                ),
                experimental_seed=42,
                max_updates=1,
            )
        )


@pytest.mark.parametrize(
    "seed",
    [
        -1,
        2**32,
    ],
)
def test_invalid_seed_fails_closed(
    seed,
):
    with pytest.raises(
        ValueError
    ):
        list(
            iter_training_batches(
                np.arange(
                    5,
                    dtype=np.int64,
                ),
                experimental_seed=seed,
                max_updates=1,
            )
        )


@pytest.mark.parametrize(
    "max_updates",
    [
        0,
        -1,
    ],
)
def test_invalid_update_limit_fails_closed(
    max_updates,
):
    with pytest.raises(
        ValueError
    ):
        list(
            iter_training_batches(
                np.arange(
                    5,
                    dtype=np.int64,
                ),
                experimental_seed=42,
                max_updates=max_updates,
            )
        )
