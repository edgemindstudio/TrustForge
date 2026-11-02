"""Focused tests for governed revision-2026 RBM traversal."""

from __future__ import annotations

import math

import numpy as np
import pytest

from trustforge.models.paper01_revision_2026.restricted_boltzmann_traversal import (
    BATCH_SIZE,
    CLASS_BALANCE_EXPONENT,
    RBMTraversalBatch,
    RBMTraversalSpec,
    batches_per_nominal_epoch,
    build_traversal_spec,
    nominal_epoch_draw_count,
    sample_training_position,
    training_batch,
)


def _positions():
    return {
        0: np.arange(0, 10, dtype=np.int64),
        1: np.arange(10, 30, dtype=np.int64),
        2: np.arange(30, 70, dtype=np.int64),
    }


def _spec():
    return build_traversal_spec(
        dataset_id="synthetic-audit-dataset",
        class_positions=_positions(),
    )


def test_governed_traversal_constants():
    assert CLASS_BALANCE_EXPONENT == 0.85
    assert BATCH_SIZE == 256


def test_build_traversal_spec_records_counts_and_total():
    spec = _spec()

    assert isinstance(
        spec,
        RBMTraversalSpec,
    )

    np.testing.assert_array_equal(
        spec.class_ids,
        np.array(
            [0, 1, 2],
            dtype=np.int64,
        ),
    )

    np.testing.assert_array_equal(
        spec.class_counts,
        np.array(
            [10, 20, 40],
            dtype=np.int64,
        ),
    )

    assert spec.total_training_examples == 70


def test_class_probabilities_follow_n_to_point_85_normalized():
    spec = _spec()

    weights = np.power(
        np.array(
            [10, 20, 40],
            dtype=np.float64,
        ),
        CLASS_BALANCE_EXPONENT,
    )

    expected = (
        weights
        / weights.sum(
            dtype=np.float64
        )
    )

    np.testing.assert_allclose(
        spec.class_probabilities,
        expected,
        rtol=0.0,
        atol=1.0e-15,
    )

    assert math.isclose(
        float(
            spec.class_probabilities.sum(
                dtype=np.float64
            )
        ),
        1.0,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    )


def test_nominal_epoch_draw_count_equals_n_train():
    spec = _spec()

    assert nominal_epoch_draw_count(
        spec
    ) == 70


def test_batches_per_nominal_epoch_is_ceil_n_over_256():
    spec = _spec()

    assert batches_per_nominal_epoch(
        spec
    ) == 1

    large_positions = {
        0: np.arange(
            0,
            300,
            dtype=np.int64,
        ),
        1: np.arange(
            300,
            600,
            dtype=np.int64,
        ),
    }

    large_spec = build_traversal_spec(
        dataset_id="large",
        class_positions=large_positions,
    )

    assert batches_per_nominal_epoch(
        large_spec
    ) == math.ceil(
        600 / 256
    )


def test_same_identity_and_training_position_reproduce_same_draw():
    positions = _positions()
    spec = _spec()

    first = sample_training_position(
        spec=spec,
        class_positions=positions,
        experimental_seed=42,
        training_position=12345,
    )

    second = sample_training_position(
        spec=spec,
        class_positions=positions,
        experimental_seed=42,
        training_position=12345,
    )

    assert first == second


def test_changed_seed_can_change_draw():
    positions = _positions()
    spec = _spec()

    first = [
        sample_training_position(
            spec=spec,
            class_positions=positions,
            experimental_seed=42,
            training_position=position,
        )
        for position in range(128)
    ]

    second = [
        sample_training_position(
            spec=spec,
            class_positions=positions,
            experimental_seed=43,
            training_position=position,
        )
        for position in range(128)
    ]

    assert first != second


def test_changed_dataset_identity_can_change_draw():
    positions = _positions()

    first_spec = build_traversal_spec(
        dataset_id="dataset-A",
        class_positions=positions,
    )

    second_spec = build_traversal_spec(
        dataset_id="dataset-B",
        class_positions=positions,
    )

    first = [
        sample_training_position(
            spec=first_spec,
            class_positions=positions,
            experimental_seed=42,
            training_position=position,
        )
        for position in range(128)
    ]

    second = [
        sample_training_position(
            spec=second_spec,
            class_positions=positions,
            experimental_seed=42,
            training_position=position,
        )
        for position in range(128)
    ]

    assert first != second


def test_changed_training_position_can_change_draw():
    positions = _positions()
    spec = _spec()

    draws = {
        sample_training_position(
            spec=spec,
            class_positions=positions,
            experimental_seed=42,
            training_position=position,
        )
        for position in range(128)
    }

    assert len(draws) > 1


def test_sampled_logical_position_belongs_to_selected_class():
    positions = _positions()
    spec = _spec()

    for training_position in range(
        512
    ):
        class_id, logical_position = sample_training_position(
            spec=spec,
            class_positions=positions,
            experimental_seed=42,
            training_position=training_position,
        )

        assert logical_position in positions[
            class_id
        ]


def test_sampling_is_with_replacement():
    positions = {
        0: np.array(
            [5],
            dtype=np.int64,
        ),
    }

    spec = build_traversal_spec(
        dataset_id="replacement-audit",
        class_positions=positions,
    )

    draws = [
        sample_training_position(
            spec=spec,
            class_positions=positions,
            experimental_seed=42,
            training_position=position,
        )
        for position in range(16)
    ]

    assert draws == [
        (0, 5)
    ] * 16


def test_training_batch_contains_consecutive_scientific_positions():
    positions = _positions()
    spec = _spec()

    batch = training_batch(
        spec=spec,
        class_positions=positions,
        experimental_seed=42,
        start_training_position=1000,
        batch_size=16,
    )

    assert isinstance(
        batch,
        RBMTraversalBatch,
    )

    np.testing.assert_array_equal(
        batch.training_positions,
        np.arange(
            1000,
            1016,
            dtype=np.int64,
        ),
    )


def test_training_batch_matches_independent_position_draws():
    positions = _positions()
    spec = _spec()

    batch = training_batch(
        spec=spec,
        class_positions=positions,
        experimental_seed=42,
        start_training_position=1000,
        batch_size=32,
    )

    independent = [
        sample_training_position(
            spec=spec,
            class_positions=positions,
            experimental_seed=42,
            training_position=int(
                position
            ),
        )
        for position in batch.training_positions
    ]

    np.testing.assert_array_equal(
        batch.class_ids,
        np.array(
            [
                item[0]
                for item in independent
            ],
            dtype=np.int64,
        ),
    )

    np.testing.assert_array_equal(
        batch.logical_positions,
        np.array(
            [
                item[1]
                for item in independent
            ],
            dtype=np.int64,
        ),
    )


def test_training_batch_default_size_is_256():
    positions = _positions()
    spec = _spec()

    batch = training_batch(
        spec=spec,
        class_positions=positions,
        experimental_seed=42,
        start_training_position=0,
    )

    assert batch.training_positions.shape == (
        256,
    )
    assert batch.class_ids.shape == (
        256,
    )
    assert batch.logical_positions.shape == (
        256,
    )


@pytest.mark.parametrize(
    "bad_size",
    [
        0,
        -1,
        True,
        257,
        1.5,
    ],
)
def test_training_batch_rejects_invalid_batch_size(
    bad_size,
):
    with pytest.raises(
        (TypeError, ValueError)
    ):
        training_batch(
            spec=_spec(),
            class_positions=_positions(),
            experimental_seed=42,
            start_training_position=0,
            batch_size=bad_size,
        )


@pytest.mark.parametrize(
    "bad_seed",
    [
        -1,
        True,
        1.5,
        2**32,
    ],
)
def test_sampling_rejects_invalid_experimental_seed(
    bad_seed,
):
    with pytest.raises(
        (TypeError, ValueError)
    ):
        sample_training_position(
            spec=_spec(),
            class_positions=_positions(),
            experimental_seed=bad_seed,
            training_position=0,
        )


@pytest.mark.parametrize(
    "bad_position",
    [
        -1,
        True,
        1.5,
    ],
)
def test_sampling_rejects_invalid_training_position(
    bad_position,
):
    with pytest.raises(
        (TypeError, ValueError)
    ):
        sample_training_position(
            spec=_spec(),
            class_positions=_positions(),
            experimental_seed=42,
            training_position=bad_position,
        )


@pytest.mark.parametrize(
    "bad_dataset_id",
    [
        "",
        "   ",
        42,
        None,
    ],
)
def test_build_spec_rejects_invalid_dataset_id(
    bad_dataset_id,
):
    with pytest.raises(
        (TypeError, ValueError)
    ):
        build_traversal_spec(
            dataset_id=bad_dataset_id,
            class_positions=_positions(),
        )


def test_build_spec_requires_nonempty_class_mapping():
    with pytest.raises(ValueError):
        build_traversal_spec(
            dataset_id="dataset",
            class_positions={},
        )


def test_build_spec_requires_contiguous_class_universe():
    with pytest.raises(ValueError):
        build_traversal_spec(
            dataset_id="dataset",
            class_positions={
                0: np.array(
                    [0, 1],
                    dtype=np.int64,
                ),
                2: np.array(
                    [2, 3],
                    dtype=np.int64,
                ),
            },
        )


def test_build_spec_rejects_empty_class():
    with pytest.raises(ValueError):
        build_traversal_spec(
            dataset_id="dataset",
            class_positions={
                0: np.array(
                    [],
                    dtype=np.int64,
                ),
            },
        )


def test_build_spec_rejects_noninteger_positions():
    with pytest.raises(TypeError):
        build_traversal_spec(
            dataset_id="dataset",
            class_positions={
                0: np.array(
                    [0.0, 1.0],
                    dtype=np.float64,
                ),
            },
        )


def test_build_spec_rejects_negative_positions():
    with pytest.raises(ValueError):
        build_traversal_spec(
            dataset_id="dataset",
            class_positions={
                0: np.array(
                    [-1, 0],
                    dtype=np.int64,
                ),
            },
        )


def test_build_spec_rejects_duplicate_positions_within_class():
    with pytest.raises(ValueError):
        build_traversal_spec(
            dataset_id="dataset",
            class_positions={
                0: np.array(
                    [1, 1],
                    dtype=np.int64,
                ),
            },
        )


def test_build_spec_rejects_cross_class_position_overlap():
    with pytest.raises(ValueError):
        build_traversal_spec(
            dataset_id="dataset",
            class_positions={
                0: np.array(
                    [0, 1],
                    dtype=np.int64,
                ),
                1: np.array(
                    [1, 2],
                    dtype=np.int64,
                ),
            },
        )


def test_sampling_rejects_inventory_count_mismatch_against_spec():
    spec = _spec()

    changed_positions = {
        0: np.arange(
            0,
            9,
            dtype=np.int64,
        ),
        1: np.arange(
            10,
            30,
            dtype=np.int64,
        ),
        2: np.arange(
            30,
            70,
            dtype=np.int64,
        ),
    }

    with pytest.raises(ValueError):
        sample_training_position(
            spec=spec,
            class_positions=changed_positions,
            experimental_seed=42,
            training_position=0,
        )
