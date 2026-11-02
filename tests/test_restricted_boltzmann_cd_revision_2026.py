"""Focused tests for governed revision-2026 RBM CD-1."""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from trustforge.models.paper01_revision_2026.restricted_boltzmann_cd import (
    CD_K,
    RBMCD1Result,
    cd1_update_statistics,
)
from trustforge.models.paper01_revision_2026.restricted_boltzmann_initialization import (
    initialize_shared_conditional_rbm,
)
from trustforge.models.paper01_revision_2026.restricted_boltzmann_model import (
    HIDDEN_UNITS,
    VISIBLE_DIM,
)


def _model(
    class_count: int = 3,
):
    model, _ = initialize_shared_conditional_rbm(
        class_count=class_count,
        experimental_seed=42,
        training_scalar_mean=0.4,
    )
    return model


def _visible_batch():
    return np.stack(
        [
            np.full(
                (VISIBLE_DIM,),
                0.20,
                dtype=np.float64,
            ),
            np.full(
                (VISIBLE_DIM,),
                0.50,
                dtype=np.float64,
            ),
            np.full(
                (VISIBLE_DIM,),
                0.80,
                dtype=np.float64,
            ),
            np.linspace(
                0.0,
                1.0,
                VISIBLE_DIM,
                dtype=np.float64,
            ),
        ],
        axis=0,
    )


def _labels():
    return np.array(
        [0, 1, 2, 1],
        dtype=np.int64,
    )


def _result(
    *,
    seed: int = 42,
    update_index: int = 7,
    dataset_id: str = "synthetic-cd-audit",
):
    return cd1_update_statistics(
        model=_model(),
        visible_batch=_visible_batch(),
        class_ids=_labels(),
        dataset_id=dataset_id,
        experimental_seed=seed,
        update_index=update_index,
    )


def test_cd_depth_is_exactly_one():
    assert CD_K == 1


def test_cd_result_type_and_shapes():
    result = _result()

    assert isinstance(
        result,
        RBMCD1Result,
    )

    assert result.positive_hidden_probabilities.shape == (
        4,
        HIDDEN_UNITS,
    )
    assert result.sampled_hidden_state.shape == (
        4,
        HIDDEN_UNITS,
    )
    assert result.negative_visible_state.shape == (
        4,
        VISIBLE_DIM,
    )
    assert result.negative_hidden_probabilities.shape == (
        4,
        HIDDEN_UNITS,
    )

    assert result.delta_W.shape == (
        VISIBLE_DIM,
        HIDDEN_UNITS,
    )
    assert result.delta_a.shape == (
        VISIBLE_DIM,
    )
    assert result.delta_b.shape == (
        HIDDEN_UNITS,
    )
    assert result.delta_U.shape == (
        3,
        HIDDEN_UNITS,
    )


def test_all_cd_outputs_are_float64():
    result = _result()

    for value in (
        result.positive_hidden_probabilities,
        result.sampled_hidden_state,
        result.negative_visible_state,
        result.negative_hidden_probabilities,
        result.delta_W,
        result.delta_a,
        result.delta_b,
        result.delta_U,
    ):
        assert value.dtype == np.float64


def test_sampled_hidden_state_is_binary():
    result = _result()

    assert np.all(
        (result.sampled_hidden_state == 0.0)
        | (result.sampled_hidden_state == 1.0)
    )


def test_negative_visible_state_remains_inside_closed_unit_interval():
    result = _result()

    assert np.all(
        result.negative_visible_state >= 0.0
    )
    assert np.all(
        result.negative_visible_state <= 1.0
    )


def test_all_update_statistics_are_finite():
    result = _result()

    for value in (
        result.delta_W,
        result.delta_a,
        result.delta_b,
        result.delta_U,
    ):
        assert np.all(
            np.isfinite(value)
        )


def test_positive_hidden_statistics_are_conditional_expectations():
    model = _model()
    visible = _visible_batch()
    labels = _labels()

    result = cd1_update_statistics(
        model=model,
        visible_batch=visible,
        class_ids=labels,
        dataset_id="positive-expectation-audit",
        experimental_seed=42,
        update_index=0,
    )

    expected = model.hidden_probabilities(
        visible,
        labels,
    ).numpy()

    np.testing.assert_array_equal(
        result.positive_hidden_probabilities,
        expected,
    )


def test_terminal_negative_hidden_statistics_are_conditional_expectations():
    model = _model()
    visible = _visible_batch()
    labels = _labels()

    result = cd1_update_statistics(
        model=model,
        visible_batch=visible,
        class_ids=labels,
        dataset_id="negative-expectation-audit",
        experimental_seed=42,
        update_index=0,
    )

    expected = model.hidden_probabilities(
        result.negative_visible_state,
        labels,
    ).numpy()

    np.testing.assert_array_equal(
        result.negative_hidden_probabilities,
        expected,
    )


def test_same_governed_identity_reproduces_exact_transition_and_statistics():
    first = _result(
        seed=42,
        update_index=7,
    )
    second = _result(
        seed=42,
        update_index=7,
    )

    for first_value, second_value in (
        (
            first.sampled_hidden_state,
            second.sampled_hidden_state,
        ),
        (
            first.negative_visible_state,
            second.negative_visible_state,
        ),
        (
            first.negative_hidden_probabilities,
            second.negative_hidden_probabilities,
        ),
        (
            first.delta_W,
            second.delta_W,
        ),
        (
            first.delta_a,
            second.delta_a,
        ),
        (
            first.delta_b,
            second.delta_b,
        ),
        (
            first.delta_U,
            second.delta_U,
        ),
    ):
        np.testing.assert_array_equal(
            first_value,
            second_value,
        )


def test_changed_experimental_seed_changes_stochastic_transition():
    first = _result(
        seed=42
    )
    second = _result(
        seed=43
    )

    assert not np.array_equal(
        first.negative_visible_state,
        second.negative_visible_state,
    )


def test_changed_update_index_changes_stochastic_transition():
    first = _result(
        update_index=7
    )
    second = _result(
        update_index=8
    )

    assert not np.array_equal(
        first.negative_visible_state,
        second.negative_visible_state,
    )


def test_changed_dataset_identity_changes_stochastic_transition():
    first = _result(
        dataset_id="dataset-A"
    )
    second = _result(
        dataset_id="dataset-B"
    )

    assert not np.array_equal(
        first.negative_visible_state,
        second.negative_visible_state,
    )


def test_unused_class_rows_have_zero_delta_U():
    model = _model(
        class_count=4
    )

    visible = _visible_batch()

    labels = np.array(
        [0, 1, 1, 0],
        dtype=np.int64,
    )

    result = cd1_update_statistics(
        model=model,
        visible_batch=visible,
        class_ids=labels,
        dataset_id="class-row-audit",
        experimental_seed=42,
        update_index=0,
    )

    assert np.all(
        result.delta_U[
            2:
        ]
        == 0.0
    )


def test_class_conditioning_remains_clamped_in_negative_phase():
    model = _model(
        class_count=3
    )

    visible = np.full(
        (
            3,
            VISIBLE_DIM,
        ),
        0.5,
        dtype=np.float64,
    )

    labels = np.array(
        [0, 1, 2],
        dtype=np.int64,
    )

    result = cd1_update_statistics(
        model=model,
        visible_batch=visible,
        class_ids=labels,
        dataset_id="class-clamp-audit",
        experimental_seed=42,
        update_index=0,
    )

    expected_negative_hidden = model.hidden_probabilities(
        result.negative_visible_state,
        labels,
    ).numpy()

    np.testing.assert_array_equal(
        result.negative_hidden_probabilities,
        expected_negative_hidden,
    )


def test_delta_a_is_positive_minus_negative_batch_mean():
    visible = _visible_batch()

    result = cd1_update_statistics(
        model=_model(),
        visible_batch=visible,
        class_ids=_labels(),
        dataset_id="delta-a-audit",
        experimental_seed=42,
        update_index=0,
    )

    expected = (
        visible.mean(
            axis=0,
            dtype=np.float64,
        )
        - result.negative_visible_state.mean(
            axis=0,
            dtype=np.float64,
        )
    )

    np.testing.assert_allclose(
        result.delta_a,
        expected,
        rtol=0.0,
        atol=1.0e-14,
    )


def test_delta_b_is_positive_minus_negative_hidden_batch_mean():
    result = _result()

    expected = (
        result.positive_hidden_probabilities.mean(
            axis=0,
            dtype=np.float64,
        )
        - result.negative_hidden_probabilities.mean(
            axis=0,
            dtype=np.float64,
        )
    )

    np.testing.assert_allclose(
        result.delta_b,
        expected,
        rtol=0.0,
        atol=1.0e-14,
    )


def test_delta_W_is_positive_minus_negative_cross_statistic():
    visible = _visible_batch()

    result = cd1_update_statistics(
        model=_model(),
        visible_batch=visible,
        class_ids=_labels(),
        dataset_id="delta-W-audit",
        experimental_seed=42,
        update_index=0,
    )

    batch_size = visible.shape[0]

    expected = (
        visible.T
        @ result.positive_hidden_probabilities
        - result.negative_visible_state.T
        @ result.negative_hidden_probabilities
    ) / float(
        batch_size
    )

    np.testing.assert_allclose(
        result.delta_W,
        expected,
        rtol=0.0,
        atol=1.0e-13,
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
def test_cd_rejects_invalid_experimental_seed(
    bad_seed,
):
    with pytest.raises(
        (TypeError, ValueError)
    ):
        cd1_update_statistics(
            model=_model(),
            visible_batch=_visible_batch(),
            class_ids=_labels(),
            dataset_id="dataset",
            experimental_seed=bad_seed,
            update_index=0,
        )


@pytest.mark.parametrize(
    "bad_update_index",
    [
        -1,
        True,
        1.5,
    ],
)
def test_cd_rejects_invalid_update_index(
    bad_update_index,
):
    with pytest.raises(
        (TypeError, ValueError)
    ):
        cd1_update_statistics(
            model=_model(),
            visible_batch=_visible_batch(),
            class_ids=_labels(),
            dataset_id="dataset",
            experimental_seed=42,
            update_index=bad_update_index,
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
def test_cd_rejects_invalid_dataset_id(
    bad_dataset_id,
):
    with pytest.raises(
        (TypeError, ValueError)
    ):
        cd1_update_statistics(
            model=_model(),
            visible_batch=_visible_batch(),
            class_ids=_labels(),
            dataset_id=bad_dataset_id,
            experimental_seed=42,
            update_index=0,
        )


def test_cd_rejects_empty_visible_batch():
    with pytest.raises(
        (ValueError, tf.errors.InvalidArgumentError)
    ):
        cd1_update_statistics(
            model=_model(),
            visible_batch=np.empty(
                (
                    0,
                    VISIBLE_DIM,
                ),
                dtype=np.float64,
            ),
            class_ids=np.empty(
                (0,),
                dtype=np.int64,
            ),
            dataset_id="dataset",
            experimental_seed=42,
            update_index=0,
        )


def test_cd_rejects_visible_batch_wrong_width():
    with pytest.raises(
        (ValueError, tf.errors.InvalidArgumentError)
    ):
        cd1_update_statistics(
            model=_model(),
            visible_batch=np.zeros(
                (
                    2,
                    VISIBLE_DIM - 1,
                ),
                dtype=np.float64,
            ),
            class_ids=np.array(
                [0, 1],
                dtype=np.int64,
            ),
            dataset_id="dataset",
            experimental_seed=42,
            update_index=0,
        )


def test_cd_rejects_label_batch_mismatch():
    with pytest.raises(
        (ValueError, tf.errors.InvalidArgumentError)
    ):
        cd1_update_statistics(
            model=_model(),
            visible_batch=np.zeros(
                (
                    2,
                    VISIBLE_DIM,
                ),
                dtype=np.float64,
            ),
            class_ids=np.array(
                [0],
                dtype=np.int64,
            ),
            dataset_id="dataset",
            experimental_seed=42,
            update_index=0,
        )


def test_cd_rejects_out_of_universe_class_id():
    with pytest.raises(
        (ValueError, tf.errors.InvalidArgumentError)
    ):
        cd1_update_statistics(
            model=_model(
                class_count=3
            ),
            visible_batch=np.zeros(
                (
                    2,
                    VISIBLE_DIM,
                ),
                dtype=np.float64,
            ),
            class_ids=np.array(
                [0, 3],
                dtype=np.int64,
            ),
            dataset_id="dataset",
            experimental_seed=42,
            update_index=0,
        )


@pytest.mark.parametrize(
    "bad_value",
    [
        -1.0e-6,
        1.0 + 1.0e-6,
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_cd_rejects_invalid_visible_values(
    bad_value,
):
    visible = np.zeros(
        (
            2,
            VISIBLE_DIM,
        ),
        dtype=np.float64,
    )

    visible[
        0,
        0,
    ] = bad_value

    with pytest.raises(
        (ValueError, tf.errors.InvalidArgumentError)
    ):
        cd1_update_statistics(
            model=_model(),
            visible_batch=visible,
            class_ids=np.array(
                [0, 1],
                dtype=np.int64,
            ),
            dataset_id="dataset",
            experimental_seed=42,
            update_index=0,
        )
