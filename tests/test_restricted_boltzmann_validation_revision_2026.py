"""Focused tests for governed revision-2026 RBM validation."""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from trustforge.models.paper01_revision_2026.restricted_boltzmann_initialization import (
    initialize_shared_conditional_rbm,
)
from trustforge.models.paper01_revision_2026.restricted_boltzmann_model import (
    VISIBLE_DIM,
)
from trustforge.models.paper01_revision_2026.restricted_boltzmann_validation import (
    VISIBLE_TOLERANCE,
    RBMValidationDiagnostics,
    validate_reconstruction_batch,
)


def _model(
    *,
    class_count: int = 3,
):
    model, _ = initialize_shared_conditional_rbm(
        class_count=class_count,
        experimental_seed=42,
        training_scalar_mean=0.4,
    )
    return model


def _visible():
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


def test_visible_tolerance_is_governed_value():
    assert VISIBLE_TOLERANCE == 1.0e-12


def test_validation_returns_diagnostics():
    result = validate_reconstruction_batch(
        model=_model(),
        visible_batch=_visible(),
        class_ids=_labels(),
    )

    assert isinstance(
        result,
        RBMValidationDiagnostics,
    )


def test_validation_reconstruction_mse_is_finite_and_nonnegative():
    result = validate_reconstruction_batch(
        model=_model(),
        visible_batch=_visible(),
        class_ids=_labels(),
    )

    assert np.isfinite(
        result.reconstruction_mse
    )

    assert result.reconstruction_mse >= 0.0


def test_validation_records_exact_example_count():
    result = validate_reconstruction_batch(
        model=_model(),
        visible_batch=_visible(),
        class_ids=_labels(),
    )

    assert result.example_count == 4


def test_validation_hidden_probabilities_remain_inside_unit_interval():
    result = validate_reconstruction_batch(
        model=_model(),
        visible_batch=_visible(),
        class_ids=_labels(),
    )

    assert (
        0.0
        <= result.hidden_probability_min
        <= result.hidden_probability_max
        <= 1.0
    )


def test_validation_reconstructed_means_remain_inside_tolerance():
    result = validate_reconstruction_batch(
        model=_model(),
        visible_batch=_visible(),
        class_ids=_labels(),
    )

    assert result.reconstructed_min >= (
        -VISIBLE_TOLERANCE
    )

    assert result.reconstructed_max <= (
        1.0 + VISIBLE_TOLERANCE
    )


def test_validation_records_visible_range():
    visible = _visible()

    result = validate_reconstruction_batch(
        model=_model(),
        visible_batch=visible,
        class_ids=_labels(),
    )

    assert result.visible_min == 0.0
    assert result.visible_max == 1.0


def test_validation_is_deterministic():
    model = _model()
    visible = _visible()
    labels = _labels()

    first = validate_reconstruction_batch(
        model=model,
        visible_batch=visible,
        class_ids=labels,
    )

    second = validate_reconstruction_batch(
        model=model,
        visible_batch=visible,
        class_ids=labels,
    )

    assert first == second


def test_validation_does_not_mutate_model_state():
    model = _model()

    before = model.state_dict()

    validate_reconstruction_batch(
        model=model,
        visible_batch=_visible(),
        class_ids=_labels(),
    )

    after = model.state_dict()

    for name in before:
        np.testing.assert_array_equal(
            before[name],
            after[name],
        )


def test_validation_matches_direct_expectation_reconstruction_mse():
    model = _model()
    visible = _visible()
    labels = _labels()

    result = validate_reconstruction_batch(
        model=model,
        visible_batch=visible,
        class_ids=labels,
    )

    hidden_probabilities = model.hidden_probabilities(
        visible,
        labels,
    )

    reconstructed = model.visible_mean_given_hidden(
        hidden_probabilities
    ).numpy()

    expected_mse = float(
        np.mean(
            np.square(
                reconstructed
                - visible
            ),
            dtype=np.float64,
        )
    )

    assert result.reconstruction_mse == pytest.approx(
        expected_mse,
        rel=0.0,
        abs=1.0e-15,
    )


def test_validation_rejects_wrong_model_type():
    with pytest.raises(TypeError):
        validate_reconstruction_batch(
            model=object(),
            visible_batch=_visible(),
            class_ids=_labels(),
        )


def test_validation_rejects_empty_batch():
    with pytest.raises(
        (ValueError, tf.errors.InvalidArgumentError)
    ):
        validate_reconstruction_batch(
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
        )


def test_validation_rejects_wrong_visible_width():
    with pytest.raises(
        (ValueError, tf.errors.InvalidArgumentError)
    ):
        validate_reconstruction_batch(
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
        )


def test_validation_rejects_label_batch_mismatch():
    with pytest.raises(
        (ValueError, tf.errors.InvalidArgumentError)
    ):
        validate_reconstruction_batch(
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
        )


def test_validation_rejects_out_of_universe_class_id():
    with pytest.raises(
        (ValueError, tf.errors.InvalidArgumentError)
    ):
        validate_reconstruction_batch(
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
def test_validation_rejects_invalid_visible_values(
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
        validate_reconstruction_batch(
            model=_model(),
            visible_batch=visible,
            class_ids=np.array(
                [0, 1],
                dtype=np.int64,
            ),
        )


def test_validation_fails_closed_on_nonfinite_model_state():
    model = _model()

    model.W.assign(
        tf.tensor_scatter_nd_update(
            model.W,
            indices=tf.constant(
                [[0, 0]],
                dtype=tf.int32,
            ),
            updates=tf.constant(
                [np.nan],
                dtype=tf.float64,
            ),
        )
    )

    with pytest.raises(
        (ValueError, tf.errors.InvalidArgumentError)
    ):
        validate_reconstruction_batch(
            model=model,
            visible_batch=_visible(),
            class_ids=_labels(),
        )


class _ValidationAdapter:
    def __init__(
        self,
        *,
        images: np.ndarray,
        labels: np.ndarray,
    ):
        self._images = np.asarray(
            images
        )
        self._labels = np.asarray(
            labels
        )
        self.requested_splits = []
        self.requested_positions = []

    def split_size(
        self,
        split: str,
    ) -> int:
        self.requested_splits.append(
            ("split_size", split)
        )
        if split != "validation":
            raise AssertionError(
                "validation evaluator accessed non-validation split"
            )
        return int(
            self._images.shape[0]
        )

    def canonical_batch(
        self,
        split: str,
        logical_positions: np.ndarray,
    ):
        self.requested_splits.append(
            ("canonical_batch", split)
        )
        if split != "validation":
            raise AssertionError(
                "validation evaluator accessed non-validation split"
            )

        positions = np.asarray(
            logical_positions,
            dtype=np.int64,
        )

        self.requested_positions.extend(
            positions.tolist()
        )

        return (
            self._images[
                positions
            ],
            self._labels[
                positions
            ],
        )


def _validation_adapter_fixture(
    *,
    count: int = 5,
    class_count: int = 3,
):
    values = np.linspace(
        0.1,
        0.9,
        count,
        dtype=np.float32,
    )

    images = np.stack(
        [
            np.full(
                (
                    40,
                    40,
                    3,
                ),
                value,
                dtype=np.float32,
            )
            for value in values
        ],
        axis=0,
    )

    class_ids = np.arange(
        count,
        dtype=np.int64,
    ) % class_count

    labels = np.eye(
        class_count,
        dtype=np.float32,
    )[
        class_ids
    ]

    return _ValidationAdapter(
        images=images,
        labels=labels,
    )


def test_full_validation_split_visits_every_example_exactly_once():
    from trustforge.models.paper01_revision_2026.restricted_boltzmann_validation import (
        evaluate_full_validation_split,
    )

    adapter = _validation_adapter_fixture(
        count=5
    )

    result = evaluate_full_validation_split(
        model=_model(),
        adapter=adapter,
        batch_size=2,
    )

    assert result.example_count == 5
    assert adapter.requested_positions == [
        0,
        1,
        2,
        3,
        4,
    ]

    assert all(
        split == "validation"
        for _, split in adapter.requested_splits
    )


def test_full_validation_split_matches_direct_single_batch_result():
    from trustforge.models.paper01_revision_2026.restricted_boltzmann_model import (
        canonical_images_to_vectors,
    )
    from trustforge.models.paper01_revision_2026.restricted_boltzmann_validation import (
        evaluate_full_validation_split,
    )

    model = _model()
    adapter = _validation_adapter_fixture(
        count=5
    )

    streamed = evaluate_full_validation_split(
        model=model,
        adapter=adapter,
        batch_size=2,
    )

    visible = canonical_images_to_vectors(
        adapter._images
    )

    class_ids = np.argmax(
        adapter._labels,
        axis=1,
    ).astype(
        np.int64,
        copy=False,
    )

    direct = validate_reconstruction_batch(
        model=model,
        visible_batch=visible,
        class_ids=class_ids,
    )

    assert streamed.reconstruction_mse == pytest.approx(
        direct.reconstruction_mse,
        rel=0.0,
        abs=1.0e-15,
    )
    assert streamed.example_count == direct.example_count
    assert streamed.visible_min == direct.visible_min
    assert streamed.visible_max == direct.visible_max
    assert streamed.reconstructed_min == pytest.approx(
        direct.reconstructed_min,
        rel=0.0,
        abs=1.0e-15,
    )
    assert streamed.reconstructed_max == pytest.approx(
        direct.reconstructed_max,
        rel=0.0,
        abs=1.0e-15,
    )
    assert streamed.hidden_probability_min == pytest.approx(
        direct.hidden_probability_min,
        rel=0.0,
        abs=1.0e-15,
    )
    assert streamed.hidden_probability_max == pytest.approx(
        direct.hidden_probability_max,
        rel=0.0,
        abs=1.0e-15,
    )


def test_full_validation_split_does_not_mutate_model():
    from trustforge.models.paper01_revision_2026.restricted_boltzmann_validation import (
        evaluate_full_validation_split,
    )

    model = _model()
    adapter = _validation_adapter_fixture(
        count=5
    )

    before = model.state_dict()

    evaluate_full_validation_split(
        model=model,
        adapter=adapter,
        batch_size=2,
    )

    after = model.state_dict()

    for name in before:
        np.testing.assert_array_equal(
            before[name],
            after[name],
        )


def test_full_validation_split_rejects_non_float32_images():
    from trustforge.models.paper01_revision_2026.restricted_boltzmann_validation import (
        evaluate_full_validation_split,
    )

    adapter = _validation_adapter_fixture(
        count=3
    )
    adapter._images = adapter._images.astype(
        np.float64
    )

    with pytest.raises(TypeError):
        evaluate_full_validation_split(
            model=_model(),
            adapter=adapter,
            batch_size=2,
        )


def test_full_validation_split_rejects_non_float32_one_hot_labels():
    from trustforge.models.paper01_revision_2026.restricted_boltzmann_validation import (
        evaluate_full_validation_split,
    )

    adapter = _validation_adapter_fixture(
        count=3
    )
    adapter._labels = adapter._labels.astype(
        np.float64
    )

    with pytest.raises(TypeError):
        evaluate_full_validation_split(
            model=_model(),
            adapter=adapter,
            batch_size=2,
        )


def test_full_validation_split_rejects_non_exact_one_hot_labels():
    from trustforge.models.paper01_revision_2026.restricted_boltzmann_validation import (
        evaluate_full_validation_split,
    )

    adapter = _validation_adapter_fixture(
        count=3
    )

    adapter._labels = adapter._labels.copy()
    adapter._labels[
        0,
        :
    ] = np.array(
        [0.5, 0.5, 0.0],
        dtype=np.float32,
    )

    with pytest.raises(ValueError):
        evaluate_full_validation_split(
            model=_model(),
            adapter=adapter,
            batch_size=2,
        )


def test_full_validation_split_rejects_label_width_mismatch():
    from trustforge.models.paper01_revision_2026.restricted_boltzmann_validation import (
        evaluate_full_validation_split,
    )

    adapter = _validation_adapter_fixture(
        count=3,
        class_count=2,
    )

    with pytest.raises(ValueError):
        evaluate_full_validation_split(
            model=_model(
                class_count=3
            ),
            adapter=adapter,
            batch_size=2,
        )


def test_full_validation_split_rejects_empty_validation_split():
    from trustforge.models.paper01_revision_2026.restricted_boltzmann_validation import (
        evaluate_full_validation_split,
    )

    adapter = _ValidationAdapter(
        images=np.empty(
            (
                0,
                40,
                40,
                3,
            ),
            dtype=np.float32,
        ),
        labels=np.empty(
            (
                0,
                3,
            ),
            dtype=np.float32,
        ),
    )

    with pytest.raises(ValueError):
        evaluate_full_validation_split(
            model=_model(),
            adapter=adapter,
            batch_size=2,
        )


@pytest.mark.parametrize(
    "bad_batch_size",
    [
        0,
        -1,
        True,
        1.5,
    ],
)
def test_full_validation_split_rejects_invalid_batch_size(
    bad_batch_size,
):
    from trustforge.models.paper01_revision_2026.restricted_boltzmann_validation import (
        evaluate_full_validation_split,
    )

    with pytest.raises(
        (TypeError, ValueError)
    ):
        evaluate_full_validation_split(
            model=_model(),
            adapter=_validation_adapter_fixture(),
            batch_size=bad_batch_size,
        )
