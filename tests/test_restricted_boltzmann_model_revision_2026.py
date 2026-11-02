"""Focused tests for the governed revision-2026 shared conditional RBM."""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from trustforge.models.paper01_revision_2026.restricted_boltzmann_model import (
    CANONICAL_CHANNELS,
    CANONICAL_HEIGHT,
    CANONICAL_WIDTH,
    HIDDEN_UNITS,
    STATE_DTYPE,
    VISIBLE_DIM,
    SharedConditionalRBM,
    canonical_images_to_vectors,
    validate_class_ids,
)


def _zero_model(
    class_count: int = 3,
) -> SharedConditionalRBM:
    return SharedConditionalRBM(
        W=np.zeros(
            (VISIBLE_DIM, HIDDEN_UNITS),
            dtype=np.float64,
        ),
        a=np.zeros(
            (VISIBLE_DIM,),
            dtype=np.float64,
        ),
        b=np.zeros(
            (HIDDEN_UNITS,),
            dtype=np.float64,
        ),
        U=np.zeros(
            (class_count, HIDDEN_UNITS),
            dtype=np.float64,
        ),
    )


def test_governed_architecture_constants():
    assert CANONICAL_HEIGHT == 40
    assert CANONICAL_WIDTH == 40
    assert CANONICAL_CHANNELS == 3
    assert VISIBLE_DIM == 4800
    assert HIDDEN_UNITS == 128
    assert STATE_DTYPE == tf.float64


def test_model_state_shapes_and_dtypes():
    model = _zero_model(
        class_count=5
    )

    state = model.state_dict()

    assert set(state) == {"W", "a", "b", "U"}
    assert state["W"].shape == (4800, 128)
    assert state["a"].shape == (4800,)
    assert state["b"].shape == (128,)
    assert state["U"].shape == (5, 128)

    for value in state.values():
        assert value.dtype == np.float64
        assert np.all(np.isfinite(value))


def test_class_count_matches_U_rows():
    model = _zero_model(
        class_count=7
    )
    assert model.class_count == 7


def test_hidden_probabilities_use_class_interaction_row():
    W = np.zeros(
        (VISIBLE_DIM, HIDDEN_UNITS),
        dtype=np.float64,
    )
    a = np.zeros(
        (VISIBLE_DIM,),
        dtype=np.float64,
    )
    b = np.zeros(
        (HIDDEN_UNITS,),
        dtype=np.float64,
    )
    U = np.zeros(
        (3, HIDDEN_UNITS),
        dtype=np.float64,
    )

    U[1, 0] = 2.0
    U[2, 0] = -2.0

    model = SharedConditionalRBM(
        W=W,
        a=a,
        b=b,
        U=U,
    )

    visible = tf.zeros(
        (3, VISIBLE_DIM),
        dtype=tf.float64,
    )
    classes = tf.constant(
        [0, 1, 2],
        dtype=tf.int64,
    )

    probabilities = model.hidden_probabilities(
        visible,
        classes,
    ).numpy()

    np.testing.assert_allclose(
        probabilities[:, 0],
        [
            0.5,
            1.0 / (1.0 + np.exp(-2.0)),
            1.0 / (1.0 + np.exp(2.0)),
        ],
        rtol=0.0,
        atol=1.0e-12,
    )


def test_visible_natural_parameter_is_class_independent_given_hidden():
    W = np.zeros(
        (VISIBLE_DIM, HIDDEN_UNITS),
        dtype=np.float64,
    )
    W[0, 0] = 2.0

    a = np.zeros(
        (VISIBLE_DIM,),
        dtype=np.float64,
    )
    a[0] = 0.25

    model = SharedConditionalRBM(
        W=W,
        a=a,
        b=np.zeros(
            (HIDDEN_UNITS,),
            dtype=np.float64,
        ),
        U=np.ones(
            (4, HIDDEN_UNITS),
            dtype=np.float64,
        ),
    )

    hidden = tf.zeros(
        (2, HIDDEN_UNITS),
        dtype=tf.float64,
    )
    hidden = tf.tensor_scatter_nd_update(
        hidden,
        indices=[[0, 0], [1, 0]],
        updates=tf.constant(
            [1.0, 1.0],
            dtype=tf.float64,
        ),
    )

    theta = model.visible_natural_parameter(
        hidden
    ).numpy()

    np.testing.assert_allclose(
        theta[0],
        theta[1],
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        theta[:, 0],
        [2.25, 2.25],
        rtol=0.0,
        atol=1.0e-15,
    )


def test_hidden_sampling_uses_caller_supplied_uniform_draws():
    model = _zero_model(
        class_count=2
    )

    visible = tf.zeros(
        (2, VISIBLE_DIM),
        dtype=tf.float64,
    )
    classes = tf.constant(
        [0, 1],
        dtype=tf.int64,
    )

    draws = tf.concat(
        [
            tf.zeros(
                (1, HIDDEN_UNITS),
                dtype=tf.float64,
            ),
            tf.ones(
                (1, HIDDEN_UNITS),
                dtype=tf.float64,
            ),
        ],
        axis=0,
    )

    hidden = model.sample_hidden_from_uniform(
        visible,
        classes,
        draws,
    ).numpy()

    assert np.all(hidden[0] == 1.0)
    assert np.all(hidden[1] == 0.0)


def test_hidden_sampling_output_is_binary_float64():
    model = _zero_model()

    hidden = model.sample_hidden_from_uniform(
        tf.zeros(
            (2, VISIBLE_DIM),
            dtype=tf.float64,
        ),
        tf.constant(
            [0, 1],
            dtype=tf.int64,
        ),
        tf.fill(
            (2, HIDDEN_UNITS),
            tf.constant(
                0.5,
                dtype=tf.float64,
            ),
        ),
    )

    assert hidden.dtype == tf.float64
    values = hidden.numpy()
    assert np.all(
        (values == 0.0)
        | (values == 1.0)
    )


def test_visible_sampling_uses_governed_observation_model():
    model = _zero_model()

    hidden = tf.zeros(
        (2, HIDDEN_UNITS),
        dtype=tf.float64,
    )
    draws = tf.fill(
        (2, VISIBLE_DIM),
        tf.constant(
            0.5,
            dtype=tf.float64,
        ),
    )

    visible = model.sample_visible_from_uniform(
        hidden,
        draws,
    ).numpy()

    assert np.all(np.isfinite(visible))
    assert np.all(visible >= 0.0)
    assert np.all(visible <= 1.0)

    np.testing.assert_allclose(
        visible,
        0.5,
        rtol=0.0,
        atol=1.0e-15,
    )


def test_visible_mean_given_hidden_zero_state_is_half():
    model = _zero_model()

    hidden = tf.zeros(
        (2, HIDDEN_UNITS),
        dtype=tf.float64,
    )

    visible_mean = model.visible_mean_given_hidden(
        hidden
    ).numpy()

    np.testing.assert_allclose(
        visible_mean,
        0.5,
        rtol=0.0,
        atol=1.0e-15,
    )


def test_canonical_images_convert_to_float64_vectors():
    images = np.zeros(
        (2, 40, 40, 3),
        dtype=np.float32,
    )
    images[1] = 1.0

    vectors = canonical_images_to_vectors(
        images
    )

    assert vectors.shape == (2, 4800)
    assert vectors.dtype == tf.float64
    assert np.all(vectors.numpy()[0] == 0.0)
    assert np.all(vectors.numpy()[1] == 1.0)


def test_canonical_images_require_float32():
    with pytest.raises(
        (TypeError, tf.errors.InvalidArgumentError)
    ):
        canonical_images_to_vectors(
            np.zeros(
                (1, 40, 40, 3),
                dtype=np.float64,
            )
        )


@pytest.mark.parametrize(
    "shape",
    [
        (40, 40, 3),
        (1, 39, 40, 3),
        (1, 40, 39, 3),
        (1, 40, 40, 1),
    ],
)
def test_canonical_images_reject_wrong_shapes(shape):
    with pytest.raises(
        (ValueError, tf.errors.InvalidArgumentError)
    ):
        canonical_images_to_vectors(
            np.zeros(
                shape,
                dtype=np.float32,
            )
        )


@pytest.mark.parametrize(
    "bad_value",
    [-1.0e-6, 1.0 + 1.0e-6],
)
def test_canonical_images_reject_out_of_range_values(bad_value):
    images = np.zeros(
        (1, 40, 40, 3),
        dtype=np.float32,
    )
    images[0, 0, 0, 0] = bad_value

    with pytest.raises(
        tf.errors.InvalidArgumentError
    ):
        canonical_images_to_vectors(
            images
        )


def test_validate_class_ids_accepts_int32_and_int64():
    for dtype in (
        tf.int32,
        tf.int64,
    ):
        labels = validate_class_ids(
            tf.constant(
                [0, 1, 2],
                dtype=dtype,
            ),
            class_count=3,
        )

        assert labels.dtype == tf.int64
        np.testing.assert_array_equal(
            labels.numpy(),
            [0, 1, 2],
        )


@pytest.mark.parametrize(
    "labels",
    [
        [-1, 0],
        [0, 3],
    ],
)
def test_validate_class_ids_rejects_out_of_universe(labels):
    with pytest.raises(
        tf.errors.InvalidArgumentError
    ):
        validate_class_ids(
            tf.constant(
                labels,
                dtype=tf.int64,
            ),
            class_count=3,
        )


def test_validate_class_ids_rejects_float_labels():
    with pytest.raises(TypeError):
        validate_class_ids(
            tf.constant(
                [0.0, 1.0],
                dtype=tf.float64,
            ),
            class_count=2,
        )


def test_model_rejects_wrong_W_shape():
    with pytest.raises(
        tf.errors.InvalidArgumentError
    ):
        SharedConditionalRBM(
            W=np.zeros(
                (VISIBLE_DIM - 1, HIDDEN_UNITS),
                dtype=np.float64,
            ),
            a=np.zeros(
                (VISIBLE_DIM,),
                dtype=np.float64,
            ),
            b=np.zeros(
                (HIDDEN_UNITS,),
                dtype=np.float64,
            ),
            U=np.zeros(
                (2, HIDDEN_UNITS),
                dtype=np.float64,
            ),
        )


def test_model_rejects_wrong_U_hidden_width():
    with pytest.raises(
        tf.errors.InvalidArgumentError
    ):
        SharedConditionalRBM(
            W=np.zeros(
                (VISIBLE_DIM, HIDDEN_UNITS),
                dtype=np.float64,
            ),
            a=np.zeros(
                (VISIBLE_DIM,),
                dtype=np.float64,
            ),
            b=np.zeros(
                (HIDDEN_UNITS,),
                dtype=np.float64,
            ),
            U=np.zeros(
                (2, HIDDEN_UNITS - 1),
                dtype=np.float64,
            ),
        )


def test_model_rejects_nonfinite_state():
    W = np.zeros(
        (VISIBLE_DIM, HIDDEN_UNITS),
        dtype=np.float64,
    )
    W[0, 0] = np.nan

    with pytest.raises(
        tf.errors.InvalidArgumentError
    ):
        SharedConditionalRBM(
            W=W,
            a=np.zeros(
                (VISIBLE_DIM,),
                dtype=np.float64,
            ),
            b=np.zeros(
                (HIDDEN_UNITS,),
                dtype=np.float64,
            ),
            U=np.zeros(
                (2, HIDDEN_UNITS),
                dtype=np.float64,
            ),
        )


def test_state_dict_returns_independent_copies():
    model = _zero_model()

    state = model.state_dict()
    state["W"][0, 0] = 123.0

    assert float(
        model.W[0, 0].numpy()
    ) == 0.0
