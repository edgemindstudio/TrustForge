"""Regression tests for the Paper 1 revision-2026 trainable GMM class model."""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from trustforge.models.paper01_revision_2026.gaussian_mixture_model import (
    CANONICAL_CHANNELS,
    CANONICAL_HEIGHT,
    CANONICAL_WIDTH,
    CensoredDiagonalGaussianMixture,
    canonical_images_to_vectors,
)
from trustforge.models.paper01_revision_2026.gaussian_mixture_observation import (
    CANONICAL_DIM,
    MIN_SCALE,
    NUM_COMPONENTS,
    inverse_softplus_positive,
)


def _initial_state():
    logits = np.zeros(
        (NUM_COMPONENTS,),
        dtype=np.float64,
    )

    means = np.linspace(
        0.1,
        0.9,
        NUM_COMPONENTS,
        dtype=np.float64,
    )[:, None]

    means = np.repeat(
        means,
        CANONICAL_DIM,
        axis=1,
    )

    desired_scale = np.float64(
        0.10
    )

    offset = (
        desired_scale
        - np.float64(
            MIN_SCALE
        )
    )

    raw_scalar = float(
        inverse_softplus_positive(
            tf.constant(
                offset,
                dtype=tf.float64,
            )
        ).numpy()
    )

    raw_scales = np.full(
        (
            NUM_COMPONENTS,
            CANONICAL_DIM,
        ),
        raw_scalar,
        dtype=np.float64,
    )

    return (
        logits,
        means,
        raw_scales,
    )


def _model():
    (
        logits,
        means,
        raw_scales,
    ) = _initial_state()

    return CensoredDiagonalGaussianMixture(
        logits=logits,
        means=means,
        raw_scales=raw_scales,
    )


def test_canonical_constants():
    assert CANONICAL_HEIGHT == 40
    assert CANONICAL_WIDTH == 40
    assert CANONICAL_CHANNELS == 3

    assert (
        CANONICAL_HEIGHT
        * CANONICAL_WIDTH
        * CANONICAL_CHANNELS
        == CANONICAL_DIM
    )


def test_canonical_images_cast_batchwise_to_float64_vectors():
    images = tf.reshape(
        tf.linspace(
            tf.constant(
                0.0,
                dtype=tf.float32,
            ),
            tf.constant(
                1.0,
                dtype=tf.float32,
            ),
            2 * CANONICAL_DIM,
        ),
        (
            2,
            CANONICAL_HEIGHT,
            CANONICAL_WIDTH,
            CANONICAL_CHANNELS,
        ),
    )

    original = images.numpy().copy()

    vectors = canonical_images_to_vectors(
        images
    )

    assert tuple(
        vectors.shape
    ) == (
        2,
        CANONICAL_DIM,
    )

    assert vectors.dtype == tf.float64

    assert np.isfinite(
        vectors.numpy()
    ).all()

    np.testing.assert_array_equal(
        images.numpy(),
        original,
    )


def test_canonical_input_dtype_is_fail_closed():
    images = tf.zeros(
        (
            1,
            CANONICAL_HEIGHT,
            CANONICAL_WIDTH,
            CANONICAL_CHANNELS,
        ),
        dtype=tf.float64,
    )

    with pytest.raises(
        TypeError
    ):
        canonical_images_to_vectors(
            images
        )


def test_canonical_input_shape_is_fail_closed():
    wrong_height = tf.zeros(
        (
            1,
            39,
            CANONICAL_WIDTH,
            CANONICAL_CHANNELS,
        ),
        dtype=tf.float32,
    )

    with pytest.raises(
        tf.errors.InvalidArgumentError
    ):
        canonical_images_to_vectors(
            wrong_height
        )

    wrong_channels = tf.zeros(
        (
            1,
            CANONICAL_HEIGHT,
            CANONICAL_WIDTH,
            1,
        ),
        dtype=tf.float32,
    )

    with pytest.raises(
        tf.errors.InvalidArgumentError
    ):
        canonical_images_to_vectors(
            wrong_channels
        )


def test_canonical_input_domain_is_fail_closed():
    below = np.zeros(
        (
            1,
            CANONICAL_HEIGHT,
            CANONICAL_WIDTH,
            CANONICAL_CHANNELS,
        ),
        dtype=np.float32,
    )

    below[
        0,
        0,
        0,
        0,
    ] = -0.01

    with pytest.raises(
        tf.errors.InvalidArgumentError
    ):
        canonical_images_to_vectors(
            tf.constant(
                below
            )
        )

    above = np.zeros_like(
        below
    )

    above[
        0,
        0,
        0,
        0,
    ] = 1.01

    with pytest.raises(
        tf.errors.InvalidArgumentError
    ):
        canonical_images_to_vectors(
            tf.constant(
                above
            )
        )


def test_model_state_contract():
    model = _model()

    assert tuple(
        model.logits.shape
    ) == (
        NUM_COMPONENTS,
    )

    assert tuple(
        model.means.shape
    ) == (
        NUM_COMPONENTS,
        CANONICAL_DIM,
    )

    assert tuple(
        model.raw_scales.shape
    ) == (
        NUM_COMPONENTS,
        CANONICAL_DIM,
    )

    assert model.logits.dtype == tf.float64
    assert model.means.dtype == tf.float64
    assert model.raw_scales.dtype == tf.float64

    assert model.logits.trainable
    assert model.means.trainable
    assert model.raw_scales.trainable

    assert model.trainable_parameters == (
        model.logits,
        model.means,
        model.raw_scales,
    )


def test_uniform_logits_produce_uniform_weights():
    model = _model()

    weights = model.mixture_weights().numpy()

    np.testing.assert_allclose(
        weights,
        np.full(
            (NUM_COMPONENTS,),
            0.2,
            dtype=np.float64,
        ),
        rtol=0.0,
        atol=1e-15,
    )


def test_model_scales_are_float64_and_above_floor():
    model = _model()

    scales = model.scales()

    assert scales.dtype == tf.float64

    assert tuple(
        scales.shape
    ) == (
        NUM_COMPONENTS,
        CANONICAL_DIM,
    )

    assert bool(
        tf.reduce_all(
            scales
            > tf.constant(
                MIN_SCALE,
                dtype=tf.float64,
            )
        ).numpy()
    )

    np.testing.assert_allclose(
        scales.numpy(),
        np.full(
            (
                NUM_COMPONENTS,
                CANONICAL_DIM,
            ),
            0.10,
            dtype=np.float64,
        ),
        rtol=1e-13,
        atol=1e-15,
    )


def test_model_log_prob_and_nll_are_finite():
    model = _model()

    images = tf.fill(
        (
            2,
            CANONICAL_HEIGHT,
            CANONICAL_WIDTH,
            CANONICAL_CHANNELS,
        ),
        tf.constant(
            0.5,
            dtype=tf.float32,
        ),
    )

    log_prob = model.log_prob(
        images
    )

    loss = model.nll(
        images
    )

    assert tuple(
        log_prob.shape
    ) == (2,)

    assert tuple(
        loss.shape
    ) == ()

    assert log_prob.dtype == tf.float64
    assert loss.dtype == tf.float64

    assert bool(
        tf.reduce_all(
            tf.math.is_finite(
                log_prob
            )
        ).numpy()
    )

    assert bool(
        tf.math.is_finite(
            loss
        ).numpy()
    )


def test_endpoint_heavy_canonical_images_are_finite():
    model = _model()

    images = np.zeros(
        (
            2,
            CANONICAL_HEIGHT,
            CANONICAL_WIDTH,
            CANONICAL_CHANNELS,
        ),
        dtype=np.float32,
    )

    images[
        1,
        ...,
    ] = 1.0

    log_prob = model.log_prob(
        tf.constant(
            images
        )
    )

    assert bool(
        tf.reduce_all(
            tf.math.is_finite(
                log_prob
            )
        ).numpy()
    )


def test_invalid_initial_state_shape_fails_closed():
    (
        logits,
        means,
        raw_scales,
    ) = _initial_state()

    with pytest.raises(
        tf.errors.InvalidArgumentError
    ):
        CensoredDiagonalGaussianMixture(
            logits=logits[:-1],
            means=means,
            raw_scales=raw_scales,
        )

    with pytest.raises(
        tf.errors.InvalidArgumentError
    ):
        CensoredDiagonalGaussianMixture(
            logits=logits,
            means=means[:, :-1],
            raw_scales=raw_scales,
        )


def test_nonfinite_initial_state_fails_closed():
    (
        logits,
        means,
        raw_scales,
    ) = _initial_state()

    bad_means = means.copy()

    bad_means[
        0,
        0,
    ] = np.nan

    with pytest.raises(
        tf.errors.InvalidArgumentError
    ):
        CensoredDiagonalGaussianMixture(
            logits=logits,
            means=bad_means,
            raw_scales=raw_scales,
        )


def test_state_snapshot_is_exact_detached_float64_copy():
    model = _model()

    state = model.state_arrays()

    assert set(
        state
    ) == {
        "logits",
        "means",
        "raw_scales",
    }

    for value in state.values():
        assert value.dtype == np.float64
        assert np.isfinite(
            value
        ).all()

    np.testing.assert_array_equal(
        state[
            "logits"
        ],
        model.logits.numpy(),
    )

    np.testing.assert_array_equal(
        state[
            "means"
        ],
        model.means.numpy(),
    )

    np.testing.assert_array_equal(
        state[
            "raw_scales"
        ],
        model.raw_scales.numpy(),
    )

    original_model_value = float(
        model.means[
            0,
            0,
        ].numpy()
    )

    state[
        "means"
    ][
        0,
        0,
    ] = 123.0

    assert float(
        model.means[
            0,
            0,
        ].numpy()
    ) == original_model_value


def test_all_trainable_parameter_gradients_are_finite():
    model = _model()

    values = np.linspace(
        0.0,
        1.0,
        CANONICAL_DIM,
        dtype=np.float32,
    ).reshape(
        1,
        CANONICAL_HEIGHT,
        CANONICAL_WIDTH,
        CANONICAL_CHANNELS,
    )

    images = tf.constant(
        values
    )

    with tf.GradientTape() as tape:
        loss = model.nll(
            images
        )

    gradients = tape.gradient(
        loss,
        model.trainable_parameters,
    )

    assert bool(
        tf.math.is_finite(
            loss
        ).numpy()
    )

    assert len(
        gradients
    ) == 3

    for gradient in gradients:
        assert gradient is not None

        assert gradient.dtype == tf.float64

        assert bool(
            tf.reduce_all(
                tf.math.is_finite(
                    gradient
                )
            ).numpy()
        )
