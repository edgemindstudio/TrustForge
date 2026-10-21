"""Tests for Paper 1 revision-2026 governed GMM SGD updates."""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from trustforge.models.paper01_revision_2026.gaussian_mixture_model import (
    CensoredDiagonalGaussianMixture,
)
from trustforge.models.paper01_revision_2026.gaussian_mixture_observation import (
    CANONICAL_DIM,
    NUM_COMPONENTS,
)
from trustforge.models.paper01_revision_2026.gaussian_mixture_optimizer import (
    GRADIENT_CLIPPING,
    LEARNING_RATE,
    MAX_TRAIN_BATCH_SIZE,
    SGD_MOMENTUM,
    sgd_train_step,
)


def _model() -> CensoredDiagonalGaussianMixture:
    return CensoredDiagonalGaussianMixture(
        logits=np.zeros(
            NUM_COMPONENTS,
            dtype=np.float64,
        ),
        means=np.full(
            (
                NUM_COMPONENTS,
                CANONICAL_DIM,
            ),
            0.45,
            dtype=np.float64,
        ),
        raw_scales=np.zeros(
            (
                NUM_COMPONENTS,
                CANONICAL_DIM,
            ),
            dtype=np.float64,
        ),
    )


def _images(
    batch_size: int,
    *,
    value: float = 0.5,
) -> tf.Tensor:
    return tf.fill(
        (
            batch_size,
            40,
            40,
            3,
        ),
        tf.constant(
            value,
            dtype=tf.float32,
        ),
    )


def test_governed_optimizer_constants():
    assert LEARNING_RATE == 0.001
    assert SGD_MOMENTUM == 0.0
    assert GRADIENT_CLIPPING is False
    assert MAX_TRAIN_BATCH_SIZE == 128


@pytest.mark.parametrize(
    "batch_size",
    [
        1,
        79,
        103,
        126,
        127,
        128,
    ],
)
def test_governed_balanced_batch_sizes_are_accepted(
    batch_size,
):
    model = _model()

    result = sgd_train_step(
        model,
        _images(
            batch_size
        ),
    )

    assert np.isfinite(
        result.loss
    )

    assert np.isfinite(
        result.gradient_global_norm
    )

    assert result.gradient_global_norm >= 0.0


def test_one_step_matches_exact_manual_momentum_free_sgd():
    model = _model()

    images = _images(
        7,
        value=0.37,
    )

    parameters = model.trainable_parameters

    before = [
        parameter.numpy().copy()
        for parameter in parameters
    ]

    with tf.GradientTape() as tape:
        loss = model.nll(
            images
        )

    gradients = tape.gradient(
        loss,
        parameters,
    )

    expected = [
        parameter_before
        - LEARNING_RATE
        * gradient.numpy()
        for parameter_before, gradient in zip(
            before,
            gradients,
        )
    ]

    result = sgd_train_step(
        model,
        images,
    )

    assert result.loss == pytest.approx(
        float(
            loss.numpy()
        ),
        rel=0.0,
        abs=0.0,
    )

    for actual, wanted in zip(
        model.state_arrays().values(),
        expected,
    ):
        np.testing.assert_array_equal(
            actual,
            wanted,
        )


def test_optimizer_changes_no_latent_mean_by_projection():
    model = _model()

    model.means.assign(
        tf.fill(
            model.means.shape,
            tf.constant(
                1.25,
                dtype=tf.float64,
            ),
        )
    )

    sgd_train_step(
        model,
        _images(
            3,
            value=1.0,
        ),
    )

    assert np.any(
        model.means.numpy()
        > 1.0
    )


def test_mixture_weights_remain_finite_positive_and_normalized():
    model = _model()

    sgd_train_step(
        model,
        _images(
            5
        ),
    )

    weights = model.mixture_weights().numpy()

    assert np.isfinite(
        weights
    ).all()

    assert (
        weights
        > 0.0
    ).all()

    assert np.sum(
        weights
    ) == pytest.approx(
        1.0,
        rel=0.0,
        abs=1e-12,
    )


def test_scales_remain_finite_after_update():
    model = _model()

    sgd_train_step(
        model,
        _images(
            5
        ),
    )

    scales = model.scales().numpy()

    assert np.isfinite(
        scales
    ).all()

    assert (
        scales
        > 0.0
    ).all()


def test_empty_batch_fails_without_mutating_state():
    model = _model()

    before = model.state_arrays()

    with pytest.raises(
        ValueError
    ):
        sgd_train_step(
            model,
            _images(
                0
            ),
        )

    after = model.state_arrays()

    for key in before:
        np.testing.assert_array_equal(
            after[
                key
            ],
            before[
                key
            ],
        )


def test_oversized_batch_fails_without_mutating_state():
    model = _model()

    before = model.state_arrays()

    with pytest.raises(
        ValueError
    ):
        sgd_train_step(
            model,
            _images(
                129
            ),
        )

    after = model.state_arrays()

    for key in before:
        np.testing.assert_array_equal(
            after[
                key
            ],
            before[
                key
            ],
        )


def test_noncanonical_dtype_fails_without_mutating_state():
    model = _model()

    before = model.state_arrays()

    images = tf.zeros(
        (
            2,
            40,
            40,
            3,
        ),
        dtype=tf.float64,
    )

    with pytest.raises(
        TypeError
    ):
        sgd_train_step(
            model,
            images,
        )

    after = model.state_arrays()

    for key in before:
        np.testing.assert_array_equal(
            after[
                key
            ],
            before[
                key
            ],
        )


def test_nonfinite_input_fails_without_mutating_state():
    model = _model()

    before = model.state_arrays()

    images = np.full(
        (
            2,
            40,
            40,
            3,
        ),
        0.5,
        dtype=np.float32,
    )

    images[
        0,
        0,
        0,
        0,
    ] = np.nan

    with pytest.raises(
        tf.errors.InvalidArgumentError
    ):
        sgd_train_step(
            model,
            tf.convert_to_tensor(
                images
            ),
        )

    after = model.state_arrays()

    for key in before:
        np.testing.assert_array_equal(
            after[
                key
            ],
            before[
                key
            ],
        )


def test_model_parameters_remain_float64():
    model = _model()

    sgd_train_step(
        model,
        _images(
            4
        ),
    )

    for parameter in model.trainable_parameters:
        assert parameter.dtype == tf.float64

    for value in model.state_arrays().values():
        assert value.dtype == np.float64
