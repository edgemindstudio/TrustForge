"""Regression tests for the Paper 1 revision-2026 GMM observation model."""

from __future__ import annotations

import math

import numpy as np
import pytest
import tensorflow as tf
from scipy.special import log_ndtr

from trustforge.models.paper01_revision_2026.gaussian_mixture_observation import (
    CANONICAL_DIM,
    DTYPE,
    MIN_SCALE,
    MIN_VARIANCE,
    NUM_COMPONENTS,
    TAIL_CUTOFF,
    censored_normal_log_prob,
    inverse_softplus_positive,
    mean_censored_mixture_nll,
    mixture_log_prob,
    stable_log_ndtr,
    transformed_scale,
)


def test_governed_constants():
    assert NUM_COMPONENTS == 5
    assert CANONICAL_DIM == 4800
    assert MIN_VARIANCE == pytest.approx(
        1e-3,
        rel=0.0,
        abs=0.0,
    )
    assert MIN_SCALE == pytest.approx(
        math.sqrt(1e-3),
        rel=0.0,
        abs=0.0,
    )
    assert TAIL_CUTOFF == -10.0
    assert DTYPE == tf.float64


def test_stable_log_ndtr_matches_scipy_oracle():
    z = np.array(
        [
            -100.0,
            -60.0,
            -40.0,
            -20.0,
            -12.0,
            -10.000001,
            -10.0,
            -9.999999,
            -8.0,
            -4.0,
            -2.0,
            0.0,
            2.0,
            4.0,
            8.0,
            20.0,
            100.0,
        ],
        dtype=np.float64,
    )

    observed = stable_log_ndtr(
        tf.constant(
            z,
            dtype=tf.float64,
        )
    ).numpy()

    expected = log_ndtr(
        z
    )

    assert np.isfinite(
        observed
    ).all()

    max_abs_error = float(
        np.max(
            np.abs(
                observed
                - expected
            )
        )
    )

    assert max_abs_error < 2e-8


def test_stable_log_ndtr_gradient_is_finite_across_cutoff():
    z = tf.Variable(
        [
            -100.0,
            -20.0,
            -10.0001,
            -10.0,
            -9.9999,
            -2.0,
            0.0,
            2.0,
            10.0,
        ],
        dtype=tf.float64,
    )

    with tf.GradientTape() as tape:
        value = tf.reduce_sum(
            stable_log_ndtr(
                z
            )
        )

    gradient = tape.gradient(
        value,
        z,
    )

    assert gradient is not None

    assert bool(
        tf.reduce_all(
            tf.math.is_finite(
                gradient
            )
        ).numpy()
    )

    assert bool(
        tf.reduce_all(
            gradient >= 0.0
        ).numpy()
    )


def test_cutoff_neighborhood_matches_scipy():
    z = np.array(
        [
            -10.001,
            -10.0001,
            -10.000001,
            -10.0,
            -9.999999,
            -9.9999,
            -9.999,
        ],
        dtype=np.float64,
    )

    observed = stable_log_ndtr(
        tf.constant(
            z,
            dtype=tf.float64,
        )
    ).numpy()

    expected = log_ndtr(
        z
    )

    np.testing.assert_allclose(
        observed,
        expected,
        rtol=0.0,
        atol=2e-8,
    )


def test_scale_parameterization_and_inverse_round_trip():
    desired_scale = tf.constant(
        [
            MIN_SCALE + 1e-12,
            0.05,
            0.10,
            0.25,
            0.75,
        ],
        dtype=tf.float64,
    )

    positive_offset = (
        desired_scale
        - tf.constant(
            MIN_SCALE,
            dtype=tf.float64,
        )
    )

    raw = inverse_softplus_positive(
        positive_offset
    )

    recovered = transformed_scale(
        raw
    )

    assert bool(
        tf.reduce_all(
            recovered
            > tf.constant(
                MIN_SCALE,
                dtype=tf.float64,
            )
        ).numpy()
    )

    np.testing.assert_allclose(
        recovered.numpy(),
        desired_scale.numpy(),
        rtol=1e-12,
        atol=2e-15,
    )


def test_censored_scalar_endpoint_and_interior_oracle():
    x = tf.constant(
        [
            0.0,
            0.5,
            1.0,
        ],
        dtype=tf.float64,
    )

    mean = tf.constant(
        [
            0.2,
            0.5,
            0.8,
        ],
        dtype=tf.float64,
    )

    scale = tf.constant(
        [
            0.25,
            0.25,
            0.25,
        ],
        dtype=tf.float64,
    )

    observed = censored_normal_log_prob(
        x,
        mean,
        scale,
    ).numpy()

    expected = np.array(
        [
            log_ndtr(
                -0.2 / 0.25
            ),
            (
                -math.log(0.25)
                - 0.5
                * math.log(
                    2.0
                    * math.pi
                )
            ),
            log_ndtr(
                (
                    0.8
                    - 1.0
                )
                / 0.25
            ),
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(
        observed,
        expected,
        rtol=0.0,
        atol=1e-11,
    )


def test_censored_scalar_mixed_boundaries_are_finite():
    x = tf.constant(
        [
            0.0,
            0.1,
            0.25,
            0.5,
            0.75,
            0.9,
            1.0,
        ],
        dtype=tf.float64,
    )

    mean = tf.constant(
        [
            -3.0,
            -0.5,
            0.0,
            0.5,
            1.0,
            1.5,
            4.0,
        ],
        dtype=tf.float64,
    )

    scale = tf.constant(
        [
            0.04,
            0.05,
            0.10,
            0.20,
            0.10,
            0.05,
            0.04,
        ],
        dtype=tf.float64,
    )

    log_prob = censored_normal_log_prob(
        x,
        mean,
        scale,
    )

    assert tuple(
        log_prob.shape
    ) == (7,)

    assert bool(
        tf.reduce_all(
            tf.math.is_finite(
                log_prob
            )
        ).numpy()
    )


def test_mixture_log_prob_contract():
    batch = 3
    dimension = 8

    x = tf.constant(
        [
            [
                0.0,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                1.0,
            ],
            [
                1.0,
                0.9,
                0.8,
                0.7,
                0.6,
                0.5,
                0.4,
                0.0,
            ],
            [
                0.0,
                1.0,
                0.25,
                0.75,
                0.5,
                0.5,
                1.0,
                0.0,
            ],
        ],
        dtype=tf.float64,
    )

    logits = tf.zeros(
        (NUM_COMPONENTS,),
        dtype=tf.float64,
    )

    means = tf.reshape(
        tf.linspace(
            tf.constant(
                -0.1,
                dtype=tf.float64,
            ),
            tf.constant(
                1.1,
                dtype=tf.float64,
            ),
            NUM_COMPONENTS
            * dimension,
        ),
        (
            NUM_COMPONENTS,
            dimension,
        ),
    )

    raw_scales = tf.zeros(
        (
            NUM_COMPONENTS,
            dimension,
        ),
        dtype=tf.float64,
    )

    log_prob = mixture_log_prob(
        x,
        logits,
        means,
        raw_scales,
    )

    assert tuple(
        log_prob.shape
    ) == (batch,)

    assert bool(
        tf.reduce_all(
            tf.math.is_finite(
                log_prob
            )
        ).numpy()
    )

    nll = mean_censored_mixture_nll(
        x,
        logits,
        means,
        raw_scales,
    )

    assert tuple(
        nll.shape
    ) == ()

    assert bool(
        tf.math.is_finite(
            nll
        ).numpy()
    )


def test_mixture_probability_is_component_order_invariant():
    x = tf.constant(
        [
            [
                0.0,
                0.25,
                0.5,
                0.75,
                1.0,
            ]
        ],
        dtype=tf.float64,
    )

    logits = tf.constant(
        [
            -1.0,
            -0.5,
            0.0,
            0.5,
            1.0,
        ],
        dtype=tf.float64,
    )

    means = tf.constant(
        np.linspace(
            -0.2,
            1.2,
            NUM_COMPONENTS * 5,
            dtype=np.float64,
        ).reshape(
            NUM_COMPONENTS,
            5,
        )
    )

    raw_scales = tf.constant(
        np.linspace(
            -2.0,
            0.5,
            NUM_COMPONENTS * 5,
            dtype=np.float64,
        ).reshape(
            NUM_COMPONENTS,
            5,
        )
    )

    permutation = tf.constant(
        [
            3,
            0,
            4,
            1,
            2,
        ],
        dtype=tf.int32,
    )

    baseline = mixture_log_prob(
        x,
        logits,
        means,
        raw_scales,
    )

    reordered = mixture_log_prob(
        x,
        tf.gather(
            logits,
            permutation,
        ),
        tf.gather(
            means,
            permutation,
        ),
        tf.gather(
            raw_scales,
            permutation,
        ),
    )

    np.testing.assert_allclose(
        baseline.numpy(),
        reordered.numpy(),
        rtol=0.0,
        atol=1e-12,
    )


def test_mixture_gradients_are_finite():
    x = tf.constant(
        [
            [
                0.0,
                0.1,
                0.5,
                0.9,
                1.0,
                0.25,
            ],
            [
                1.0,
                0.9,
                0.5,
                0.1,
                0.0,
                0.75,
            ],
        ],
        dtype=tf.float64,
    )

    logits = tf.Variable(
        tf.zeros(
            (NUM_COMPONENTS,),
            dtype=tf.float64,
        )
    )

    means = tf.Variable(
        tf.reshape(
            tf.linspace(
                tf.constant(
                    -0.1,
                    dtype=tf.float64,
                ),
                tf.constant(
                    1.1,
                    dtype=tf.float64,
                ),
                NUM_COMPONENTS * 6,
            ),
            (
                NUM_COMPONENTS,
                6,
            ),
        )
    )

    raw_scales = tf.Variable(
        tf.zeros(
            (
                NUM_COMPONENTS,
                6,
            ),
            dtype=tf.float64,
        )
    )

    with tf.GradientTape() as tape:
        loss = mean_censored_mixture_nll(
            x,
            logits,
            means,
            raw_scales,
        )

    gradients = tape.gradient(
        loss,
        [
            logits,
            means,
            raw_scales,
        ],
    )

    assert bool(
        tf.math.is_finite(
            loss
        ).numpy()
    )

    for gradient in gradients:
        assert gradient is not None

        assert bool(
            tf.reduce_all(
                tf.math.is_finite(
                    gradient
                )
            ).numpy()
        )


def test_out_of_domain_observation_fails_closed():
    logits = tf.zeros(
        (NUM_COMPONENTS,),
        dtype=tf.float64,
    )

    means = tf.zeros(
        (
            NUM_COMPONENTS,
            2,
        ),
        dtype=tf.float64,
    )

    raw_scales = tf.zeros(
        (
            NUM_COMPONENTS,
            2,
        ),
        dtype=tf.float64,
    )

    with pytest.raises(
        tf.errors.InvalidArgumentError
    ):
        mixture_log_prob(
            tf.constant(
                [
                    [
                        -0.01,
                        0.50,
                    ]
                ],
                dtype=tf.float64,
            ),
            logits,
            means,
            raw_scales,
        )

    with pytest.raises(
        tf.errors.InvalidArgumentError
    ):
        mixture_log_prob(
            tf.constant(
                [
                    [
                        0.50,
                        1.01,
                    ]
                ],
                dtype=tf.float64,
            ),
            logits,
            means,
            raw_scales,
        )
