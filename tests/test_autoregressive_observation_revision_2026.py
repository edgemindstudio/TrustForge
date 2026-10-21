"""Regression tests for the governed Paper 1 revision-2026 AR observation model."""

from __future__ import annotations

import inspect

import numpy as np
import tensorflow as tf

from trustforge.models.paper01_revision_2026.autoregressive_observation import (
    censored_logistic_log_prob,
    coupled_rgb_censored_logistic_nll,
    coupled_rgb_locations,
    sample_coupled_rgb_from_parameters,
    split_censored_logistic_parameters,
)


def test_parameter_split_contract():
    params = tf.constant(
        [
            [
                [
                    0.20,
                    0.40,
                    0.60,
                    -1.00,
                    0.00,
                    1.00,
                    0.50,
                    -0.75,
                    1.25,
                ]
            ]
        ],
        dtype=tf.float32,
    )

    base_mu, scale, alpha = split_censored_logistic_parameters(
        params
    )

    assert tuple(base_mu.shape) == (1, 1, 3)
    assert tuple(scale.shape) == (1, 1, 3)
    assert tuple(alpha.shape) == (1, 1, 3)

    assert bool(
        tf.reduce_all(
            tf.math.is_finite(
                base_mu
            )
        )
    )

    assert bool(
        tf.reduce_all(
            tf.math.is_finite(
                scale
            )
        )
    )

    assert bool(
        tf.reduce_all(
            scale > 0.0
        )
    )

    assert bool(
        tf.reduce_all(
            tf.math.abs(
                alpha
            ) <= 1.0
        )
    )


def test_boundary_and_interior_log_probabilities_are_finite():
    x = tf.constant(
        [
            0.0,
            0.25,
            0.50,
            0.75,
            1.0,
        ],
        dtype=tf.float32,
    )

    mu = tf.constant(
        [
            0.2,
            0.2,
            0.5,
            0.8,
            0.8,
        ],
        dtype=tf.float32,
    )

    scale = tf.ones_like(
        x
    ) * 0.25

    log_prob = censored_logistic_log_prob(
        x,
        mu,
        scale,
    )

    assert bool(
        tf.reduce_all(
            tf.math.is_finite(
                log_prob
            )
        )
    )

    assert bool(
        tf.math.is_finite(
            log_prob[0]
        )
    )

    assert bool(
        tf.math.is_finite(
            log_prob[-1]
        )
    )


def test_likelihood_api_has_no_endpoint_epsilon():
    signature = inspect.signature(
        censored_logistic_log_prob
    )

    assert "epsilon" not in signature.parameters


def test_rgb_channel_coupling_order():
    base = tf.constant(
        [
            [
                [
                    0.5,
                    0.5,
                    0.5,
                ]
            ]
        ],
        dtype=tf.float32,
    )

    coupling = tf.constant(
        [
            [
                [
                    0.8,
                    0.6,
                    -0.4,
                ]
            ]
        ],
        dtype=tf.float32,
    )

    context_a = tf.constant(
        [
            [
                [
                    0.20,
                    0.30,
                    0.40,
                ]
            ]
        ],
        dtype=tf.float32,
    )

    context_b = tf.constant(
        [
            [
                [
                    0.80,
                    0.30,
                    0.40,
                ]
            ]
        ],
        dtype=tf.float32,
    )

    context_c = tf.constant(
        [
            [
                [
                    0.20,
                    0.90,
                    0.40,
                ]
            ]
        ],
        dtype=tf.float32,
    )

    loc_a = coupled_rgb_locations(
        base,
        coupling,
        context_a,
    )

    loc_b = coupled_rgb_locations(
        base,
        coupling,
        context_b,
    )

    loc_c = coupled_rgb_locations(
        base,
        coupling,
        context_c,
    )

    tf.debugging.assert_near(
        loc_a[..., 0],
        loc_b[..., 0],
        atol=0.0,
    )

    tf.debugging.assert_near(
        loc_a[..., 0],
        loc_c[..., 0],
        atol=0.0,
    )

    assert not np.array_equal(
        loc_a[..., 1].numpy(),
        loc_b[..., 1].numpy(),
    )

    assert not np.array_equal(
        loc_a[..., 2].numpy(),
        loc_b[..., 2].numpy(),
    )

    assert not np.array_equal(
        loc_a[..., 2].numpy(),
        loc_c[..., 2].numpy(),
    )


def test_mixed_rgb_nll_is_finite():
    targets = tf.constant(
        [
            [
                [
                    [0.0, 0.25, 1.0],
                    [0.10, 0.50, 0.90],
                ],
                [
                    [1.0, 0.75, 0.0],
                    [0.33, 0.66, 0.50],
                ],
            ]
        ],
        dtype=tf.float32,
    )

    params = tf.zeros(
        (
            1,
            2,
            2,
            9,
        ),
        dtype=tf.float32,
    )

    nll = coupled_rgb_censored_logistic_nll(
        targets,
        params,
    )

    assert bool(
        tf.math.is_finite(
            nll
        )
    )


def test_sampling_reproducibility_and_bounds():
    params = tf.zeros(
        (
            4,
            5,
            9,
        ),
        dtype=tf.float32,
    )

    a = sample_coupled_rgb_from_parameters(
        params,
        temperature=1.0,
        seed=42,
    ).numpy()

    b = sample_coupled_rgb_from_parameters(
        params,
        temperature=1.0,
        seed=42,
    ).numpy()

    c = sample_coupled_rgb_from_parameters(
        params,
        temperature=1.0,
        seed=43,
    ).numpy()

    assert a.shape == (4, 5, 3)

    assert np.array_equal(
        a,
        b,
    )

    assert not np.array_equal(
        a,
        c,
    )

    assert np.isfinite(
        a
    ).all()

    assert a.min() >= 0.0
    assert a.max() <= 1.0


def test_temperature_zero_is_seed_independent():
    params = tf.constant(
        [
            [
                [
                    0.20,
                    0.40,
                    0.60,
                    -1.00,
                    0.00,
                    1.00,
                    0.50,
                    -0.75,
                    1.25,
                ]
            ]
        ],
        dtype=tf.float32,
    )

    a = sample_coupled_rgb_from_parameters(
        params,
        temperature=0.0,
        seed=42,
    ).numpy()

    b = sample_coupled_rgb_from_parameters(
        params,
        temperature=0.0,
        seed=999,
    ).numpy()

    assert np.array_equal(
        a,
        b,
    )

    assert np.isfinite(
        a
    ).all()

    assert a.min() >= 0.0
    assert a.max() <= 1.0


def test_censored_logistic_nll_has_finite_gradients():
    target = tf.constant(
        [
            [
                [
                    [0.0, 0.25, 1.0],
                    [0.10, 0.50, 0.90],
                ],
                [
                    [1.0, 0.75, 0.0],
                    [0.33, 0.66, 0.50],
                ],
            ]
        ],
        dtype=tf.float32,
    )

    params = tf.Variable(
        tf.zeros(
            (
                1,
                2,
                2,
                9,
            ),
            dtype=tf.float32,
        )
    )

    with tf.GradientTape() as tape:
        loss = coupled_rgb_censored_logistic_nll(
            target,
            params,
        )

    gradients = tape.gradient(
        loss,
        params,
    )

    assert gradients is not None

    assert bool(
        tf.math.is_finite(
            loss
        )
    )

    assert bool(
        tf.reduce_all(
            tf.math.is_finite(
                gradients
            )
        )
    )


def test_rgb_coupling_changes_governed_channel_likelihoods():
    base = tf.constant(
        [
            [
                [
                    0.5,
                    0.5,
                    0.5,
                ]
            ]
        ],
        dtype=tf.float32,
    )

    scale = tf.constant(
        [
            [
                [
                    0.25,
                    0.25,
                    0.25,
                ]
            ]
        ],
        dtype=tf.float32,
    )

    alpha = tf.constant(
        [
            [
                [
                    0.8,
                    0.6,
                    -0.4,
                ]
            ]
        ],
        dtype=tf.float32,
    )

    context_a = tf.constant(
        [
            [
                [
                    0.20,
                    0.30,
                    0.40,
                ]
            ]
        ],
        dtype=tf.float32,
    )

    context_red_changed = tf.constant(
        [
            [
                [
                    0.80,
                    0.30,
                    0.40,
                ]
            ]
        ],
        dtype=tf.float32,
    )

    context_green_changed = tf.constant(
        [
            [
                [
                    0.20,
                    0.90,
                    0.40,
                ]
            ]
        ],
        dtype=tf.float32,
    )

    loc_a = coupled_rgb_locations(
        base,
        alpha,
        context_a,
    )

    loc_red_changed = coupled_rgb_locations(
        base,
        alpha,
        context_red_changed,
    )

    loc_green_changed = coupled_rgb_locations(
        base,
        alpha,
        context_green_changed,
    )

    green_target = tf.constant(
        0.60,
        dtype=tf.float32,
    )

    green_lp_a = censored_logistic_log_prob(
        green_target,
        loc_a[..., 1],
        scale[..., 1],
    )

    green_lp_red_changed = censored_logistic_log_prob(
        green_target,
        loc_red_changed[..., 1],
        scale[..., 1],
    )

    assert not np.array_equal(
        green_lp_a.numpy(),
        green_lp_red_changed.numpy(),
    )

    blue_target = tf.constant(
        0.70,
        dtype=tf.float32,
    )

    blue_lp_a = censored_logistic_log_prob(
        blue_target,
        loc_a[..., 2],
        scale[..., 2],
    )

    blue_lp_red_changed = censored_logistic_log_prob(
        blue_target,
        loc_red_changed[..., 2],
        scale[..., 2],
    )

    blue_lp_green_changed = censored_logistic_log_prob(
        blue_target,
        loc_green_changed[..., 2],
        scale[..., 2],
    )

    assert not np.array_equal(
        blue_lp_a.numpy(),
        blue_lp_red_changed.numpy(),
    )

    assert not np.array_equal(
        blue_lp_a.numpy(),
        blue_lp_green_changed.numpy(),
    )
