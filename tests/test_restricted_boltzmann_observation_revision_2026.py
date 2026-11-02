"""Focused tests for the governed revision-2026 RBM observation model."""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from trustforge.models.paper01_revision_2026.restricted_boltzmann_observation import (
    DTYPE,
    VISIBLE_MAX,
    VISIBLE_MIN,
    inverse_cdf,
    log_normalizer,
    log_prob,
    mean,
    sample_from_uniform,
    validate_visible_values,
)


def test_governed_dtype_and_visible_domain_constants():
    assert DTYPE == tf.float64
    assert VISIBLE_MIN == 0.0
    assert VISIBLE_MAX == 1.0


def test_log_normalizer_zero_limit_is_exact():
    result = log_normalizer(
        tf.constant(0.0, dtype=tf.float64)
    )
    assert result.dtype == tf.float64
    np.testing.assert_allclose(
        result.numpy(),
        0.0,
        rtol=0.0,
        atol=1.0e-15,
    )


def test_mean_zero_limit_is_exact_half():
    result = mean(
        tf.constant(0.0, dtype=tf.float64)
    )
    np.testing.assert_allclose(
        result.numpy(),
        0.5,
        rtol=0.0,
        atol=1.0e-15,
    )


def test_extreme_theta_forward_values_are_finite():
    theta = tf.constant(
        [-1.0e6, -100.0, -1.0, 0.0, 1.0, 100.0, 1.0e6],
        dtype=tf.float64,
    )

    lnz = log_normalizer(theta)
    mu = mean(theta)

    assert np.all(np.isfinite(lnz.numpy()))
    assert np.all(np.isfinite(mu.numpy()))
    assert np.all(mu.numpy() >= 0.0)
    assert np.all(mu.numpy() <= 1.0)


def test_inverse_cdf_extreme_theta_is_finite_and_bounded():
    theta = tf.constant(
        [-1.0e6, -100.0, -1.0, -1.0e-8, 0.0, 1.0e-8, 1.0, 100.0, 1.0e6],
        dtype=tf.float64,
    )
    u = tf.constant(
        [0.0, 1.0e-12, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0 - 1.0e-12, 1.0],
        dtype=tf.float64,
    )

    samples = inverse_cdf(u, theta).numpy()

    assert np.all(np.isfinite(samples))
    assert np.all(samples >= 0.0)
    assert np.all(samples <= 1.0)


def test_inverse_cdf_uniform_limit_at_zero():
    u = tf.constant(
        [0.0, 0.1, 0.25, 0.5, 0.9, 1.0],
        dtype=tf.float64,
    )
    theta = tf.zeros_like(u)

    samples = inverse_cdf(u, theta)

    np.testing.assert_allclose(
        samples.numpy(),
        u.numpy(),
        rtol=0.0,
        atol=1.0e-15,
    )


def test_inverse_cdf_is_monotone_in_uniform_draw():
    u = tf.linspace(
        tf.constant(0.0, dtype=tf.float64),
        tf.constant(1.0, dtype=tf.float64),
        1001,
    )

    for theta_value in (-100.0, -1.0, 0.0, 1.0, 100.0):
        theta = tf.fill(
            tf.shape(u),
            tf.constant(theta_value, dtype=tf.float64),
        )
        samples = inverse_cdf(u, theta).numpy()
        assert np.all(np.diff(samples) >= -1.0e-14)


def test_analytic_mean_matches_large_empirical_sample():
    rng = np.random.default_rng(20261023)

    for theta_value in (-10.0, -1.0, 0.0, 1.0, 10.0):
        u = rng.random(
            200_000,
            dtype=np.float64,
        )
        theta = np.full(
            u.shape,
            theta_value,
            dtype=np.float64,
        )

        samples = sample_from_uniform(
            theta,
            u,
        ).numpy()

        expected = float(
            mean(
                tf.constant(theta_value, dtype=tf.float64)
            ).numpy()
        )

        assert abs(float(samples.mean()) - expected) < 5.0e-3


def test_log_prob_is_finite_at_visible_endpoints():
    values = tf.constant(
        [0.0, 1.0],
        dtype=tf.float64,
    )

    for theta_value in (-100.0, -1.0, 0.0, 1.0, 100.0):
        theta = tf.fill(
            tf.shape(values),
            tf.constant(theta_value, dtype=tf.float64),
        )
        result = log_prob(values, theta).numpy()
        assert np.all(np.isfinite(result))


def test_validate_visible_values_accepts_endpoints():
    values = tf.constant(
        [0.0, 0.25, 0.5, 0.75, 1.0],
        dtype=tf.float64,
    )

    result = validate_visible_values(values)

    assert result.dtype == tf.float64
    np.testing.assert_array_equal(
        result.numpy(),
        values.numpy(),
    )


@pytest.mark.parametrize(
    "bad_value",
    [-1.0e-12, 1.0 + 1.0e-12],
)
def test_validate_visible_values_rejects_out_of_range(bad_value):
    with pytest.raises(tf.errors.InvalidArgumentError):
        validate_visible_values(
            tf.constant(
                [bad_value],
                dtype=tf.float64,
            )
        )


@pytest.mark.parametrize(
    "bad_value",
    [np.nan, np.inf, -np.inf],
)
def test_nonfinite_inputs_fail_closed(bad_value):
    with pytest.raises(tf.errors.InvalidArgumentError):
        mean(
            tf.constant(
                bad_value,
                dtype=tf.float64,
            )
        )


def test_gradients_are_finite_through_zero():
    with tf.GradientTape() as tape:
        theta = tf.Variable(
            [-1.0, -1.0e-8, 0.0, 1.0e-8, 1.0],
            dtype=tf.float64,
        )
        objective = tf.reduce_sum(
            log_normalizer(theta)
            + mean(theta)
        )

    gradient = tape.gradient(
        objective,
        theta,
    )

    assert gradient is not None
    assert gradient.dtype == tf.float64
    assert np.all(np.isfinite(gradient.numpy()))


def test_sample_from_uniform_is_deterministic_for_identical_inputs():
    theta = tf.constant(
        [-2.0, -1.0, 0.0, 1.0, 2.0],
        dtype=tf.float64,
    )
    u = tf.constant(
        [0.1, 0.2, 0.3, 0.4, 0.5],
        dtype=tf.float64,
    )

    first = sample_from_uniform(
        theta,
        u,
    ).numpy()
    second = sample_from_uniform(
        theta,
        u,
    ).numpy()

    np.testing.assert_array_equal(
        first,
        second,
    )
