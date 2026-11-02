"""Regression tests for Paper 1 revision-2026 VAE observation math."""

from __future__ import annotations

import math

import numpy as np
import pytest
import tensorflow as tf

from trustforge.models.paper01_revision_2026 import (
    variational_autoencoder_observation as vae,
)


SHAPE = (40, 40, 3)
D = 4800


def image_batch(value: float = 0.0, batch: int = 2) -> tf.Tensor:
    return tf.fill((batch, *SHAPE), tf.cast(value, tf.float32))


def test_canonical_internal_round_trip():
    x = image_batch(0.25)
    internal = vae.canonical_to_internal(x)
    np.testing.assert_allclose(internal.numpy(), -0.5)
    np.testing.assert_allclose(
        vae.internal_to_canonical(internal).numpy(),
        x.numpy(),
        atol=0.0,
    )


@pytest.mark.parametrize(
    ("canonical", "internal"),
    [(0.0, -1.0), (0.5, 0.0), (1.0, 1.0)],
)
def test_canonical_endpoints_and_midpoint(canonical, internal):
    x = image_batch(canonical)
    actual = vae.canonical_to_internal(x)
    np.testing.assert_allclose(actual.numpy(), internal, atol=0.0)


def test_gaussian_nll_zero_residual_reference():
    x = image_batch()
    nll = vae.gaussian_nll_per_example(x, x)
    expected = 0.5 * D * math.log(2.0 * math.pi)
    np.testing.assert_allclose(nll.numpy(), expected, rtol=1e-6)


def test_gaussian_nll_nonzero_residual_reference():
    x = image_batch()
    means = image_batch(1.0)
    actual = vae.gaussian_nll_per_example(x, means)
    expected = 0.5 * D * (1.0 + math.log(2.0 * math.pi))
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-6)


def test_gaussian_nll_returns_one_value_per_example():
    x = image_batch(batch=3)
    means = image_batch(batch=3)
    assert tuple(vae.gaussian_nll_per_example(x, means).shape) == (3,)


def test_standard_normal_posterior_has_zero_kl():
    mu = tf.zeros((2, 4), dtype=tf.float32)
    logvar = tf.zeros_like(mu)
    kl = vae.diagonal_gaussian_kl_per_example(mu, logvar)
    np.testing.assert_allclose(kl.numpy(), 0.0, atol=1e-7)


def test_shifted_posterior_kl_reference():
    mu = tf.ones((2, 4), dtype=tf.float32)
    logvar = tf.zeros_like(mu)
    kl = vae.diagonal_gaussian_kl_per_example(mu, logvar)
    np.testing.assert_allclose(kl.numpy(), 2.0, atol=1e-6)


def test_nonunit_variance_kl_reference():
    mu = tf.zeros((2, 4), dtype=tf.float32)
    logvar = tf.fill((2, 4), math.log(2.0))
    kl = vae.diagonal_gaussian_kl_per_example(mu, logvar)
    expected = 0.5 * 4 * (2.0 - 1.0 - math.log(2.0))
    np.testing.assert_allclose(kl.numpy(), expected, rtol=1e-5)


def test_combined_objective_matches_components():
    x = image_batch()
    means = image_batch()
    mu = tf.ones((2, 4), dtype=tf.float32)
    logvar = tf.zeros_like(mu)

    total, recon, kl = vae.vae_objective(x, means, mu, logvar)

    np.testing.assert_allclose(
        total.numpy(), (recon + kl).numpy(), rtol=1e-6
    )
    np.testing.assert_allclose(kl.numpy(), 2.0, atol=1e-6)


def test_reconstruction_gradient_finite_and_nonzero():
    x = image_batch()
    means = tf.Variable(image_batch(0.25))

    with tf.GradientTape() as tape:
        loss = tf.reduce_mean(
            vae.gaussian_nll_per_example(x, means)
        )

    grad = tape.gradient(loss, means)
    assert grad is not None
    assert bool(tf.reduce_all(tf.math.is_finite(grad)))
    assert bool(tf.reduce_any(tf.not_equal(grad, 0.0)))


def test_kl_gradient_finite_and_nonzero():
    mu = tf.Variable(tf.ones((2, 4), dtype=tf.float32))
    logvar = tf.Variable(tf.zeros((2, 4), dtype=tf.float32))

    with tf.GradientTape() as tape:
        loss = tf.reduce_mean(
            vae.diagonal_gaussian_kl_per_example(mu, logvar)
        )

    grad_mu, grad_logvar = tape.gradient(loss, [mu, logvar])
    assert grad_mu is not None
    assert grad_logvar is not None
    assert bool(tf.reduce_all(tf.math.is_finite(grad_mu)))
    assert bool(tf.reduce_all(tf.math.is_finite(grad_logvar)))
    assert bool(tf.reduce_any(tf.not_equal(grad_mu, 0.0)))


@pytest.mark.parametrize(
    "bad_shape",
    [(2, 40, 40, 1), (2, 32, 40, 3), (2, 4800)],
)
def test_rejects_wrong_image_shapes(bad_shape):
    x = tf.zeros(bad_shape, dtype=tf.float32)
    with pytest.raises((tf.errors.InvalidArgumentError, ValueError)):
        vae.canonical_to_internal(x)


def test_rejects_wrong_image_dtype():
    x = tf.zeros((2, *SHAPE), dtype=tf.float64)
    with pytest.raises(TypeError):
        vae.canonical_to_internal(x)


@pytest.mark.parametrize("invalid", [-0.01, 1.01])
def test_rejects_out_of_range_canonical_inputs(invalid):
    with pytest.raises(tf.errors.InvalidArgumentError):
        vae.canonical_to_internal(image_batch(invalid))


def test_rejects_nonfinite_image():
    x = image_batch(float("nan"))
    with pytest.raises(tf.errors.InvalidArgumentError):
        vae.canonical_to_internal(x)


def test_rejects_mismatched_reconstruction_batch_size():
    x = image_batch(batch=2)
    means = image_batch(batch=3)
    with pytest.raises(tf.errors.InvalidArgumentError):
        vae.gaussian_nll_per_example(x, means)


def test_rejects_posterior_parameter_shape_mismatch():
    mu = tf.zeros((2, 4), dtype=tf.float32)
    logvar = tf.zeros((2, 5), dtype=tf.float32)
    with pytest.raises(tf.errors.InvalidArgumentError):
        vae.diagonal_gaussian_kl_per_example(mu, logvar)


def test_rejects_nonfinite_posterior_parameter():
    mu = tf.zeros((2, 4), dtype=tf.float32)
    logvar = tf.fill((2, 4), float("inf"))
    with pytest.raises(tf.errors.InvalidArgumentError):
        vae.diagonal_gaussian_kl_per_example(mu, logvar)


def test_rejects_overflowed_posterior_variance():
    mu = tf.zeros((2, 4), dtype=tf.float32)
    logvar = tf.fill((2, 4), 100.0)
    with pytest.raises(tf.errors.InvalidArgumentError):
        vae.diagonal_gaussian_kl_per_example(mu, logvar)


def test_rejects_reconstruction_kl_batch_mismatch():
    x = image_batch(batch=2)
    mu = tf.zeros((3, 4), dtype=tf.float32)
    logvar = tf.zeros_like(mu)
    with pytest.raises(tf.errors.InvalidArgumentError):
        vae.vae_objective(x, x, mu, logvar)
