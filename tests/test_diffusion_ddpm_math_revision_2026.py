"""Focused tests for governed DDPM reverse mathematics."""

import numpy as np
import pytest

from trustforge.models.paper01_revision_2026.diffusion_ddpm_math import (
    linear_ddpm_schedule,
    ddpm_reverse_step,
)


def fixtures():
    schedule = linear_ddpm_schedule(timesteps=10)
    x = np.full((2, 4, 4, 1), 0.3, dtype=np.float32)
    eps = np.full_like(x, 0.1)
    z = np.full_like(x, -0.2)
    return schedule, x, eps, z


def test_schedule_is_finite_and_monotone():
    s, *_ = fixtures()
    assert len(s["betas"]) == 10
    assert np.all(np.diff(s["betas"]) > 0)
    assert np.all(np.diff(s["alpha_bars"]) < 0)


def test_reverse_step_matches_independent_formula():
    s, x, eps, z = fixtures()
    t = 7
    beta = s["betas"][t]
    alpha = s["alphas"][t]
    bar = s["alpha_bars"][t]
    prev = s["alpha_bars"][t - 1]

    mean = (x - beta / np.sqrt(1 - bar) * eps) / np.sqrt(alpha)
    var = beta * (1 - prev) / (1 - bar)
    expected = mean + np.sqrt(var) * z

    actual = ddpm_reverse_step(
        x_t=x,
        predicted_noise=eps,
        transition_noise=z,
        timestep=t,
        schedule=s,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_final_transition_has_no_stochastic_noise():
    s, x, eps, z = fixtures()
    a = ddpm_reverse_step(
        x_t=x, predicted_noise=eps,
        transition_noise=z, timestep=0, schedule=s,
    )
    b = ddpm_reverse_step(
        x_t=x, predicted_noise=eps,
        transition_noise=np.zeros_like(z),
        timestep=0, schedule=s,
    )
    np.testing.assert_array_equal(a, b)


def test_repeatability_with_identical_noise():
    s, x, eps, z = fixtures()
    kwargs = dict(
        x_t=x, predicted_noise=eps,
        transition_noise=z, timestep=5, schedule=s,
    )
    np.testing.assert_array_equal(
        ddpm_reverse_step(**kwargs),
        ddpm_reverse_step(**kwargs),
    )


def test_bad_schedule_rejected():
    with pytest.raises(ValueError):
        linear_ddpm_schedule(timesteps=1)
    with pytest.raises(ValueError):
        linear_ddpm_schedule(
            timesteps=10, beta_start=0.2, beta_end=0.1
        )


def test_nonfinite_tensor_rejected():
    s, x, eps, z = fixtures()
    x[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        ddpm_reverse_step(
            x_t=x, predicted_noise=eps,
            transition_noise=z, timestep=1, schedule=s,
        )


def test_shape_mismatch_rejected():
    s, x, eps, z = fixtures()
    with pytest.raises(ValueError, match="rank-four"):
        ddpm_reverse_step(
            x_t=x, predicted_noise=eps,
            transition_noise=z[:1],
            timestep=1, schedule=s,
        )


def test_schedule_coefficient_tampering_rejected():
    s, x, eps, z = fixtures()
    altered = dict(s)
    altered["alphas"] = s["alphas"].copy()
    altered["alphas"][5] = 0.8
    with pytest.raises(ValueError, match="inconsistent"):
        ddpm_reverse_step(
            x_t=x, predicted_noise=eps,
            transition_noise=z, timestep=5,
            schedule=altered,
        )
