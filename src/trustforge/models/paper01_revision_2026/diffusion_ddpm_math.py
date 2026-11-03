"""Deterministic DDPM coefficient and reverse-step mathematics.

No training, model loading, random-state mutation, checkpoint I/O,
or execution authorization is performed by this module.

A single explicitly supplied noise tensor controls the stochastic
reverse transition. The final transition (t=0) adds no noise.
"""

from __future__ import annotations

import numpy as np


def linear_ddpm_schedule(
    *,
    timesteps,
    beta_start=1e-4,
    beta_end=2e-2,
):
    """Return validated per-step and cumulative DDPM coefficients."""
    if type(timesteps) is not int or timesteps < 2:
        raise ValueError("timesteps must be an integer >= 2")

    for value in (beta_start, beta_end):
        if type(value) not in (float, int):
            raise TypeError("beta bounds must be numeric")
        if not np.isfinite(value):
            raise ValueError("nonfinite beta bound")

    if not 0 < beta_start <= beta_end < 1:
        raise ValueError("invalid beta range")

    betas = np.linspace(
        beta_start, beta_end, timesteps, dtype=np.float64
    )
    alphas = 1.0 - betas
    alpha_bars = np.cumprod(alphas)

    if not (
        np.all(np.isfinite(alpha_bars))
        and np.all(alpha_bars > 0)
        and np.all(alpha_bars < 1)
    ):
        raise ValueError("invalid cumulative diffusion schedule")

    for array in (betas, alphas, alpha_bars):
        array.setflags(write=False)

    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_bars": alpha_bars,
        "timesteps": timesteps,
        "schedule": "linear",
    }


def ddpm_reverse_step(
    *,
    x_t,
    predicted_noise,
    transition_noise,
    timestep,
    schedule,
):
    """Compute a DDPM posterior-variance reverse transition.

    All input tensors must have identical four-dimensional shapes.
    `transition_noise` is supplied externally for reproducibility.
    """
    if not isinstance(schedule, dict):
        raise TypeError("schedule must be a dictionary")

    T = schedule.get("timesteps")
    if type(T) is not int or T < 2:
        raise ValueError("invalid schedule timestep count")

    if type(timestep) is not int or not 0 <= timestep < T:
        raise ValueError("reverse timestep out of range")

    x = np.asarray(x_t)
    eps = np.asarray(predicted_noise)
    z = np.asarray(transition_noise)

    if x.ndim != 4 or x.shape != eps.shape or x.shape != z.shape:
        raise ValueError("reverse tensors must share a rank-four shape")

    if any(a.dtype != np.dtype("float32") for a in (x, eps, z)):
        raise TypeError("reverse tensors must be float32")

    if not all(np.all(np.isfinite(a)) for a in (x, eps, z)):
        raise ValueError("nonfinite reverse tensor")

    coeffs = []
    for key in ("betas", "alphas", "alpha_bars"):
        array = np.asarray(schedule.get(key))
        if array.shape != (T,) or not np.all(np.isfinite(array)):
            raise ValueError(f"invalid schedule coefficients: {key}")
        coeffs.append(array)

    betas, alphas, bars = coeffs

    beta = float(betas[timestep])
    alpha = float(alphas[timestep])
    bar = float(bars[timestep])
    previous_bar = (
        float(bars[timestep - 1]) if timestep > 0 else 1.0
    )

    if not (
        0 < beta < 1
        and 0 < alpha < 1
        and 0 < bar < 1
        and 0 < previous_bar <= 1
    ):
        raise ValueError("invalid reverse transition coefficients")

    # Check that cumulative and individual coefficients agree.
    if not np.isclose(alpha + beta, 1.0, atol=1e-10):
        raise ValueError("inconsistent alpha/beta coefficients")
    if not np.isclose(
        bar, previous_bar * alpha, rtol=1e-7, atol=1e-10
    ):
        raise ValueError("inconsistent cumulative coefficients")

    mean = (
        x.astype(np.float64)
        - beta / np.sqrt(1.0 - bar) * eps.astype(np.float64)
    ) / np.sqrt(alpha)

    variance = beta * (1.0 - previous_bar) / (1.0 - bar)

    if not np.isfinite(variance) or variance < 0:
        raise ValueError("invalid reverse posterior variance")

    result = mean
    if timestep > 0:
        result = result + np.sqrt(variance) * z.astype(np.float64)

    if not np.all(np.isfinite(result)):
        raise ValueError("nonfinite reverse result")

    result32 = result.astype(np.float32)

    if not np.all(np.isfinite(result32)):
        raise ValueError("reverse result overflows float32")

    return result32


__all__ = ["linear_ddpm_schedule", "ddpm_reverse_step"]
