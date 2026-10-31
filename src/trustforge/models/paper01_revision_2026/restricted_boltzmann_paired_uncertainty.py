"""Exploratory paired-trajectory uncertainty for RBF MMD-squared.

This module provides no Gibbs-depth selection or authorization.
Rows represent matched, independent trajectory identities.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


_MATH_PATH = (
    Path(__file__).resolve().with_name(
        "restricted_boltzmann_distributional_math.py"
    )
)

_spec = importlib.util.spec_from_file_location(
    "_rbm_distributional_math_for_bootstrap",
    _MATH_PATH,
)
if _spec is None or _spec.loader is None:
    raise ImportError("Cannot load RBM distributional mathematics")

_math = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_math)


def paired_mmd_bootstrap(
    x: np.ndarray,
    y: np.ndarray,
    *,
    bandwidth: float,
    repetitions: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, object]:
    """Estimate MMD-squared variability by paired row resampling.

    Assumes different trajectory rows are independent.
    Resamples the same indices for x and y.

    A percentile interval is descriptive; coverage and suitability
    for equivalence decisions are not established here.
    """
    left = _math._validate_matrix(x, name="x")
    right = _math._validate_matrix(y, name="y")

    if left.shape != right.shape:
        raise ValueError("paired trajectory arrays must match shape")

    if left.shape[0] < 2:
        raise ValueError("at least two trajectories required")

    if type(repetitions) is not int or repetitions < 2:
        raise ValueError("repetitions must be an integer >= 2")

    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")

    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between 0 and 1")

    observed = _math.empirical_rbf_mmd2(
        left, right, bandwidth=bandwidth
    )

    rng = np.random.default_rng(seed)
    values = np.empty(repetitions, dtype=np.float64)
    n = left.shape[0]

    for index in range(repetitions):
        chosen = rng.integers(0, n, size=n)

        values[index] = _math.empirical_rbf_mmd2(
            left[chosen],
            right[chosen],
            bandwidth=bandwidth,
        )

    alpha = 1.0 - confidence

    lower, upper = np.quantile(
        values,
        [alpha / 2.0, 1.0 - alpha / 2.0],
    )

    return {
        "observed_mmd2": float(observed),
        "lower": float(lower),
        "upper": float(upper),
        "confidence": float(confidence),
        "repetitions": repetitions,
        "seed": seed,
        "trajectory_count": n,
        "resampling_unit": "paired_trajectory",
        "decision": "NONE",
    }


__all__ = ["paired_mmd_bootstrap"]
