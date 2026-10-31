"""Exploratory paired bootstrap for unbiased RBM MMD-squared.

This module provides descriptive uncertainty estimates only.

Bootstrap percentile coverage is not generally established,
especially at the degenerate equality boundary.

No production equivalence decision is authorized.
"""

from __future__ import annotations

import numpy as np

from .restricted_boltzmann_paired_mmd_math import (
    paired_unbiased_rbf_mmd2,
)


def paired_unbiased_mmd_bootstrap(
    x: np.ndarray,
    y: np.ndarray,
    *,
    bandwidth: float,
    repetitions: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, object]:
    """Resample independent trajectory pairs as complete units.

    Rows of x and y must correspond to identical trajectory
    identities, in identical order.

    Distinct original trajectory pairs are assumed independent
    and identically distributed. Dependence within each pair
    is permitted.

    Returned percentile bounds are exploratory and must not
    be used for production convergence or equivalence decisions.
    """
    left = np.asarray(x)
    right = np.asarray(y)

    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("trajectory arrays must be rank two")

    if left.shape != right.shape:
        raise ValueError("paired trajectory arrays must match shape")

    if left.shape[0] < 2:
        raise ValueError("at least two trajectory pairs required")

    if type(repetitions) is not int or repetitions < 2:
        raise ValueError("repetitions must be an integer >= 2")

    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")

    if (
        not np.isscalar(confidence)
        or not np.isfinite(confidence)
        or not 0.0 < confidence < 1.0
    ):
        raise ValueError("confidence must lie strictly between 0 and 1")

    observed = paired_unbiased_rbf_mmd2(
        left,
        right,
        bandwidth=bandwidth,
    )

    n = left.shape[0]
    rng = np.random.default_rng(seed)
    bootstrap_values = np.empty(
        repetitions,
        dtype=np.float64,
    )

    for index in range(repetitions):
        selected = rng.integers(
            0,
            n,
            size=n,
        )

        bootstrap_values[index] = paired_unbiased_rbf_mmd2(
            left[selected],
            right[selected],
            bandwidth=bandwidth,
        )

    alpha = 1.0 - float(confidence)

    lower, upper = np.quantile(
        bootstrap_values,
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
        "estimator": "paired_unbiased_rbf_mmd2",
        "interval_type": "exploratory_percentile",
        "inference_approved": False,
        "decision": "NONE",
    }
