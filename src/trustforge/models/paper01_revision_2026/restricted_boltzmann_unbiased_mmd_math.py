"""Exploratory unbiased RBF MMD-squared for independent samples.

The estimator excludes within-sample diagonals and may be negative.
For paired Gibbs-depth samples, the usual independent-sample
unbiasedness claim does not apply without further adjustment.

No RBM training, sampling, convergence decision, or authorization.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


_MATH_PATH = Path(__file__).resolve().with_name(
    "restricted_boltzmann_distributional_math.py"
)

_spec = importlib.util.spec_from_file_location(
    "_rbm_distributional_math_for_unbiased_mmd",
    _MATH_PATH,
)

if _spec is None or _spec.loader is None:
    raise ImportError("Cannot load distributional mathematics")

_math = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_math)


def independent_sample_unbiased_rbf_mmd2(
    x: np.ndarray,
    y: np.ndarray,
    *,
    bandwidth: float,
) -> float:
    """Unbiased two-sample RBF MMD-squared estimator.

    Assumes independent observations within and between groups.
    Not an approved estimator for prefix-coupled Gibbs depths.
    """
    left = _math._validate_matrix(x, name="x")
    right = _math._validate_matrix(y, name="y")

    m, d = left.shape
    n, other_d = right.shape

    if d != other_d:
        raise ValueError("feature dimensions must match")

    if m < 2 or n < 2:
        raise ValueError("each group requires at least two samples")

    kxx = _math.rbf_kernel_matrix(
        left, left, bandwidth=bandwidth
    )
    kyy = _math.rbf_kernel_matrix(
        right, right, bandwidth=bandwidth
    )
    kxy = _math.rbf_kernel_matrix(
        left, right, bandwidth=bandwidth
    )

    within_x = (
        np.sum(kxx, dtype=np.float64)
        - np.trace(kxx, dtype=np.float64)
    ) / (m * (m - 1))

    within_y = (
        np.sum(kyy, dtype=np.float64)
        - np.trace(kyy, dtype=np.float64)
    ) / (n * (n - 1))

    cross = np.mean(kxy, dtype=np.float64)

    value = float(within_x + within_y - 2.0 * cross)

    if not np.isfinite(value):
        raise ValueError("Unbiased MMD-squared is non-finite")

    return value


__all__ = ["independent_sample_unbiased_rbf_mmd2"]
