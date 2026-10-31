"""Exploratory paired-trajectory RBF MMD-squared estimator.

This estimator compares two observations from each independent
trajectory and excludes within-trajectory cross-depth kernel terms.

It does not establish Gibbs-chain convergence, distributional
equivalence, or production sampling authorization.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


_MATH_PATH = Path(__file__).resolve().with_name(
    "restricted_boltzmann_distributional_math.py"
)

_spec = importlib.util.spec_from_file_location(
    "_rbm_distributional_math_for_paired_mmd",
    _MATH_PATH,
)

if _spec is None or _spec.loader is None:
    raise ImportError("Cannot load RBM distributional mathematics")

_math = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_math)


def paired_unbiased_rbf_mmd2(
    x: np.ndarray,
    y: np.ndarray,
    *,
    bandwidth: float,
) -> float:
    """Estimate marginal MMD-squared from independent trajectory pairs.

    The rows of x and y must correspond to identical trajectory
    identities in identical order.

    Different rows are assumed independent and identically
    distributed. Dependence within each row pair is permitted.

    Finite-sample estimates may be negative.
    """
    left = _math._validate_matrix(x, name="x")
    right = _math._validate_matrix(y, name="y")

    if left.shape != right.shape:
        raise ValueError(
            "paired trajectory arrays must have matching shapes"
        )

    n = left.shape[0]

    if n < 2:
        raise ValueError(
            "at least two independent trajectory pairs required"
        )

    kxx = _math.rbf_kernel_matrix(
        left, left, bandwidth=bandwidth
    )

    kyy = _math.rbf_kernel_matrix(
        right, right, bandwidth=bandwidth
    )

    kxy = _math.rbf_kernel_matrix(
        left, right, bandwidth=bandwidth
    )

    terms = kxx + kyy - kxy - kxy.T

    np.fill_diagonal(terms, 0.0)

    result = float(
        np.sum(terms, dtype=np.float64) / (n * (n - 1))
    )

    if not np.isfinite(result):
        raise ValueError("paired MMD-squared is non-finite")

    return result


__all__ = ["paired_unbiased_rbf_mmd2"]
