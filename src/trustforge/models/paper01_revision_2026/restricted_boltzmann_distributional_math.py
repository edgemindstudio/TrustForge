"""Exploratory numerical primitives for prospective RBM diagnostics.

STATUS: mathematical prototype only.
No RBM sampling, training, convergence decision, or authorization.

All inputs are caller-provided mathematical arrays.
Bandwidth is explicit and is not selected by this module.
"""

from __future__ import annotations

import numpy as np


def _validate_matrix(value, *, name: str) -> np.ndarray:
    """Validate a finite, nonempty two-dimensional numeric matrix."""
    array = np.asarray(value)

    if array.ndim != 2:
        raise ValueError(f"{name} must be rank two")

    if array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError(f"{name} must be nonempty")

    if array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numeric values")

    array = np.asarray(array, dtype=np.float64)

    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")

    return array


def rbf_kernel_matrix(
    x: np.ndarray,
    y: np.ndarray,
    *,
    bandwidth: float,
) -> np.ndarray:
    """Compute a float64 Gaussian RBF cross-kernel matrix."""
    left = _validate_matrix(x, name="x")
    right = _validate_matrix(y, name="y")

    if left.shape[1] != right.shape[1]:
        raise ValueError("x and y must have matching feature widths")

    sigma = float(bandwidth)

    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("bandwidth must be finite and positive")

    # Scale before squaring to reduce overflow risk.
    left_scaled = left / sigma
    right_scaled = right / sigma

    if (
        not np.isfinite(left_scaled).all()
        or not np.isfinite(right_scaled).all()
    ):
        raise ValueError("scaled kernel inputs are non-finite")

    left_norm = np.sum(left_scaled ** 2, axis=1)
    right_norm = np.sum(right_scaled ** 2, axis=1)

    if (
        not np.isfinite(left_norm).all()
        or not np.isfinite(right_norm).all()
    ):
        raise ValueError("kernel input norms are non-finite")

    squared = (
        left_norm[:, None]
        + right_norm[None, :]
        - 2.0 * (left_scaled @ right_scaled.T)
    )

    squared = np.maximum(squared, 0.0)

    if not np.isfinite(squared).all():
        raise ValueError("kernel distances are non-finite")

    kernel = np.exp(-0.5 * squared)

    if not np.isfinite(kernel).all():
        raise ValueError("kernel values are non-finite")

    return kernel


def empirical_rbf_mmd2(
    x: np.ndarray,
    y: np.ndarray,
    *,
    bandwidth: float,
) -> float:
    """Compute biased empirical RBF MMD-squared.

    This mathematical statistic is not a convergence test.
    It does not select Gibbs depth or establish equivalence.
    """
    left = _validate_matrix(x, name="x")
    right = _validate_matrix(y, name="y")

    if left.shape[1] != right.shape[1]:
        raise ValueError("feature dimensions must match")

    kxx = rbf_kernel_matrix(
        left, left, bandwidth=bandwidth
    )
    kyy = rbf_kernel_matrix(
        right, right, bandwidth=bandwidth
    )
    kxy = rbf_kernel_matrix(
        left, right, bandwidth=bandwidth
    )

    value = float(
        np.mean(kxx, dtype=np.float64)
        + np.mean(kyy, dtype=np.float64)
        - 2.0 * np.mean(kxy, dtype=np.float64)
    )

    if not np.isfinite(value):
        raise ValueError("MMD-squared is non-finite")

    # Tiny negative results can arise from floating-point cancellation.
    if value < -1.0e-10:
        raise ValueError("MMD-squared is unexpectedly negative")

    return max(0.0, value)


__all__ = [
    "rbf_kernel_matrix",
    "empirical_rbf_mmd2",
]
