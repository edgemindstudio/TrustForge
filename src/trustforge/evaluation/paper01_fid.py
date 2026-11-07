"""Empirical feature-space FID and complete class-macro cFID.

Uses sample covariances (N-1 denominator) and SVD covariance factors.
No diagonal regularization, image processing or provenance authentication.
Finite-sample and singular-covariance limitations remain reportable.
"""

import math
import numpy as np


def _features(value):
    if (
        not isinstance(value, np.ndarray)
        or value.ndim != 2
        or value.shape[0] < 2
        or value.shape[1] < 1
        or value.dtype not in (np.dtype("float32"), np.dtype("float64"))
        or not np.isfinite(value).all()
    ):
        raise ValueError("finite floating feature matrix with >=2 rows required")
    return value.astype(np.float64, copy=False)


def _factor(features):
    mean = features.mean(axis=0)
    centered = (features - mean) / math.sqrt(len(features) - 1)
    _, singular, right = np.linalg.svd(centered, full_matrices=False)
    # factor.T @ factor equals the empirical covariance.
    factor = singular[:, None] * right
    return mean, factor, float(np.sum(singular ** 2))


def feature_fid(real_features, synthetic_features):
    """Return FID plus explicit estimator and numerical-handling evidence."""
    real = _features(real_features)
    synthetic = _features(synthetic_features)
    if real.shape[1] != synthetic.shape[1]:
        raise ValueError("feature dimensions differ")

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        real_mean, real_factor, real_trace = _factor(real)
        synthetic_mean, synthetic_factor, synthetic_trace = _factor(synthetic)
        mean_term = float(np.sum((real_mean - synthetic_mean) ** 2))
        cross = float(np.linalg.svd(
            real_factor @ synthetic_factor.T, compute_uv=False,
        ).sum())
        raw = mean_term + real_trace + synthetic_trace - 2 * cross
        scale = max(1.0, mean_term + real_trace + synthetic_trace + 2 * cross)
        tolerance = 64 * np.finfo(np.float64).eps * real.shape[1] * scale

    if not math.isfinite(raw) or not math.isfinite(tolerance):
        raise ValueError("nonfinite FID calculation")
    if raw < -tolerance:
        raise ValueError("negative FID exceeds roundoff tolerance")
    return {
        "fid": max(0.0, raw),
        "raw_fid": raw,
        "roundoff_correction_applied": raw < 0,
        "roundoff_tolerance": tolerance,
        "estimator": "empirical_gaussian_frechet_distance",
        "covariance_denominator": "sample_count_minus_one",
        "cross_term_method": "svd_covariance_factor_nuclear_norm",
        "diagonal_regularization": 0.0,
        "feature_dimension": real.shape[1],
        "real_count": len(real),
        "synthetic_count": len(synthetic),
        "real_covariance_necessarily_singular": len(real) <= real.shape[1],
        "synthetic_covariance_necessarily_singular":
            len(synthetic) <= synthetic.shape[1],
    }


def _labels(labels, count, classes):
    if (
        not isinstance(labels, np.ndarray)
        or labels.shape != (count,)
        or labels.dtype.kind not in "iu"
        or np.any(labels < 0)
        or np.any(labels >= classes)
    ):
        raise ValueError("authoritative ordered integer labels required")
    return labels


def complete_class_fid(
    real_features, synthetic_features, *,
    real_labels, synthetic_labels, class_count,
):
    """Require all classes; propagate every class failure."""
    real = _features(real_features)
    synthetic = _features(synthetic_features)
    if type(class_count) is not int or class_count < 2:
        raise ValueError("class count must be an integer >=2")
    if real.shape[1] != synthetic.shape[1]:
        raise ValueError("feature dimensions differ")
    real_labels = _labels(real_labels, len(real), class_count)
    synthetic_labels = _labels(synthetic_labels, len(synthetic), class_count)
    per_class = {}
    for class_id in range(class_count):
        r = real[real_labels == class_id]
        s = synthetic[synthetic_labels == class_id]
        if len(r) < 2 or len(s) < 2:
            raise ValueError("every class requires >=2 rows per condition")
        per_class[str(class_id)] = feature_fid(r, s)
    return {
        "per_class": per_class,
        "macro_cfid": float(np.mean([
            per_class[str(c)]["fid"] for c in range(class_count)
        ])),
        "class_count": class_count,
        "class_coverage_complete": True,
    }
