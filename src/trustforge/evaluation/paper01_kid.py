"""Paper 1 polynomial KID estimator mathematics.

Caller must establish class-balanced selection, fixed reference identities,
Inception feature provenance and preprocessing. No image loading or feature
extraction occurs here. Subset draws are not independent experiment seeds.
"""

import numpy as np


def _features(value):
    if (
        not isinstance(value, np.ndarray)
        or value.ndim != 2
        or value.shape[0] < 2
        or value.shape[1] < 1
        or value.dtype not in (np.dtype("float32"), np.dtype("float64"))
    ):
        raise ValueError("expected at least two floating-point feature rows")
    # Bound temporary validation arrays; do not convert the full feature set.
    for start in range(0, len(value), 200):
        if not np.isfinite(value[start:start + 200]).all():
            raise ValueError("nonfinite feature data")
    return value


def polynomial_mmd2(real_features, synthetic_features):
    """Unbiased MMD-squared with k(x,y)=(x.y/d + 1)^3.

    Computes one selected subset, capped at 200 rows per input.
    Negative unbiased estimates are retained.
    """
    real = _features(real_features)
    synthetic = _features(synthetic_features)
    if real.shape[1] != synthetic.shape[1]:
        raise ValueError("feature dimensions differ")
    if len(real) > 200 or len(synthetic) > 200:
        raise ValueError("estimator subsets must contain at most 200 rows")
    real = real.astype(np.float64, copy=False)
    synthetic = synthetic.astype(np.float64, copy=False)
    dimension = real.shape[1]
    m, n = len(real), len(synthetic)

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        rr = (real @ real.T / dimension + 1.0) ** 3
        ss = (synthetic @ synthetic.T / dimension + 1.0) ** 3
        rs = (real @ synthetic.T / dimension + 1.0) ** 3
        # Exclude diagonals directly rather than subtracting large totals.
        np.fill_diagonal(rr, 0.0)
        np.fill_diagonal(ss, 0.0)
        value = (
            rr.sum(dtype=np.float64) / (m * (m - 1))
            + ss.sum(dtype=np.float64) / (n * (n - 1))
            - 2.0 * rs.mean(dtype=np.float64)
        )
    if not np.isfinite(value):
        raise ValueError("nonfinite polynomial MMD estimate")
    return float(value)


def kid_subsets(real_features, synthetic_features):
    """Ten fixed-seed subset estimates from already selected feature sets.

    For each draw, sample real indices then synthetic indices using the same
    advancing default_rng(42). Sampling is without replacement within a draw;
    different draws may overlap. Persist returned indices with feature identities.
    """
    real = _features(real_features)
    synthetic = _features(synthetic_features)
    if real.shape != synthetic.shape:
        raise ValueError("selected real/synthetic feature sets must align in size")
    size = min(200, len(real))
    rng = np.random.default_rng(42)
    draws = []
    for index in range(10):
        real_indices = rng.choice(len(real), size=size, replace=False)
        synthetic_indices = rng.choice(len(synthetic), size=size, replace=False)
        estimate = polynomial_mmd2(
            real[real_indices], synthetic[synthetic_indices],
        )
        draws.append({
            "draw_index": index,
            "real_indices": real_indices.tolist(),
            "synthetic_indices": synthetic_indices.tolist(),
            "estimate": estimate,
        })
    mean = float(np.mean(
        [draw["estimate"] for draw in draws], dtype=np.float64,
    ))
    if not np.isfinite(mean):
        raise ValueError("nonfinite KID mean")
    return {
        "estimator": "unbiased_degree3_polynomial_mmd2",
        "gamma": 1.0 / real.shape[1],
        "coef0": 1.0,
        "degree": 3,
        "subset_seed": 42,
        "subset_size": size,
        "subset_count": 10,
        "real_count": len(real),
        "synthetic_count": len(synthetic),
        "feature_dimension": real.shape[1],
        "draws": draws,
        "mean": mean,
    }
