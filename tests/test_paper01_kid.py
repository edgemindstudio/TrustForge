"""Independent estimator reference and deterministic subset checks."""

import numpy as np
import pytest

from trustforge.evaluation.paper01_kid import polynomial_mmd2, kid_subsets


def direct_reference(x, y):
    def kernel(a, b):
        return (sum(float(u) * float(v) for u, v in zip(a, b)) / len(a) + 1) ** 3
    m, n = len(x), len(y)
    rr = sum(kernel(x[i], x[j]) for i in range(m) for j in range(m) if i != j)
    ss = sum(kernel(y[i], y[j]) for i in range(n) for j in range(n) if i != j)
    rs = sum(kernel(a, b) for a in x for b in y)
    return rr / (m * (m - 1)) + ss / (n * (n - 1)) - 2 * rs / (m * n)


def test_matches_direct_unbiased_reference():
    x = np.array([[0, 1], [2, -1], [1, 3]], dtype=np.float64)
    y = np.array([[1, 0], [3, 1], [-1, 2], [2, 2]], dtype=np.float64)
    assert polynomial_mmd2(x, y) == pytest.approx(
        direct_reference(x, y), rel=1e-12, abs=1e-12,
    )


def test_symmetry_and_input_preservation():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(4, 3)).astype(np.float32)
    y = rng.normal(size=(5, 3)).astype(np.float32)
    before_x, before_y = x.copy(), y.copy()
    assert polynomial_mmd2(x, y) == pytest.approx(polynomial_mmd2(y, x))
    np.testing.assert_array_equal(x, before_x)
    np.testing.assert_array_equal(y, before_y)


def test_negative_unbiased_estimate_is_retained():
    x = np.array([[0], [1]], dtype=np.float64)
    assert polynomial_mmd2(x, x) == pytest.approx(-3.5)


@pytest.mark.parametrize("fault", [
    "singleton", "nonfinite", "integer", "dimension", "oversized",
])
def test_invalid_estimator_inputs(fault):
    x = np.zeros((3, 2), dtype=np.float64)
    y = x.copy()
    if fault == "singleton":
        x = x[:1]
    elif fault == "nonfinite":
        x[0, 0] = np.nan
    elif fault == "integer":
        x = x.astype(np.int64)
    elif fault == "dimension":
        y = np.zeros((3, 3), dtype=np.float64)
    else:
        x = np.zeros((201, 2), dtype=np.float64)
    with pytest.raises(ValueError):
        polynomial_mmd2(x, y)


def test_subset_replay_and_recorded_mean():
    rng = np.random.default_rng(8)
    x = rng.normal(size=(205, 3)).astype(np.float32)
    y = rng.normal(size=(205, 3)).astype(np.float32)
    result = kid_subsets(x, y)
    assert result == kid_subsets(x, y)
    assert result["subset_size"] == 200
    assert len(result["draws"]) == 10
    for draw in result["draws"]:
        assert len(set(draw["real_indices"])) == 200
        assert len(set(draw["synthetic_indices"])) == 200
        assert draw["estimate"] == polynomial_mmd2(
            x[draw["real_indices"]], y[draw["synthetic_indices"]],
        )
    assert result["mean"] == pytest.approx(
        np.mean([draw["estimate"] for draw in result["draws"]]),
    )


def test_small_feature_sets_use_all_rows():
    x = np.array([[0], [1]], dtype=np.float64)
    result = kid_subsets(x, x)
    assert result["subset_size"] == 2
    assert result["mean"] == pytest.approx(-3.5)
