import numpy as np
import pytest
from scipy.linalg import sqrtm

from trustforge.evaluation.paper01_fid import (
    feature_fid, complete_class_fid,
)


def independent_fid(real, synthetic):
    a = np.cov(real, rowvar=False)
    b = np.cov(synthetic, rowvar=False)
    root = sqrtm(a @ b)
    assert np.max(np.abs(np.imag(root))) < 1e-8
    return float(
        np.sum((real.mean(axis=0) - synthetic.mean(axis=0)) ** 2)
        + np.trace(a + b - 2 * np.real(root))
    )


def test_agrees_with_independent_full_rank_reference():
    rng = np.random.default_rng(91)
    real = rng.normal(size=(31, 4))
    synthetic = rng.normal(size=(27, 4)) * [1, 2, 3, 4] + 0.3
    observed = feature_fid(real, synthetic)
    assert observed["fid"] == pytest.approx(
        independent_fid(real, synthetic), rel=1e-11, abs=1e-11,
    )
    assert observed["diagonal_regularization"] == 0


def test_singular_diagonal_covariances_have_analytic_distance():
    real = np.array([[-1., 0, 0], [1., 0, 0]])
    synthetic = np.array([[-2., 0, 0], [2., 0, 0]])
    result = feature_fid(real, synthetic)
    assert result["fid"] == pytest.approx(2.0)
    assert result["real_covariance_necessarily_singular"] is True


def test_identical_rank_deficient_features():
    x = np.arange(24, dtype=np.float64).reshape(3, 8)
    assert feature_fid(x, x)["fid"] == pytest.approx(0, abs=1e-10)


def test_constant_features_reduce_to_mean_distance():
    real = np.zeros((3, 4), dtype=np.float64)
    synthetic = np.ones((5, 4), dtype=np.float64)
    assert feature_fid(real, synthetic)["fid"] == pytest.approx(4)


def test_symmetry_and_common_orthogonal_transform():
    rng = np.random.default_rng(19)
    real = rng.normal(size=(7, 5))
    synthetic = rng.normal(size=(9, 5))
    q, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    value = feature_fid(real, synthetic)["fid"]
    assert feature_fid(synthetic, real)["fid"] == pytest.approx(value)
    assert feature_fid(real @ q, synthetic @ q)["fid"] == pytest.approx(value)


def test_complete_macro_uses_equal_class_weights():
    real = np.zeros((6, 3), dtype=np.float64)
    synthetic = real.copy()
    labels = np.array([0, 0, 1, 1, 1, 1])
    synthetic[labels == 0] = 1
    synthetic[labels == 1] = 2
    result = complete_class_fid(
        real, synthetic, real_labels=labels,
        synthetic_labels=labels, class_count=2,
    )
    assert result["macro_cfid"] == pytest.approx((3 + 12) / 2)
    assert set(result["per_class"]) == {"0", "1"}


def test_missing_class_prevents_macro():
    x = np.zeros((4, 3), dtype=np.float64)
    labels = np.zeros(4, dtype=np.int64)
    with pytest.raises(ValueError):
        complete_class_fid(
            x, x, real_labels=labels, synthetic_labels=labels, class_count=2,
        )


@pytest.mark.parametrize("bad", [
    np.zeros((1, 3)),
    np.zeros((3, 3), dtype=np.int64),
    np.full((3, 3), np.nan),
    np.full((3, 3), np.inf),
])
def test_invalid_features_rejected(bad):
    with pytest.raises(ValueError):
        feature_fid(bad, np.zeros((3, 3)))


def test_dimension_mismatch_rejected():
    with pytest.raises(ValueError):
        feature_fid(np.zeros((3, 2)), np.zeros((3, 3)))
