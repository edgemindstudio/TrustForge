"""Tests for exploratory RBM distributional mathematics only."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src/trustforge/models/paper01_revision_2026"
    / "restricted_boltzmann_distributional_math.py"
)

SPEC = importlib.util.spec_from_file_location(
    "rbm_distributional_math_under_test",
    MODULE_PATH,
)

assert SPEC is not None
assert SPEC.loader is not None

math_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(math_module)


def test_rbf_kernel_shape_symmetry_and_diagonal():
    x = np.array([[0., 0.], [1., 0.], [0., 1.]])
    kernel = math_module.rbf_kernel_matrix(
        x, x, bandwidth=1.0
    )
    assert kernel.shape == (3, 3)
    np.testing.assert_allclose(kernel, kernel.T)
    np.testing.assert_allclose(np.diag(kernel), 1.0)
    assert np.all(kernel >= 0.0)
    assert np.all(kernel <= 1.0)


def test_rbf_kernel_analytical_value():
    x = np.array([[0., 0.]])
    y = np.array([[1., 0.]])
    kernel = math_module.rbf_kernel_matrix(
        x, y, bandwidth=1.0
    )
    np.testing.assert_allclose(kernel[0, 0], np.exp(-0.5))


def test_identical_empirical_distributions_have_zero_mmd():
    x = np.array([[0., 0.], [1., 0.], [0., 1.]])
    observed = math_module.empirical_rbf_mmd2(
        x, x, bandwidth=1.0
    )
    assert observed == pytest.approx(0.0, abs=1e-12)


def test_identical_constant_samples_have_zero_mmd():
    x = np.full((8, 4), 0.25)
    observed = math_module.empirical_rbf_mmd2(
        x, x, bandwidth=2.0
    )
    assert observed == pytest.approx(0.0, abs=1e-12)


def test_singleton_mmd_matches_analytical_formula():
    x = np.array([[0., 0.]])
    y = np.array([[1., 0.]])
    expected = 2.0 * (1.0 - np.exp(-0.5))
    observed = math_module.empirical_rbf_mmd2(
        x, y, bandwidth=1.0
    )
    assert observed == pytest.approx(expected, abs=1e-12)


def test_mmd_is_symmetric():
    x = np.array([[0., 0.], [1., 0.]])
    y = np.array([[2., 2.], [3., 2.]])
    xy = math_module.empirical_rbf_mmd2(
        x, y, bandwidth=1.0
    )
    yx = math_module.empirical_rbf_mmd2(
        y, x, bandwidth=1.0
    )
    assert xy == pytest.approx(yx, abs=1e-12)


def test_distinct_empirical_distributions_have_positive_mmd():
    x = np.array([[0., 0.], [1., 0.]])
    y = np.array([[2., 2.], [3., 2.]])
    observed = math_module.empirical_rbf_mmd2(
        x, y, bandwidth=1.0
    )
    assert observed > 0.0


def test_random_shift_is_finite_and_repeatable():
    rng = np.random.default_rng(2026)
    x = rng.normal(size=(32, 16))
    y = rng.normal(loc=1.0, size=(32, 16))
    first = math_module.empirical_rbf_mmd2(
        x, y, bandwidth=4.0
    )
    second = math_module.empirical_rbf_mmd2(
        x, y, bandwidth=4.0
    )
    assert np.isfinite(first)
    assert first >= 0.0
    assert first == second


@pytest.mark.parametrize("bandwidth", [
    0.0, -1.0, float("nan"), float("inf")
])
def test_invalid_bandwidth_is_rejected(bandwidth):
    x = np.array([[0., 0.]])
    with pytest.raises(ValueError):
        math_module.empirical_rbf_mmd2(
            x, x, bandwidth=bandwidth
        )


def test_nonfinite_input_is_rejected():
    x = np.array([[float("nan"), 0.]])
    y = np.array([[0., 0.]])
    with pytest.raises(ValueError):
        math_module.empirical_rbf_mmd2(
            x, y, bandwidth=1.0
        )


def test_incompatible_dimensions_are_rejected():
    x = np.ones((3, 2))
    y = np.ones((3, 4))
    with pytest.raises(ValueError):
        math_module.empirical_rbf_mmd2(
            x, y, bandwidth=1.0
        )


def test_empty_inputs_are_rejected():
    x = np.empty((0, 2))
    y = np.ones((3, 2))
    with pytest.raises(ValueError):
        math_module.empirical_rbf_mmd2(
            x, y, bandwidth=1.0
        )


def test_complex_inputs_are_rejected():
    x = np.ones((3, 2), dtype=np.complex128)
    y = np.ones((3, 2))
    with pytest.raises(TypeError):
        math_module.empirical_rbf_mmd2(
            x, y, bandwidth=1.0
        )
