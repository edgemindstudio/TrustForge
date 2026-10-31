"""Regression tests for exploratory paired-trajectory RBF MMD."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src/trustforge/models/paper01_revision_2026"
    / "restricted_boltzmann_paired_mmd_math.py"
)

SPEC = importlib.util.spec_from_file_location(
    "rbm_paired_mmd_under_test",
    MODULE_PATH,
)

assert SPEC is not None
assert SPEC.loader is not None

module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)

fn = module.paired_unbiased_rbf_mmd2


def test_analytical_two_trajectory_result():
    x = np.array([[0.0], [1.0]])
    y = np.array([[2.0], [3.0]])

    k = lambda a, b: np.exp(-0.5 * (a - b) ** 2)

    expected = (
        k(0.0, 1.0)
        + k(2.0, 3.0)
        - k(0.0, 3.0)
        - k(1.0, 2.0)
    )

    assert fn(x, y, bandwidth=1.0) == pytest.approx(
        expected, abs=1e-12
    )


def test_identical_paired_states_have_zero_mmd():
    x = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    assert fn(x, x, bandwidth=1.0) == pytest.approx(
        0.0, abs=1e-12
    )


def test_estimator_is_symmetric():
    rng = np.random.default_rng(2026)
    x = rng.uniform(size=(12, 5))
    y = rng.uniform(size=(12, 5))

    assert fn(x, y, bandwidth=2.0) == pytest.approx(
        fn(y, x, bandwidth=2.0),
        abs=1e-12,
    )


def test_controlled_alternative_is_finite_and_repeatable():
    rng = np.random.default_rng(42)
    x = rng.uniform(size=(32, 12))
    y = np.clip(x + 0.1, 0.0, 1.0)

    first = fn(x, y, bandwidth=2.0)
    second = fn(x, y, bandwidth=2.0)

    assert np.isfinite(first)
    assert first == second


def test_identical_point_mass_pairs_have_zero_mmd():
    x = np.full((8, 4), 0.25)

    assert fn(x, x, bandwidth=1.0) == pytest.approx(
        0.0, abs=1e-12
    )


def test_row_permutation_applied_to_both_groups_is_invariant():
    rng = np.random.default_rng(91)
    x = rng.uniform(size=(16, 6))
    y = rng.uniform(size=(16, 6))
    order = rng.permutation(len(x))

    original = fn(x, y, bandwidth=2.0)
    permuted = fn(
        x[order], y[order], bandwidth=2.0
    )

    assert original == pytest.approx(
        permuted, abs=1e-12
    )


def test_mismatched_trajectory_counts_rejected():
    x = np.ones((3, 2))
    y = np.ones((4, 2))

    with pytest.raises(ValueError):
        fn(x, y, bandwidth=1.0)


def test_mismatched_feature_dimensions_rejected():
    x = np.ones((3, 2))
    y = np.ones((3, 4))

    with pytest.raises(ValueError):
        fn(x, y, bandwidth=1.0)


def test_single_trajectory_rejected():
    x = np.ones((1, 2))

    with pytest.raises(ValueError):
        fn(x, x, bandwidth=1.0)


@pytest.mark.parametrize(
    "bandwidth",
    [0.0, -1.0, float("nan"), float("inf")],
)
def test_invalid_bandwidth_rejected(bandwidth):
    x = np.ones((3, 2))

    with pytest.raises(ValueError):
        fn(x, x, bandwidth=bandwidth)


def test_nonfinite_input_rejected():
    x = np.array([[0.0], [float("nan")]])
    y = np.ones((2, 1))

    with pytest.raises(ValueError):
        fn(x, y, bandwidth=1.0)


def test_complex_input_rejected():
    x = np.ones((3, 2), dtype=np.complex128)
    y = np.ones((3, 2))

    with pytest.raises(TypeError):
        fn(x, y, bandwidth=1.0)


def test_input_arrays_are_not_modified():
    rng = np.random.default_rng(17)
    x = rng.uniform(size=(8, 4))
    y = rng.uniform(size=(8, 4))

    original_x = x.copy()
    original_y = y.copy()

    fn(x, y, bandwidth=1.0)

    np.testing.assert_array_equal(x, original_x)
    np.testing.assert_array_equal(y, original_y)
