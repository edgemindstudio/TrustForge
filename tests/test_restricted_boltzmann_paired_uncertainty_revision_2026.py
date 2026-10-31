"""Regression tests for exploratory paired RBM MMD uncertainty."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src/trustforge/models/paper01_revision_2026"
    / "restricted_boltzmann_paired_uncertainty.py"
)

SPEC = importlib.util.spec_from_file_location(
    "rbm_paired_uncertainty_under_test",
    MODULE_PATH,
)

assert SPEC is not None
assert SPEC.loader is not None

module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def _evaluate(x, y, **overrides):
    arguments = {
        "bandwidth": 1.0,
        "repetitions": 40,
        "seed": 2026,
    }
    arguments.update(overrides)
    return module.paired_mmd_bootstrap(x, y, **arguments)


def test_identical_paired_samples_have_zero_mmd_and_interval():
    x = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ])

    result = _evaluate(x, x)

    assert result["observed_mmd2"] == pytest.approx(
        0.0, abs=1e-12
    )
    assert result["lower"] == pytest.approx(
        0.0, abs=1e-12
    )
    assert result["upper"] == pytest.approx(
        0.0, abs=1e-12
    )


def test_fixed_seed_produces_identical_result():
    rng = np.random.default_rng(91)
    x = rng.uniform(size=(12, 5))
    y = rng.uniform(size=(12, 5))

    first = _evaluate(x, y)
    second = _evaluate(x, y)

    assert first == second


def test_observed_mmd_is_nonnegative_and_interval_ordered():
    rng = np.random.default_rng(44)
    x = rng.uniform(size=(16, 8))
    y = np.clip(x + 0.1, 0.0, 1.0)

    result = _evaluate(x, y)

    assert result["observed_mmd2"] >= 0.0
    assert 0.0 <= result["lower"] <= result["upper"]
    assert np.isfinite(result["upper"])


def test_result_metadata_is_explicit_and_non_authorizing():
    x = np.array([[0.0], [1.0], [2.0]])

    result = _evaluate(x, x)

    assert result["trajectory_count"] == 3
    assert result["repetitions"] == 40
    assert result["seed"] == 2026
    assert result["confidence"] == pytest.approx(0.95)
    assert result["resampling_unit"] == "paired_trajectory"
    assert result["decision"] == "NONE"


def test_paired_resampling_preserves_identical_row_mapping():
    # x and y are identical under a trajectory-by-trajectory mapping.
    # Joint row resampling must preserve equality for every bootstrap draw.
    x = np.arange(24, dtype=np.float64).reshape(8, 3)

    result = _evaluate(x, x, repetitions=60)

    assert result["lower"] == pytest.approx(0.0, abs=1e-12)
    assert result["upper"] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("repetitions", [
    0, 1, -1, 2.5, True,
])
def test_invalid_repetitions_rejected(repetitions):
    x = np.ones((3, 2))
    with pytest.raises(ValueError):
        _evaluate(x, x, repetitions=repetitions)


@pytest.mark.parametrize("seed", [
    -1, 1.5, True,
])
def test_invalid_seed_rejected(seed):
    x = np.ones((3, 2))
    with pytest.raises(ValueError):
        _evaluate(x, x, seed=seed)


@pytest.mark.parametrize("confidence", [
    0.0, 1.0, -0.1, 1.1, float("nan"),
])
def test_invalid_confidence_rejected(confidence):
    x = np.ones((3, 2))
    with pytest.raises(ValueError):
        _evaluate(x, x, confidence=confidence)


def test_mismatched_trajectory_counts_rejected():
    x = np.ones((3, 2))
    y = np.ones((4, 2))

    with pytest.raises(ValueError):
        _evaluate(x, y)


def test_mismatched_feature_dimensions_rejected():
    x = np.ones((3, 2))
    y = np.ones((3, 4))

    with pytest.raises(ValueError):
        _evaluate(x, y)


def test_single_trajectory_rejected():
    x = np.ones((1, 2))

    with pytest.raises(ValueError):
        _evaluate(x, x)


def test_nonfinite_input_rejected():
    x = np.array([[0.0], [float("nan")]])
    y = np.ones((2, 1))

    with pytest.raises(ValueError):
        _evaluate(x, y)


def test_invalid_bandwidth_rejected():
    x = np.ones((3, 2))

    with pytest.raises(ValueError):
        _evaluate(x, x, bandwidth=0.0)
