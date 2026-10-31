"""Regression tests for exploratory paired-unbiased MMD uncertainty."""

import numpy as np
import pytest

from trustforge.models.paper01_revision_2026.restricted_boltzmann_paired_mmd_math import (
    paired_unbiased_rbf_mmd2,
)
from trustforge.models.paper01_revision_2026.restricted_boltzmann_paired_uncertainty import (
    paired_mmd_bootstrap,
)
from trustforge.models.paper01_revision_2026.restricted_boltzmann_paired_unbiased_uncertainty import (
    paired_unbiased_mmd_bootstrap,
)


def example_pairs():
    rng = np.random.default_rng(12345)
    x = rng.normal(size=(12, 3))
    y = 0.75 * x + 0.25 * rng.normal(size=(12, 3))
    return x, y


def test_observed_statistic_uses_paired_unbiased_estimator():
    x, y = example_pairs()

    actual = paired_unbiased_mmd_bootstrap(
        x, y, bandwidth=1.5, repetitions=12, seed=7
    )

    expected = paired_unbiased_rbf_mmd2(
        x, y, bandwidth=1.5
    )

    assert actual["observed_mmd2"] == pytest.approx(expected)


def test_bootstrap_resamples_complete_pairs():
    x, y = example_pairs()
    repetitions = 15
    seed = 19
    bandwidth = 1.5

    actual = paired_unbiased_mmd_bootstrap(
        x,
        y,
        bandwidth=bandwidth,
        repetitions=repetitions,
        seed=seed,
    )

    rng = np.random.default_rng(seed)
    manual = []

    for _ in range(repetitions):
        selected = rng.integers(0, len(x), size=len(x))
        manual.append(
            paired_unbiased_rbf_mmd2(
                x[selected],
                y[selected],
                bandwidth=bandwidth,
            )
        )

    lower, upper = np.quantile(manual, [0.025, 0.975])

    assert actual["lower"] == pytest.approx(lower)
    assert actual["upper"] == pytest.approx(upper)


def test_seeded_execution_is_deterministic():
    x, y = example_pairs()

    kwargs = dict(
        bandwidth=1.2,
        repetitions=20,
        seed=2026,
    )

    first = paired_unbiased_mmd_bootstrap(x, y, **kwargs)
    second = paired_unbiased_mmd_bootstrap(x, y, **kwargs)

    assert first == second


def test_original_biased_bootstrap_is_preserved():
    x, y = example_pairs()

    original = paired_mmd_bootstrap(
        x, y, bandwidth=1.5, repetitions=12, seed=7
    )

    revised = paired_unbiased_mmd_bootstrap(
        x, y, bandwidth=1.5, repetitions=12, seed=7
    )

    assert original["decision"] == "NONE"
    assert revised["decision"] == "NONE"
    assert revised["estimator"] == "paired_unbiased_rbf_mmd2"
    assert not np.isclose(
        original["observed_mmd2"],
        revised["observed_mmd2"],
        atol=1e-12,
        rtol=0,
    )


def test_interval_cannot_authorize_inference():
    x, y = example_pairs()

    result = paired_unbiased_mmd_bootstrap(
        x, y, bandwidth=1.5, repetitions=10, seed=1
    )

    assert result["inference_approved"] is False
    assert result["decision"] == "NONE"
    assert result["interval_type"] == "exploratory_percentile"
    assert result["resampling_unit"] == "paired_trajectory"


@pytest.mark.parametrize(
    "repetitions",
    [0, 1, -1, 2.5, True],
)
def test_invalid_repetitions_rejected(repetitions):
    x, y = example_pairs()

    with pytest.raises(ValueError):
        paired_unbiased_mmd_bootstrap(
            x, y, bandwidth=1.0,
            repetitions=repetitions, seed=1
        )


@pytest.mark.parametrize("seed", [-1, 2.5, True])
def test_invalid_seed_rejected(seed):
    x, y = example_pairs()

    with pytest.raises(ValueError):
        paired_unbiased_mmd_bootstrap(
            x, y, bandwidth=1.0,
            repetitions=5, seed=seed
        )


@pytest.mark.parametrize(
    "confidence",
    [0.0, 1.0, -0.1, 1.1, float("nan")],
)
def test_invalid_confidence_rejected(confidence):
    x, y = example_pairs()

    with pytest.raises(ValueError):
        paired_unbiased_mmd_bootstrap(
            x, y, bandwidth=1.0,
            repetitions=5, seed=1,
            confidence=confidence
        )


def test_insufficient_pairs_rejected():
    x, y = example_pairs()

    with pytest.raises(ValueError):
        paired_unbiased_mmd_bootstrap(
            x[:1], y[:1],
            bandwidth=1.0, repetitions=5, seed=1
        )


def test_shape_mismatch_rejected():
    x, y = example_pairs()

    with pytest.raises(ValueError):
        paired_unbiased_mmd_bootstrap(
            x, y[:-1],
            bandwidth=1.0, repetitions=5, seed=1
        )


def test_nonfinite_input_rejected():
    x, y = example_pairs()
    x[0, 0] = np.nan

    with pytest.raises(ValueError):
        paired_unbiased_mmd_bootstrap(
            x, y,
            bandwidth=1.0, repetitions=5, seed=1
        )
