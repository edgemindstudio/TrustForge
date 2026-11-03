"""Regression tests for governed GMM initialization scale precision."""

import numpy as np
import tensorflow as tf

from trustforge.models.paper01_revision_2026.gaussian_mixture_initialization import (
    _production_initial_scales,
    _positive_float64_ulp_distance,
    MAX_SCALE_ROUNDTRIP_ULP,
    MIN_VARIANCE,
)
from trustforge.models.paper01_revision_2026.gaussian_mixture_observation import (
    MIN_SCALE,
    transformed_scale,
)


REAL_MALEX_VARIANCES = np.asarray(
    [
        0.03045617589382461,
        0.029791375445950005,
        0.02985863510702843,
        0.029689035736364215,
        # Additional real MaleX residual witnesses:
        0.022754392158927272,
        0.024877103334942665,
        0.023113876311446645,
        0.02423179851578826,
    ],
    dtype=np.float64,
)


def _expected_scales(floored_variance):
    expected = np.sqrt(
        floored_variance,
        dtype=np.float64,
    )

    floor = np.float64(MIN_SCALE)

    expected[expected == floor] = np.nextafter(
        floor,
        np.float64(np.inf),
    )

    return expected


def test_real_malex_scale_roundtrip_within_governed_ulp():
    variances = np.full(
        (4800,),
        np.float64(0.05),
        dtype=np.float64,
    )

    variances[:len(REAL_MALEX_VARIANCES)] = REAL_MALEX_VARIANCES

    floored, raw, promoted = _production_initial_scales(
        variances
    )

    actual = transformed_scale(
        tf.constant(raw, dtype=tf.float64)
    ).numpy()

    expected = _expected_scales(floored)

    distances = _positive_float64_ulp_distance(
        actual,
        expected,
    )

    assert raw.shape == (4800,)
    assert raw.dtype == np.float64
    assert np.isfinite(raw).all()
    assert np.all(actual > MIN_SCALE)

    assert np.all(
        distances <= MAX_SCALE_ROUNDTRIP_ULP
    ), (
        "Initial GMM scales exceed governed ULP tolerance: "
        f"maximum={int(distances.max())}, "
        f"known_cases={distances[:len(REAL_MALEX_VARIANCES)].tolist()}"
    )


def test_floor_boundary_remains_valid():
    variances = np.full(
        (4800,),
        np.float64(MIN_VARIANCE),
        dtype=np.float64,
    )

    floored, raw, promoted = _production_initial_scales(
        variances
    )

    actual = transformed_scale(
        tf.constant(raw, dtype=tf.float64)
    ).numpy()

    expected = _expected_scales(floored)

    distances = _positive_float64_ulp_distance(
        actual,
        expected,
    )

    assert promoted == 4800
    assert np.isfinite(raw).all()
    assert np.all(actual > MIN_SCALE)
    assert np.all(
        distances <= MAX_SCALE_ROUNDTRIP_ULP
    )


def test_precision_selection_is_deterministic():
    variances = np.full(
        (4800,),
        np.float64(0.05),
        dtype=np.float64,
    )

    variances[:len(REAL_MALEX_VARIANCES)] = REAL_MALEX_VARIANCES

    first = _production_initial_scales(variances)
    second = _production_initial_scales(variances)

    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])

    assert first[2] == second[2]


def test_governed_initialization_precision_threshold():
    """Revision-2026 initialization permits at most two float64 ULPs."""
    assert MAX_SCALE_ROUNDTRIP_ULP == 2


def test_all_eight_recorded_malex_witnesses():
    """Both historical witness groups must satisfy the governed bound."""
    assert REAL_MALEX_VARIANCES.shape == (8,)
    assert np.unique(REAL_MALEX_VARIANCES).size == 8

    variances = np.full(
        (4800,),
        np.float64(0.05),
        dtype=np.float64,
    )
    variances[:8] = REAL_MALEX_VARIANCES

    floored, raw, promoted = _production_initial_scales(
        variances
    )

    actual = transformed_scale(
        tf.constant(raw, dtype=tf.float64)
    ).numpy()

    expected = _expected_scales(floored)

    distances = _positive_float64_ulp_distance(
        actual,
        expected,
    )

    assert promoted == 0
    assert np.isfinite(raw).all()
    assert np.all(actual > MIN_SCALE)
    assert np.all(
        distances[:8] <= MAX_SCALE_ROUNDTRIP_ULP
    )
