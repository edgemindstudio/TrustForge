"""Regression for the measured CIC step5 spline2 boundary failure."""

import json
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from trustforge.models.paper01_revision_2026 import masked_autoflow_spline as spline


def captured():
    path = (
        Path(__file__).resolve().parents[1]
        / "studies/paper01_benchmark/audits/revision_2026/"
          "maf_cic_boundary_311001.json"
    )
    return json.loads(path.read_text())["report"]["examples"][0]


def parameters(values):
    example = captured()
    shape = (1, len(values), 8)
    widths = tf.constant(np.broadcast_to(
        np.asarray(example["raw_widths"], np.float32), shape,
    ).copy())
    controls = tf.constant(np.broadcast_to(
        np.asarray(example["raw_slope_controls"], np.float32), shape,
    ).copy())
    return tf.constant([values], tf.float32), widths, controls


def test_captured_violation_now_stays_inside_interval():
    example = captured()
    x, widths, controls = parameters([example["input"]])
    y, forward_ld = spline.forward(x, widths, controls)
    restored, inverse_ld = spline.inverse(y, widths, controls)
    assert example["output"] > 1.0
    assert 0.0 <= float(y.numpy()[0, 0]) <= 1.0

    # Independent scalar calculation from recorded represented knots.
    x0, x1 = example["x_knots"][-2:]
    y0, y1 = example["y_knots"][-2:]
    reference = y0 + (y1 - y0) * (example["input"] - x0) / (x1 - x0)
    np.testing.assert_allclose(y.numpy(), reference, rtol=0, atol=2e-7)
    np.testing.assert_allclose(restored.numpy(), x.numpy(), rtol=0, atol=2e-7)
    np.testing.assert_allclose(
        (forward_ld + inverse_ld).numpy(), 0,
        rtol=0, atol=spline.LOGDET_CANCELLATION_TOLERANCE,
    )


@pytest.mark.parametrize("direction", ["forward", "inverse"])
def test_endpoint_neighborhood_is_bounded_and_monotone(direction):
    values = np.unique(np.concatenate([
        np.linspace(0, 1, 257, dtype=np.float32),
        np.array([
            np.nextafter(np.float32(0), np.float32(1)),
            np.nextafter(np.float32(1), np.float32(0)),
        ], dtype=np.float32),
    ]))
    x, widths, controls = parameters(values)
    output, logdet = getattr(spline, direction)(x, widths, controls)
    observed = output.numpy()[0]
    assert observed[0] == 0.0
    assert observed[-1] == 1.0
    assert np.all((observed >= 0) & (observed <= 1))
    assert np.all(np.diff(observed) >= 0)
    assert np.isfinite(logdet.numpy()).all()


def test_logdet_matches_input_derivative_away_from_knots():
    x, widths, controls = parameters([0.37])
    x = tf.Variable(x)
    with tf.GradientTape() as tape:
        y, logdet = spline.forward(x, widths, controls)
    derivative = tape.gradient(y, x)
    assert derivative is not None
    np.testing.assert_allclose(
        derivative.numpy(), tf.exp(logdet).numpy(),
        rtol=2e-5, atol=2e-6,
    )


@pytest.mark.parametrize("direction", ["forward", "inverse"])
def test_parameter_gradients_remain_finite(direction):
    x, widths, controls = parameters([0.37, 0.73])
    widths = tf.Variable(widths)
    controls = tf.Variable(controls)
    with tf.GradientTape() as tape:
        output, logdet = getattr(spline, direction)(x, widths, controls)
        objective = tf.reduce_sum(output + logdet)
    gradients = tape.gradient(objective, [widths, controls])
    assert all(g is not None for g in gradients)
    assert all(np.isfinite(g.numpy()).all() for g in gradients)


@pytest.mark.parametrize("direction", ["forward", "inverse"])
@pytest.mark.parametrize("value", [
    float(np.nextafter(np.float32(0), np.float32(-1))),
    float(np.nextafter(np.float32(1), np.float32(2))),
])
def test_out_of_range_inputs_still_rejected(direction, value):
    x, widths, controls = parameters([value])
    with pytest.raises(tf.errors.InvalidArgumentError):
        getattr(spline, direction)(x, widths, controls)


def test_random_parameter_roundtrip_and_logdet_cancellation():
    rng = np.random.default_rng(42)
    values = rng.random((4, 256), dtype=np.float32)
    values[:, :4] = [
        0.0, 1.0,
        np.nextafter(np.float32(0), np.float32(1)),
        np.nextafter(np.float32(1), np.float32(0)),
    ]
    widths = tf.constant(rng.normal(
        0, 4, (4, 256, 8),
    ).astype(np.float32))
    controls = tf.constant(rng.normal(
        0, 4, (4, 256, 8),
    ).astype(np.float32))
    y, forward_ld = spline.forward(values, widths, controls)
    restored, inverse_ld = spline.inverse(y, widths, controls)
    assert np.all((y.numpy() >= 0) & (y.numpy() <= 1))
    np.testing.assert_allclose(
        restored.numpy(), values, rtol=0,
        atol=spline.ROUNDTRIP_TOLERANCE,
    )
    np.testing.assert_allclose(
        (forward_ld + inverse_ld).numpy(), 0, rtol=0,
        atol=spline.LOGDET_CANCELLATION_TOLERANCE,
    )
