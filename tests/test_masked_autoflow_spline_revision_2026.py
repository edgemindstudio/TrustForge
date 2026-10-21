import numpy as np
import tensorflow as tf

from trustforge.models.paper01_revision_2026.masked_autoflow_spline import (
    LOGDET_CANCELLATION_TOLERANCE,
    LOG_SLOPE_LIMIT,
    MAX_SEGMENT_SLOPE,
    MIN_BIN_WIDTH,
    MIN_SEGMENT_SLOPE,
    NUM_BINS,
    ROUNDTRIP_TOLERANCE,
    forward,
    inverse,
    relative_slopes_from_controls,
    spline_parameters,
)


def _parameters(
    batch=3,
    dimensions=192,
):
    tf.keras.utils.set_random_seed(
        42
    )

    raw_widths = tf.random.normal(
        (
            batch,
            dimensions,
            NUM_BINS,
        ),
        seed=42,
    )

    raw_slope_controls = tf.random.normal(
        (
            batch,
            dimensions,
            NUM_BINS,
        ),
        seed=43,
    )

    return (
        raw_widths,
        raw_slope_controls,
    )


def test_widths_heights_and_slopes_satisfy_governed_contract():
    (
        raw_widths,
        raw_slope_controls,
    ) = _parameters()

    relative_slopes = (
        relative_slopes_from_controls(
            raw_slope_controls
        )
    )

    (
        widths,
        heights,
        x_knots,
        y_knots,
    ) = spline_parameters(
        raw_widths,
        raw_slope_controls,
    )

    slopes = heights / widths

    assert LOG_SLOPE_LIMIT == 0.75

    assert bool(
        tf.reduce_all(
            widths >= MIN_BIN_WIDTH
        ).numpy()
    )

    assert bool(
        tf.reduce_all(
            heights > 0.0
        ).numpy()
    )

    assert bool(
        tf.reduce_all(
            relative_slopes > 0.0
        ).numpy()
    )

    assert bool(
        tf.reduce_all(
            tf.math.is_finite(
                relative_slopes
            )
        ).numpy()
    )

    np.testing.assert_allclose(
        tf.reduce_sum(
            widths,
            axis=-1,
        ).numpy(),
        1.0,
        atol=1e-6,
        rtol=0.0,
    )

    np.testing.assert_allclose(
        tf.reduce_sum(
            heights,
            axis=-1,
        ).numpy(),
        1.0,
        atol=1e-6,
        rtol=0.0,
    )

    assert float(
        tf.reduce_min(
            slopes
        ).numpy()
    ) >= (
        MIN_SEGMENT_SLOPE
        - 1e-6
    )

    assert float(
        tf.reduce_max(
            slopes
        ).numpy()
    ) <= (
        MAX_SEGMENT_SLOPE
        + 1e-6
    )

    np.testing.assert_array_equal(
        x_knots[..., 0].numpy(),
        np.zeros(
            x_knots.shape[:-1],
            dtype=np.float32,
        ),
    )

    np.testing.assert_array_equal(
        x_knots[..., -1].numpy(),
        np.ones(
            x_knots.shape[:-1],
            dtype=np.float32,
        ),
    )

    np.testing.assert_array_equal(
        y_knots[..., 0].numpy(),
        np.zeros(
            y_knots.shape[:-1],
            dtype=np.float32,
        ),
    )

    np.testing.assert_array_equal(
        y_knots[..., -1].numpy(),
        np.ones(
            y_knots.shape[:-1],
            dtype=np.float32,
        ),
    )


def test_exact_zero_and_one_are_preserved():
    raw_widths = tf.zeros(
        (1, 2, NUM_BINS),
        dtype=tf.float32,
    )

    raw_slope_controls = tf.zeros(
        (1, 2, NUM_BINS),
        dtype=tf.float32,
    )

    x = tf.constant(
        [[0.0, 1.0]],
        dtype=tf.float32,
    )

    y, log_det = forward(
        x,
        raw_widths,
        raw_slope_controls,
    )

    recovered, inverse_log_det = inverse(
        y,
        raw_widths,
        raw_slope_controls,
    )

    np.testing.assert_array_equal(
        y.numpy(),
        np.array(
            [[0.0, 1.0]],
            dtype=np.float32,
        ),
    )

    np.testing.assert_array_equal(
        recovered.numpy(),
        np.array(
            [[0.0, 1.0]],
            dtype=np.float32,
        ),
    )

    assert bool(
        tf.reduce_all(
            tf.math.is_finite(
                log_det
            )
        ).numpy()
    )

    assert bool(
        tf.reduce_all(
            tf.math.is_finite(
                inverse_log_det
            )
        ).numpy()
    )


def test_forward_and_inverse_remain_inside_closed_interval():
    (
        raw_widths,
        raw_slope_controls,
    ) = _parameters(
        batch=4,
        dimensions=192,
    )

    x = tf.random.stateless_uniform(
        (4, 192),
        seed=(10, 11),
        minval=0.0,
        maxval=1.0,
        dtype=tf.float32,
    )

    y, _ = forward(
        x,
        raw_widths,
        raw_slope_controls,
    )

    recovered, _ = inverse(
        y,
        raw_widths,
        raw_slope_controls,
    )

    assert float(
        tf.reduce_min(y).numpy()
    ) >= 0.0

    assert float(
        tf.reduce_max(y).numpy()
    ) <= 1.0

    assert float(
        tf.reduce_min(
            recovered
        ).numpy()
    ) >= 0.0

    assert float(
        tf.reduce_max(
            recovered
        ).numpy()
    ) <= 1.0


def test_forward_inverse_round_trip_and_logdet_cancel():
    (
        raw_widths,
        raw_slope_controls,
    ) = _parameters(
        batch=4,
        dimensions=192,
    )

    base = tf.linspace(
        0.0,
        1.0,
        192,
    )

    x = tf.tile(
        base[None, :],
        [4, 1],
    )

    y, log_det = forward(
        x,
        raw_widths,
        raw_slope_controls,
    )

    recovered, inverse_log_det = inverse(
        y,
        raw_widths,
        raw_slope_controls,
    )

    max_roundtrip_error = float(
        tf.reduce_max(
            tf.abs(
                recovered - x
            )
        ).numpy()
    )

    max_logdet_error = float(
        tf.reduce_max(
            tf.abs(
                log_det
                + inverse_log_det
            )
        ).numpy()
    )

    assert (
        max_roundtrip_error
        < ROUNDTRIP_TOLERANCE
    )

    assert (
        max_logdet_error
        < LOGDET_CANCELLATION_TOLERANCE
    )


def test_internal_knot_convention_is_deterministic():
    raw_widths = tf.zeros(
        (1, 1, NUM_BINS),
        dtype=tf.float32,
    )

    raw_slope_controls = tf.constant(
        [[[
            -1.0,
            -0.5,
            0.0,
            0.5,
            1.0,
            0.25,
            -0.25,
            0.75,
        ]]],
        dtype=tf.float32,
    )

    (
        _,
        _,
        x_knots,
        _,
    ) = spline_parameters(
        raw_widths,
        raw_slope_controls,
    )

    x = x_knots[
        :,
        :,
        4,
    ]

    y_a, log_det_a = forward(
        x,
        raw_widths,
        raw_slope_controls,
    )

    y_b, log_det_b = forward(
        x,
        raw_widths,
        raw_slope_controls,
    )

    recovered, inverse_log_det = inverse(
        y_a,
        raw_widths,
        raw_slope_controls,
    )

    np.testing.assert_array_equal(
        y_a.numpy(),
        y_b.numpy(),
    )

    np.testing.assert_array_equal(
        log_det_a.numpy(),
        log_det_b.numpy(),
    )

    np.testing.assert_allclose(
        recovered.numpy(),
        x.numpy(),
        atol=ROUNDTRIP_TOLERANCE,
        rtol=0.0,
    )

    np.testing.assert_allclose(
        (
            log_det_a
            + inverse_log_det
        ).numpy(),
        0.0,
        atol=LOGDET_CANCELLATION_TOLERANCE,
        rtol=0.0,
    )


def test_forward_values_logdet_and_gradients_are_finite():
    tf.keras.utils.set_random_seed(
        42
    )

    raw_widths = tf.Variable(
        tf.random.normal(
            (
                2,
                192,
                NUM_BINS,
            ),
            seed=42,
        )
    )

    raw_slope_controls = tf.Variable(
        tf.random.normal(
            (
                2,
                192,
                NUM_BINS,
            ),
            seed=43,
        )
    )

    x = tf.random.stateless_uniform(
        (2, 192),
        seed=(100, 101),
        minval=0.0,
        maxval=1.0,
        dtype=tf.float32,
    )

    with tf.GradientTape() as tape:
        y, log_det = forward(
            x,
            raw_widths,
            raw_slope_controls,
        )

        loss = tf.reduce_mean(
            y * y
            + 0.01
            * log_det
            * log_det
        )

    gradients = tape.gradient(
        loss,
        [
            raw_widths,
            raw_slope_controls,
        ],
    )

    assert bool(
        tf.reduce_all(
            tf.math.is_finite(y)
        ).numpy()
    )

    assert bool(
        tf.reduce_all(
            tf.math.is_finite(
                log_det
            )
        ).numpy()
    )

    assert all(
        gradient is not None
        for gradient in gradients
    )

    assert all(
        bool(
            tf.reduce_all(
                tf.math.is_finite(
                    gradient
                )
            ).numpy()
        )
        for gradient in gradients
    )
