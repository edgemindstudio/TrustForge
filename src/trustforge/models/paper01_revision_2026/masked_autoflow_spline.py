"""Slope-conditioned bounded piecewise-linear spline for P1 revision-2026 MAF.

Each scalar uses:

- eight raw width parameters;
- eight raw relative-slope controls.

Widths are strictly positive and normalized to sum to one.

Relative-slope controls are bounded in log space and combined with the widths
to derive normalized heights. This preserves the eight-bin bounded monotonic
piecewise-linear transform while explicitly limiting numerical conditioning.

No endpoint epsilon, universal dequantization, or post-hoc clipping is used.
"""

from __future__ import annotations

import math
from typing import Tuple

import tensorflow as tf


NUM_BINS = 8

MIN_BIN_WIDTH = 1e-3

LOG_SLOPE_LIMIT = 0.75

MIN_SEGMENT_SLOPE = math.exp(
    -2.0 * LOG_SLOPE_LIMIT
)

MAX_SEGMENT_SLOPE = math.exp(
    2.0 * LOG_SLOPE_LIMIT
)

ROUNDTRIP_TOLERANCE = 2e-5
LOGDET_CANCELLATION_TOLERANCE = 1e-3


def _normalized_widths(
    raw_widths: tf.Tensor,
) -> tf.Tensor:
    """Convert unconstrained width logits to positive normalized widths."""
    raw_widths = tf.convert_to_tensor(
        raw_widths,
        dtype=tf.float32,
    )

    tf.debugging.assert_rank_at_least(
        raw_widths,
        2,
        message="width parameter tensor must have rank >= 2",
    )

    tf.debugging.assert_equal(
        tf.shape(raw_widths)[-1],
        NUM_BINS,
        message="last width dimension must equal NUM_BINS",
    )

    remaining = (
        tf.constant(
            1.0,
            dtype=tf.float32,
        )
        - tf.constant(
            MIN_BIN_WIDTH * NUM_BINS,
            dtype=tf.float32,
        )
    )

    tf.debugging.assert_positive(
        remaining,
        message="minimum bin width leaves no remaining interval",
    )

    widths = (
        tf.constant(
            MIN_BIN_WIDTH,
            dtype=tf.float32,
        )
        + remaining
        * tf.nn.softmax(
            raw_widths,
            axis=-1,
        )
    )

    tf.debugging.assert_all_finite(
        widths,
        "non-finite normalized spline widths",
    )

    tf.debugging.assert_positive(
        widths,
        message="spline widths must be positive",
    )

    return widths


def relative_slopes_from_controls(
    raw_slope_controls: tf.Tensor,
) -> tf.Tensor:
    """Map raw controls to finite positive bounded relative slopes."""
    raw_slope_controls = tf.convert_to_tensor(
        raw_slope_controls,
        dtype=tf.float32,
    )

    tf.debugging.assert_rank_at_least(
        raw_slope_controls,
        2,
        message="slope-control tensor must have rank >= 2",
    )

    tf.debugging.assert_equal(
        tf.shape(raw_slope_controls)[-1],
        NUM_BINS,
        message="last slope-control dimension must equal NUM_BINS",
    )

    bounded_log_relative_slope = (
        tf.constant(
            LOG_SLOPE_LIMIT,
            dtype=tf.float32,
        )
        * tf.math.tanh(
            raw_slope_controls
        )
    )

    relative_slopes = tf.exp(
        bounded_log_relative_slope
    )

    tf.debugging.assert_all_finite(
        relative_slopes,
        "non-finite relative spline slopes",
    )

    tf.debugging.assert_positive(
        relative_slopes,
        message="relative spline slopes must be positive",
    )

    return relative_slopes


def spline_parameters(
    raw_widths: tf.Tensor,
    raw_slope_controls: tf.Tensor,
) -> Tuple[
    tf.Tensor,
    tf.Tensor,
    tf.Tensor,
    tf.Tensor,
]:
    """Construct widths, derived heights, and exact bounded knots."""
    raw_widths = tf.convert_to_tensor(
        raw_widths,
        dtype=tf.float32,
    )

    raw_slope_controls = tf.convert_to_tensor(
        raw_slope_controls,
        dtype=tf.float32,
    )

    tf.debugging.assert_equal(
        tf.shape(raw_widths),
        tf.shape(raw_slope_controls),
        message="width and slope-control shapes must match",
    )

    widths = _normalized_widths(
        raw_widths
    )

    relative_slopes = (
        relative_slopes_from_controls(
            raw_slope_controls
        )
    )

    unnormalized_heights = (
        widths
        * relative_slopes
    )

    normalization = tf.reduce_sum(
        unnormalized_heights,
        axis=-1,
        keepdims=True,
    )

    tf.debugging.assert_positive(
        normalization,
        message="spline height normalization must be positive",
    )

    heights = (
        unnormalized_heights
        / normalization
    )

    tf.debugging.assert_all_finite(
        heights,
        "non-finite derived spline heights",
    )

    tf.debugging.assert_positive(
        heights,
        message="derived spline heights must be positive",
    )

    slopes = heights / widths

    tf.debugging.assert_all_finite(
        slopes,
        "non-finite spline segment slopes",
    )

    lower_bound = tf.constant(
        MIN_SEGMENT_SLOPE,
        dtype=tf.float32,
    )

    upper_bound = tf.constant(
        MAX_SEGMENT_SLOPE,
        dtype=tf.float32,
    )

    # Tiny float32 slack is allowed around the analytic envelope.
    slack = tf.constant(
        1e-6,
        dtype=tf.float32,
    )

    tf.debugging.assert_greater_equal(
        slopes,
        lower_bound - slack,
        message="spline slope below governed envelope",
    )

    tf.debugging.assert_less_equal(
        slopes,
        upper_bound + slack,
        message="spline slope above governed envelope",
    )

    zero = tf.zeros_like(
        widths[..., :1]
    )

    x_knots = tf.concat(
        [
            zero,
            tf.cumsum(
                widths,
                axis=-1,
            ),
        ],
        axis=-1,
    )

    y_knots = tf.concat(
        [
            zero,
            tf.cumsum(
                heights,
                axis=-1,
            ),
        ],
        axis=-1,
    )

    # Force exact canonical endpoints.
    x_knots = tf.concat(
        [
            x_knots[..., :-1],
            tf.ones_like(
                x_knots[..., -1:]
            ),
        ],
        axis=-1,
    )

    y_knots = tf.concat(
        [
            y_knots[..., :-1],
            tf.ones_like(
                y_knots[..., -1:]
            ),
        ],
        axis=-1,
    )

    return (
        widths,
        heights,
        x_knots,
        y_knots,
    )


def _select_bins(
    values: tf.Tensor,
    internal_knots: tf.Tensor,
) -> tf.Tensor:
    """Select bins using the governed right-side internal-knot convention."""
    indices = tf.searchsorted(
        internal_knots,
        values[..., None],
        side="right",
        out_type=tf.int32,
    )

    indices = tf.squeeze(
        indices,
        axis=-1,
    )

    return tf.minimum(
        indices,
        NUM_BINS - 1,
    )


def _gather_scalar_bin(
    values: tf.Tensor,
    indices: tf.Tensor,
) -> tf.Tensor:
    return tf.gather(
        values,
        indices,
        axis=-1,
        batch_dims=2,
    )


def _assert_no_negative_nonzero_float32(value):
    """Reject negative values even when CPU comparisons treat subnormals as zero."""
    bits = tf.bitcast(value, tf.int32)
    permitted = tf.logical_or(
        bits >= tf.constant(0, dtype=tf.int32),
        tf.equal(bits, tf.constant(-2147483648, dtype=tf.int32)),
    )
    tf.debugging.assert_equal(
        permitted, tf.ones_like(permitted),
        message="negative nonzero float32 input below canonical lower bound",
    )


def _interpolate_from_knots(value, source0, source1, target0, target1):
    """Interpolate using represented intervals and the nearer endpoint."""
    source_interval = source1 - source0
    target_interval = target1 - target0
    tf.debugging.assert_positive(
        source_interval, message="nonpositive represented source interval",
    )
    tf.debugging.assert_positive(
        target_interval, message="nonpositive represented target interval",
    )
    fraction = (value - source0) / source_interval
    tf.debugging.assert_greater_equal(
        fraction, tf.zeros_like(fraction),
        message="interpolation fraction below zero",
    )
    tf.debugging.assert_less_equal(
        fraction, tf.ones_like(fraction),
        message="interpolation fraction above one",
    )
    result = tf.where(
        fraction <= 0.5,
        target0 + fraction * target_interval,
        target1 - (1.0 - fraction) * target_interval,
    )
    return result


def forward(
    x: tf.Tensor,
    raw_widths: tf.Tensor,
    raw_slope_controls: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Apply the governed bounded piecewise-linear forward transform."""
    x = tf.convert_to_tensor(
        x,
        dtype=tf.float32,
    )

    tf.debugging.assert_rank(
        x,
        2,
        message="x must have shape (B,D)",
    )

    _assert_no_negative_nonzero_float32(x)

    tf.debugging.assert_greater_equal(
        x,
        tf.constant(
            0.0,
            dtype=tf.float32,
        ),
        message="spline input below canonical lower bound",
    )

    tf.debugging.assert_less_equal(
        x,
        tf.constant(
            1.0,
            dtype=tf.float32,
        ),
        message="spline input above canonical upper bound",
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

    tf.debugging.assert_equal(
        tf.shape(x),
        tf.shape(widths)[:-1],
        message="input and spline-parameter prefix shapes must match",
    )

    indices = _select_bins(
        x,
        x_knots[..., 1:-1],
    )

    x0 = _gather_scalar_bin(
        x_knots[..., :-1],
        indices,
    )

    y0 = _gather_scalar_bin(
        y_knots[..., :-1],
        indices,
    )

    width = _gather_scalar_bin(
        x_knots[..., 1:] - x_knots[..., :-1],
        indices,
    )

    height = _gather_scalar_bin(
        y_knots[..., 1:] - y_knots[..., :-1],
        indices,
    )

    slope = height / width

    x1 = _gather_scalar_bin(x_knots[..., 1:], indices)
    y1 = _gather_scalar_bin(y_knots[..., 1:], indices)
    y = _interpolate_from_knots(x, x0, x1, y0, y1)

    log_abs_det = tf.math.log(
        slope
    )

    y = tf.where(
        tf.equal(
            x,
            0.0,
        ),
        tf.zeros_like(y),
        y,
    )

    y = tf.where(
        tf.equal(
            x,
            1.0,
        ),
        tf.ones_like(y),
        y,
    )

    tf.debugging.assert_all_finite(
        y,
        "non-finite spline forward output",
    )

    tf.debugging.assert_greater_equal(
        y, tf.zeros_like(y),
        message="spline output below canonical lower bound",
    )
    tf.debugging.assert_less_equal(
        y, tf.ones_like(y),
        message="spline output above canonical upper bound",
    )

    tf.debugging.assert_all_finite(
        log_abs_det,
        "non-finite spline forward log determinant",
    )

    return (
        y,
        log_abs_det,
    )


def inverse(
    y: tf.Tensor,
    raw_widths: tf.Tensor,
    raw_slope_controls: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Invert the governed bounded piecewise-linear transform."""
    y = tf.convert_to_tensor(
        y,
        dtype=tf.float32,
    )

    tf.debugging.assert_rank(
        y,
        2,
        message="y must have shape (B,D)",
    )

    _assert_no_negative_nonzero_float32(y)

    tf.debugging.assert_greater_equal(
        y,
        tf.constant(
            0.0,
            dtype=tf.float32,
        ),
        message="inverse input below canonical lower bound",
    )

    tf.debugging.assert_less_equal(
        y,
        tf.constant(
            1.0,
            dtype=tf.float32,
        ),
        message="inverse input above canonical upper bound",
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

    tf.debugging.assert_equal(
        tf.shape(y),
        tf.shape(widths)[:-1],
        message="input and spline-parameter prefix shapes must match",
    )

    indices = _select_bins(
        y,
        y_knots[..., 1:-1],
    )

    x0 = _gather_scalar_bin(
        x_knots[..., :-1],
        indices,
    )

    y0 = _gather_scalar_bin(
        y_knots[..., :-1],
        indices,
    )

    width = _gather_scalar_bin(
        x_knots[..., 1:] - x_knots[..., :-1],
        indices,
    )

    height = _gather_scalar_bin(
        y_knots[..., 1:] - y_knots[..., :-1],
        indices,
    )

    slope = height / width

    x1 = _gather_scalar_bin(x_knots[..., 1:], indices)
    y1 = _gather_scalar_bin(y_knots[..., 1:], indices)
    x = _interpolate_from_knots(y, y0, y1, x0, x1)

    inverse_log_abs_det = (
        -tf.math.log(
            slope
        )
    )

    x = tf.where(
        tf.equal(
            y,
            0.0,
        ),
        tf.zeros_like(x),
        x,
    )

    x = tf.where(
        tf.equal(
            y,
            1.0,
        ),
        tf.ones_like(x),
        x,
    )

    tf.debugging.assert_all_finite(
        x,
        "non-finite spline inverse output",
    )

    tf.debugging.assert_greater_equal(
        x, tf.zeros_like(x),
        message="spline output below canonical lower bound",
    )
    tf.debugging.assert_less_equal(
        x, tf.ones_like(x),
        message="spline output above canonical upper bound",
    )

    tf.debugging.assert_all_finite(
        inverse_log_abs_det,
        "non-finite spline inverse log determinant",
    )

    return (
        x,
        inverse_log_abs_det,
    )


__all__ = [
    "LOGDET_CANCELLATION_TOLERANCE",
    "LOG_SLOPE_LIMIT",
    "MAX_SEGMENT_SLOPE",
    "MIN_BIN_WIDTH",
    "MIN_SEGMENT_SLOPE",
    "NUM_BINS",
    "ROUNDTRIP_TOLERANCE",
    "forward",
    "inverse",
    "relative_slopes_from_controls",
    "spline_parameters",
]
