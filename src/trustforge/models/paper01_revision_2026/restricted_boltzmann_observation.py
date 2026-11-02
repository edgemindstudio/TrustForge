"""Governed bounded continuous-visible observation model for Paper 1 RBM.

The revised Restricted Boltzmann Machine uses visible coordinates on [0,1]
with conditional density

    p(v | theta) = exp(theta * v) / Z(theta),

where

    Z(theta) = (exp(theta) - 1) / theta

with the continuous limit Z(0) = 1.

All authoritative internal arithmetic is float64.  Sampling is model-native:
no thresholding, endpoint epsilon, dequantization, or post-hoc clipping is
permitted.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf


DTYPE = tf.float64
NP_DTYPE = np.float64

VISIBLE_MIN = 0.0
VISIBLE_MAX = 1.0

_NEAR_ZERO = 1.0e-7


def _as_float64_tensor(
    value: np.ndarray | tf.Tensor | float,
    *,
    name: str,
) -> tf.Tensor:
    """Convert one numerical input to governed float64 and require finiteness."""

    tensor = tf.convert_to_tensor(
        value,
        dtype=DTYPE,
    )

    tf.debugging.assert_all_finite(
        tensor,
        f"{name} contains non-finite values",
    )

    return tensor


def log_normalizer(
    theta: np.ndarray | tf.Tensor | float,
) -> tf.Tensor:
    """Return log Z(theta) for the bounded-exponential visible conditional."""

    t = _as_float64_tensor(
        theta,
        name="theta",
    )

    abs_t = tf.abs(t)
    near_zero = abs_t < tf.constant(
        _NEAR_ZERO,
        dtype=DTYPE,
    )

    # Stable branch formulas:
    #
    # theta > 0:
    # log((exp(theta)-1)/theta)
    # = theta + log1p(-exp(-theta)) - log(theta)
    #
    # theta < 0:
    # log((1-exp(theta))/(-theta))
    # = log1p(-exp(theta)) - log(-theta)
    # Keep inactive tf.where branches numerically valid as well.  TensorFlow
    # may evaluate both branches during automatic differentiation, so raw
    # log(t), log(-t), or division by t at theta=0 would poison gradients.
    positive_t = tf.where(
        t > 0.0,
        t,
        tf.ones_like(t),
    )
    negative_t = tf.where(
        t < 0.0,
        t,
        -tf.ones_like(t),
    )

    positive = (
        positive_t
        + tf.math.log1p(
            -tf.exp(-positive_t)
        )
        - tf.math.log(positive_t)
    )

    negative = (
        tf.math.log1p(
            -tf.exp(negative_t)
        )
        - tf.math.log(-negative_t)
    )

    # log Z(theta) =
    # theta/2 + theta^2/24 - theta^4/2880 + O(theta^6)
    t2 = t * t
    series = (
        0.5 * t
        + t2 / 24.0
        - (t2 * t2) / 2880.0
    )

    result = tf.where(
        near_zero,
        series,
        tf.where(
            t > 0.0,
            positive,
            negative,
        ),
    )

    tf.debugging.assert_all_finite(
        result,
        "bounded-exponential log normalizer is non-finite",
    )

    return result


def log_prob(
    value: np.ndarray | tf.Tensor | float,
    theta: np.ndarray | tf.Tensor | float,
) -> tf.Tensor:
    """Return elementwise log density on [0,1]."""

    v = _as_float64_tensor(
        value,
        name="visible value",
    )
    t = _as_float64_tensor(
        theta,
        name="theta",
    )

    tf.debugging.assert_greater_equal(
        v,
        tf.constant(
            VISIBLE_MIN,
            dtype=DTYPE,
        ),
        message="visible value lies below zero",
    )
    tf.debugging.assert_less_equal(
        v,
        tf.constant(
            VISIBLE_MAX,
            dtype=DTYPE,
        ),
        message="visible value lies above one",
    )

    result = (
        t * v
        - log_normalizer(t)
    )

    tf.debugging.assert_all_finite(
        result,
        "bounded-exponential log probability is non-finite",
    )

    return result


def mean(
    theta: np.ndarray | tf.Tensor | float,
) -> tf.Tensor:
    """Return the analytic visible conditional mean E[v | theta]."""

    t = _as_float64_tensor(
        theta,
        name="theta",
    )

    abs_t = tf.abs(t)
    near_zero = abs_t < tf.constant(
        _NEAR_ZERO,
        dtype=DTYPE,
    )

    positive_t = tf.where(
        t > 0.0,
        t,
        tf.ones_like(t),
    )
    negative_t = tf.where(
        t < 0.0,
        t,
        -tf.ones_like(t),
    )

    positive = (
        1.0
        / (
            -tf.math.expm1(-positive_t)
        )
        - 1.0 / positive_t
    )

    negative = (
        tf.exp(negative_t)
        / tf.math.expm1(negative_t)
        - 1.0 / negative_t
    )

    # Mean series around zero:
    # 1/2 + theta/12 - theta^3/720 + theta^5/30240 + O(theta^7)
    t2 = t * t
    series = (
        0.5
        + t / 12.0
        - (t * t2) / 720.0
        + (t * t2 * t2) / 30240.0
    )

    result = tf.where(
        near_zero,
        series,
        tf.where(
            t > 0.0,
            positive,
            negative,
        ),
    )

    tf.debugging.assert_all_finite(
        result,
        "bounded-exponential analytic mean is non-finite",
    )
    tf.debugging.assert_greater_equal(
        result,
        tf.constant(
            VISIBLE_MIN,
            dtype=DTYPE,
        ),
        message="bounded-exponential analytic mean lies below zero",
    )
    tf.debugging.assert_less_equal(
        result,
        tf.constant(
            VISIBLE_MAX,
            dtype=DTYPE,
        ),
        message="bounded-exponential analytic mean lies above one",
    )

    return result


def inverse_cdf(
    uniform: np.ndarray | tf.Tensor | float,
    theta: np.ndarray | tf.Tensor | float,
) -> tf.Tensor:
    """Transform governed U[0,1] draws into bounded-exponential samples.

    Stable formulas:

    theta > 0:
        1 + log(u + (1-u) exp(-theta)) / theta

    theta < 0:
        log((1-u) + u exp(theta)) / theta

    theta ~= 0:
        the uniform limit, with a local series correction.
    """

    u = _as_float64_tensor(
        uniform,
        name="uniform draw",
    )
    t = _as_float64_tensor(
        theta,
        name="theta",
    )

    tf.debugging.assert_greater_equal(
        u,
        tf.constant(
            0.0,
            dtype=DTYPE,
        ),
        message="uniform draw lies below zero",
    )
    tf.debugging.assert_less_equal(
        u,
        tf.constant(
            1.0,
            dtype=DTYPE,
        ),
        message="uniform draw lies above one",
    )

    abs_t = tf.abs(t)
    near_zero = abs_t < tf.constant(
        _NEAR_ZERO,
        dtype=DTYPE,
    )

    positive_t = tf.where(
        t > 0.0,
        t,
        tf.ones_like(t),
    )
    negative_t = tf.where(
        t < 0.0,
        t,
        -tf.ones_like(t),
    )

    positive = (
        1.0
        + tf.math.log(
            u
            + (1.0 - u) * tf.exp(-positive_t)
        )
        / positive_t
    )

    negative = (
        tf.math.log(
            (1.0 - u)
            + u * tf.exp(negative_t)
        )
        / negative_t
    )

    # Local expansion of the inverse CDF around theta=0:
    #
    # v = u + theta*u*(1-u)/2
    #       + theta^2*u*(1-u)*(1-2u)/12 + O(theta^3)
    series = (
        u
        + t * u * (1.0 - u) / 2.0
        + (
            t * t
            * u
            * (1.0 - u)
            * (1.0 - 2.0 * u)
            / 12.0
        )
    )

    sample = tf.where(
        near_zero,
        series,
        tf.where(
            t > 0.0,
            positive,
            negative,
        ),
    )

    tf.debugging.assert_all_finite(
        sample,
        "bounded-exponential inverse-CDF sample is non-finite",
    )
    tf.debugging.assert_greater_equal(
        sample,
        tf.constant(
            VISIBLE_MIN,
            dtype=DTYPE,
        ),
        message="bounded-exponential sample lies below zero",
    )
    tf.debugging.assert_less_equal(
        sample,
        tf.constant(
            VISIBLE_MAX,
            dtype=DTYPE,
        ),
        message="bounded-exponential sample lies above one",
    )

    return sample


def sample_from_uniform(
    theta: np.ndarray | tf.Tensor,
    uniform: np.ndarray | tf.Tensor,
) -> tf.Tensor:
    """Sample using caller-supplied governed uniform randomness."""

    return inverse_cdf(
        uniform,
        theta,
    )


def validate_visible_values(
    values: np.ndarray | tf.Tensor,
) -> tf.Tensor:
    """Validate authoritative internal visible values and return float64 tensor."""

    visible = _as_float64_tensor(
        values,
        name="visible values",
    )

    tf.debugging.assert_greater_equal(
        visible,
        tf.constant(
            VISIBLE_MIN,
            dtype=DTYPE,
        ),
        message="visible values contain entries below zero",
    )
    tf.debugging.assert_less_equal(
        visible,
        tf.constant(
            VISIBLE_MAX,
            dtype=DTYPE,
        ),
        message="visible values contain entries above one",
    )

    return visible


__all__ = [
    "DTYPE",
    "NP_DTYPE",
    "VISIBLE_MIN",
    "VISIBLE_MAX",
    "inverse_cdf",
    "log_normalizer",
    "log_prob",
    "mean",
    "sample_from_uniform",
    "validate_visible_values",
]
