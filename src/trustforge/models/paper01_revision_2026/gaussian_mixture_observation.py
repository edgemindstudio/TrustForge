"""Numerical observation primitives for the governed Paper 1 revision-2026 GMM.

The revised Gaussian-mixture generator is a class-conditional censored
diagonal Gaussian mixture operating directly on the canonical 4800-scalar
representation.

This module contains only probability and scale primitives.

Historical Gaussian-mixture implementations remain unchanged.
"""

from __future__ import annotations

import math

import tensorflow as tf


DTYPE = tf.float64

NUM_COMPONENTS = 5
CANONICAL_DIM = 4800

MIN_VARIANCE = 1e-3
MIN_SCALE = math.sqrt(MIN_VARIANCE)

TAIL_CUTOFF = -10.0

_SQRT_2 = tf.constant(
    math.sqrt(2.0),
    dtype=DTYPE,
)

_LOG_SQRT_2PI = tf.constant(
    0.5 * math.log(2.0 * math.pi),
    dtype=DTYPE,
)

_TAIL_CUTOFF = tf.constant(
    TAIL_CUTOFF,
    dtype=DTYPE,
)

_MIN_SCALE = tf.constant(
    MIN_SCALE,
    dtype=DTYPE,
)


def stable_log_ndtr(
    z: tf.Tensor,
) -> tf.Tensor:
    """Return log(Phi(z)) with a stable negative-tail approximation.

    TensorFlow 2.20 in the governed execution environment does not expose
    ``tf.math.log_ndtr``.  The central region therefore uses the equivalent
    erfc expression, while values below the governed cutoff use a Mills-ratio
    asymptotic expansion.

    Computation is performed in float64.
    """

    z = tf.convert_to_tensor(
        z,
        dtype=DTYPE,
    )

    tf.debugging.assert_all_finite(
        z,
        "stable_log_ndtr received non-finite input",
    )

    use_tail = z < _TAIL_CUTOFF

    # Both branches are clamped into their numerically safe domains because
    # tf.where may evaluate both branches.
    central_z = tf.maximum(
        z,
        _TAIL_CUTOFF,
    )

    central = tf.math.log(
        tf.constant(
            0.5,
            dtype=DTYPE,
        )
        * tf.math.erfc(
            -central_z / _SQRT_2
        )
    )

    tail_z = tf.minimum(
        z,
        _TAIL_CUTOFF,
    )

    x = -tail_z

    inv_x2 = tf.math.reciprocal(
        tf.square(x)
    )

    correction = (
        tf.constant(
            1.0,
            dtype=DTYPE,
        )
        - inv_x2
        + tf.constant(
            3.0,
            dtype=DTYPE,
        )
        * tf.square(inv_x2)
        - tf.constant(
            15.0,
            dtype=DTYPE,
        )
        * tf.pow(
            inv_x2,
            3,
        )
        + tf.constant(
            105.0,
            dtype=DTYPE,
        )
        * tf.pow(
            inv_x2,
            4,
        )
        - tf.constant(
            945.0,
            dtype=DTYPE,
        )
        * tf.pow(
            inv_x2,
            5,
        )
    )

    tf.debugging.assert_positive(
        correction,
        message=(
            "negative-tail Mills-ratio correction "
            "must remain positive"
        ),
    )

    tail = (
        -tf.constant(
            0.5,
            dtype=DTYPE,
        )
        * tf.square(tail_z)
        - tf.math.log(x)
        - _LOG_SQRT_2PI
        + tf.math.log(correction)
    )

    result = tf.where(
        use_tail,
        tail,
        central,
    )

    tf.debugging.assert_all_finite(
        result,
        "stable_log_ndtr produced non-finite output",
    )

    return result


def transformed_scale(
    raw_scale: tf.Tensor,
) -> tf.Tensor:
    """Map unconstrained raw scales to the governed positive scale domain."""

    raw_scale = tf.convert_to_tensor(
        raw_scale,
        dtype=DTYPE,
    )

    tf.debugging.assert_all_finite(
        raw_scale,
        "raw GMM scales contain non-finite values",
    )

    scale = (
        _MIN_SCALE
        + tf.nn.softplus(raw_scale)
    )

    tf.debugging.assert_all_finite(
        scale,
        "transformed GMM scales contain non-finite values",
    )

    tf.debugging.assert_greater(
        scale,
        _MIN_SCALE,
        message=(
            "transformed GMM scale must remain "
            "strictly above the governed floor"
        ),
    )

    return scale


def inverse_softplus_positive(
    value: tf.Tensor,
) -> tf.Tensor:
    """Invert softplus for a strictly positive float64 target."""

    value = tf.convert_to_tensor(
        value,
        dtype=DTYPE,
    )

    tf.debugging.assert_all_finite(
        value,
        "inverse-softplus target contains non-finite values",
    )

    tf.debugging.assert_positive(
        value,
        message=(
            "inverse-softplus target must be strictly positive"
        ),
    )

    # The large-value branch avoids overflow in expm1.
    result = tf.where(
        value > tf.constant(
            20.0,
            dtype=DTYPE,
        ),
        value,
        tf.math.log(
            tf.math.expm1(value)
        ),
    )

    tf.debugging.assert_all_finite(
        result,
        "inverse-softplus produced non-finite output",
    )

    return result


def censored_normal_log_prob(
    x: tf.Tensor,
    mean: tf.Tensor,
    scale: tf.Tensor,
) -> tf.Tensor:
    """Return scalar censored-Gaussian log probability.

    Observation semantics:

    - ``x == 0`` uses the Gaussian mass at or below zero;
    - ``0 < x < 1`` uses the Gaussian density;
    - ``x == 1`` uses the Gaussian mass at or above one.

    Inputs broadcast according to normal TensorFlow rules.
    """

    x = tf.convert_to_tensor(
        x,
        dtype=DTYPE,
    )

    mean = tf.convert_to_tensor(
        mean,
        dtype=DTYPE,
    )

    scale = tf.convert_to_tensor(
        scale,
        dtype=DTYPE,
    )

    tf.debugging.assert_all_finite(
        x,
        "censored-Gaussian observations contain non-finite values",
    )

    tf.debugging.assert_all_finite(
        mean,
        "censored-Gaussian means contain non-finite values",
    )

    tf.debugging.assert_all_finite(
        scale,
        "censored-Gaussian scales contain non-finite values",
    )

    tf.debugging.assert_greater_equal(
        x,
        tf.constant(
            0.0,
            dtype=DTYPE,
        ),
        message="GMM observations contain values below zero",
    )

    tf.debugging.assert_less_equal(
        x,
        tf.constant(
            1.0,
            dtype=DTYPE,
        ),
        message="GMM observations contain values above one",
    )

    tf.debugging.assert_positive(
        scale,
        message="GMM scales must be positive",
    )

    standardized = (
        x - mean
    ) / scale

    interior = (
        -tf.constant(
            0.5,
            dtype=DTYPE,
        )
        * tf.square(standardized)
        - tf.math.log(scale)
        - _LOG_SQRT_2PI
    )

    lower_mass = stable_log_ndtr(
        -mean / scale
    )

    upper_mass = stable_log_ndtr(
        (
            mean
            - tf.constant(
                1.0,
                dtype=DTYPE,
            )
        )
        / scale
    )

    result = tf.where(
        x <= tf.constant(
            0.0,
            dtype=DTYPE,
        ),
        lower_mass,
        tf.where(
            x >= tf.constant(
                1.0,
                dtype=DTYPE,
            ),
            upper_mass,
            interior,
        ),
    )

    tf.debugging.assert_all_finite(
        result,
        "censored-Gaussian log probability is non-finite",
    )

    return result


def mixture_log_prob(
    x: tf.Tensor,
    logits: tf.Tensor,
    means: tf.Tensor,
    raw_scales: tf.Tensor,
) -> tf.Tensor:
    """Return one censored diagonal-mixture log probability per example."""

    x = tf.convert_to_tensor(
        x,
        dtype=DTYPE,
    )

    logits = tf.convert_to_tensor(
        logits,
        dtype=DTYPE,
    )

    means = tf.convert_to_tensor(
        means,
        dtype=DTYPE,
    )

    raw_scales = tf.convert_to_tensor(
        raw_scales,
        dtype=DTYPE,
    )

    tf.debugging.assert_rank(
        x,
        2,
        message="GMM input must have shape (B,D)",
    )

    tf.debugging.assert_rank(
        logits,
        1,
        message="GMM logits must have shape (K,)",
    )

    tf.debugging.assert_rank(
        means,
        2,
        message="GMM means must have shape (K,D)",
    )

    tf.debugging.assert_rank(
        raw_scales,
        2,
        message="GMM raw scales must have shape (K,D)",
    )

    tf.debugging.assert_equal(
        tf.shape(logits)[0],
        NUM_COMPONENTS,
        message="GMM requires exactly five mixture components",
    )

    tf.debugging.assert_equal(
        tf.shape(means)[0],
        NUM_COMPONENTS,
        message="GMM means require exactly five components",
    )

    tf.debugging.assert_equal(
        tf.shape(raw_scales)[0],
        NUM_COMPONENTS,
        message="GMM raw scales require exactly five components",
    )

    tf.debugging.assert_equal(
        tf.shape(x)[1],
        tf.shape(means)[1],
        message="GMM input and mean dimensions differ",
    )

    tf.debugging.assert_equal(
        tf.shape(means),
        tf.shape(raw_scales),
        message="GMM means and raw scales must have identical shape",
    )

    tf.debugging.assert_all_finite(
        logits,
        "GMM logits contain non-finite values",
    )

    tf.debugging.assert_all_finite(
        means,
        "GMM means contain non-finite values",
    )

    scales = transformed_scale(
        raw_scales
    )

    scalar_log_prob = (
        censored_normal_log_prob(
            x[:, None, :],
            means[None, :, :],
            scales[None, :, :],
        )
    )

    component_log_prob = tf.reduce_sum(
        scalar_log_prob,
        axis=-1,
    )

    log_weights = tf.nn.log_softmax(
        logits,
    )

    result = tf.reduce_logsumexp(
        component_log_prob
        + log_weights[None, :],
        axis=1,
    )

    tf.debugging.assert_all_finite(
        result,
        "GMM mixture log probability is non-finite",
    )

    return result


def mean_censored_mixture_nll(
    x: tf.Tensor,
    logits: tf.Tensor,
    means: tf.Tensor,
    raw_scales: tf.Tensor,
) -> tf.Tensor:
    """Return mean censored diagonal-mixture negative log likelihood."""

    log_prob = mixture_log_prob(
        x,
        logits,
        means,
        raw_scales,
    )

    loss = -tf.reduce_mean(
        log_prob
    )

    tf.debugging.assert_all_finite(
        loss,
        "GMM mean censored NLL is non-finite",
    )

    return loss


__all__ = [
    "CANONICAL_DIM",
    "DTYPE",
    "MIN_SCALE",
    "MIN_VARIANCE",
    "NUM_COMPONENTS",
    "TAIL_CUTOFF",
    "censored_normal_log_prob",
    "inverse_softplus_positive",
    "mean_censored_mixture_nll",
    "mixture_log_prob",
    "stable_log_ndtr",
    "transformed_scale",
]
