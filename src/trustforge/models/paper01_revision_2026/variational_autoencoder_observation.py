"""Governed conditional-VAE observation mathematics.

Paper 1 revision 2026.

Numerical baseline:
- canonical images: float32 NHWC, 40x40x3, [0,1]
- internal images: float32 NHWC, 40x40x3, [-1,1]
- observation model: fixed-variance Gaussian, variance 1.0
- KL coefficient: 1.0
- reconstruction: full per-example Gaussian NLL
- objective: mean over examples

No dataset access, checkpoint access, training orchestration,
or production synthesis is performed here.
"""

from __future__ import annotations

import math

import tensorflow as tf


CANONICAL_SHAPE = (40, 40, 3)
CANONICAL_DIM = 4800
DTYPE = tf.float32

OBSERVATION_VARIANCE = 1.0
KL_WEIGHT = 1.0

_LOG_2PI = math.log(2.0 * math.pi)


def _image_tensor(
    value: tf.Tensor,
    *,
    name: str,
    minimum: float,
    maximum: float,
) -> tf.Tensor:
    """Validate a governed image batch without modifying values."""
    value = tf.convert_to_tensor(value)

    if value.dtype != DTYPE:
        raise TypeError(f"{name} must be float32; got {value.dtype}")

    tf.debugging.assert_rank(
        value, 4, message=f"{name} must have NHWC rank four"
    )
    tf.debugging.assert_equal(
        tf.shape(value)[1:],
        tf.constant(CANONICAL_SHAPE, dtype=tf.int32),
        message=f"{name} must have shape (B,40,40,3)",
    )
    tf.debugging.assert_positive(
        tf.shape(value)[0],
        message=f"{name} batch must be nonempty",
    )
    tf.debugging.assert_all_finite(
        value, f"{name} contains non-finite values"
    )
    tf.debugging.assert_greater_equal(
        value,
        tf.cast(minimum, DTYPE),
        message=f"{name} contains values below {minimum}",
    )
    tf.debugging.assert_less_equal(
        value,
        tf.cast(maximum, DTYPE),
        message=f"{name} contains values above {maximum}",
    )
    return value


def canonical_to_internal(
    images: tf.Tensor,
) -> tf.Tensor:
    """Map canonical [0,1] images to VAE internal [-1,1]."""
    images = _image_tensor(
        images,
        name="canonical images",
        minimum=0.0,
        maximum=1.0,
    )
    return 2.0 * images - 1.0


def internal_to_canonical(
    images: tf.Tensor,
) -> tf.Tensor:
    """Map bounded internal decoder means to [0,1].

    This is a deterministic coordinate conversion, not a
    Gaussian observation draw or a posthoc clipping rule.
    """
    images = _image_tensor(
        images,
        name="internal decoder means",
        minimum=-1.0,
        maximum=1.0,
    )
    return (images + 1.0) / 2.0


def gaussian_nll_per_example(
    observations: tf.Tensor,
    decoder_means: tf.Tensor,
) -> tf.Tensor:
    """Full fixed-variance Gaussian NLL, one scalar per example.

    NLL = 0.5 * sum((x - mu)^2)
        + 0.5 * D * log(2*pi)

    The variance is governed as exactly 1.0.
    """
    observations = _image_tensor(
        observations,
        name="internal observations",
        minimum=-1.0,
        maximum=1.0,
    )
    decoder_means = _image_tensor(
        decoder_means,
        name="decoder means",
        minimum=-1.0,
        maximum=1.0,
    )
    tf.debugging.assert_equal(
        tf.shape(observations),
        tf.shape(decoder_means),
        message="observation/decoder shapes differ",
    )

    residual = observations - decoder_means
    squared = tf.reduce_sum(
        tf.square(residual), axis=(1, 2, 3)
    )
    constant = 0.5 * CANONICAL_DIM * _LOG_2PI
    nll = 0.5 * squared + tf.cast(constant, DTYPE)

    tf.debugging.assert_all_finite(
        nll, "Gaussian NLL is non-finite"
    )
    return nll


def diagonal_gaussian_kl_per_example(
    posterior_mean: tf.Tensor,
    posterior_log_variance: tf.Tensor,
) -> tf.Tensor:
    """Exact KL from diagonal Gaussian posterior to N(0,I)."""
    posterior_mean = tf.convert_to_tensor(posterior_mean)
    posterior_log_variance = tf.convert_to_tensor(
        posterior_log_variance
    )

    if (
        posterior_mean.dtype != DTYPE
        or posterior_log_variance.dtype != DTYPE
    ):
        raise TypeError("posterior parameters must be float32")

    tf.debugging.assert_rank(
        posterior_mean, 2,
        message="posterior mean must have shape (B,L)",
    )
    tf.debugging.assert_rank(
        posterior_log_variance, 2,
        message="posterior log variance must have shape (B,L)",
    )
    tf.debugging.assert_equal(
        tf.shape(posterior_mean),
        tf.shape(posterior_log_variance),
        message="posterior parameter shapes differ",
    )
    tf.debugging.assert_positive(
        tf.shape(posterior_mean)[0],
        message="posterior batch must be nonempty",
    )
    tf.debugging.assert_positive(
        tf.shape(posterior_mean)[1],
        message="latent dimension must be positive",
    )
    tf.debugging.assert_all_finite(
        posterior_mean, "posterior mean is non-finite"
    )
    tf.debugging.assert_all_finite(
        posterior_log_variance,
        "posterior log variance is non-finite",
    )

    kl = -0.5 * tf.reduce_sum(
        1.0
        + posterior_log_variance
        - tf.square(posterior_mean)
        - tf.exp(posterior_log_variance),
        axis=1,
    )
    tf.debugging.assert_all_finite(
        kl, "diagonal Gaussian KL is non-finite"
    )
    return kl


def vae_objective(
    observations: tf.Tensor,
    decoder_means: tf.Tensor,
    posterior_mean: tf.Tensor,
    posterior_log_variance: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Return (mean_total, mean_reconstruction_NLL, mean_KL).

    The reconstruction term is a one-draw estimate of the
    expected observation NLL when decoder_means are obtained
    from a sampled posterior latent vector.
    """
    nll = gaussian_nll_per_example(
        observations, decoder_means
    )
    kl = diagonal_gaussian_kl_per_example(
        posterior_mean, posterior_log_variance
    )

    tf.debugging.assert_equal(
        tf.shape(nll)[0],
        tf.shape(kl)[0],
        message="reconstruction/KL batch sizes differ",
    )

    mean_nll = tf.reduce_mean(nll)
    mean_kl = tf.reduce_mean(kl)
    mean_total = mean_nll + KL_WEIGHT * mean_kl

    tf.debugging.assert_all_finite(
        mean_total, "VAE objective is non-finite"
    )
    return mean_total, mean_nll, mean_kl


__all__ = [
    "CANONICAL_SHAPE",
    "CANONICAL_DIM",
    "OBSERVATION_VARIANCE",
    "KL_WEIGHT",
    "canonical_to_internal",
    "internal_to_canonical",
    "gaussian_nll_per_example",
    "diagonal_gaussian_kl_per_example",
    "vae_objective",
]
