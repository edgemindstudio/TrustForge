"""Governed censored-logistic observation model for Paper 1 revision 2026.

This module implements the numerical observation distribution only.

It does not alter the PixelCNN spatial-causality backbone. The active
autoregressive training and raster-sampling paths are wired to these primitives
only after their regression contract has been validated.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf


RGB_CHANNELS = 3
OBSERVATION_PARAM_CHANNELS = 9


def split_censored_logistic_parameters(
    params: tf.Tensor,
    *,
    min_log_scale: float = -7.0,
    max_log_scale: float = 3.0,
) -> tuple[
    tf.Tensor,
    tf.Tensor,
    tf.Tensor,
]:
    """Split nine raw output channels into locations, scales, and couplings.

    Governed parameter order:

    0:3
        Base RGB locations.

    3:6
        Raw RGB log scales, clipped to the governed numerical limits before
        exponentiation.

    6:9
        Raw RGB coupling coefficients in the order:
        alpha_rg, alpha_br, alpha_bg.

    Returns:
        base_mu:
            Tensor with final dimension 3.

        scale:
            Strictly positive tensor with final dimension 3.

        alpha:
            Tensor with final dimension 3 and values strictly inside (-1,1)
            for finite raw inputs.
    """
    params = tf.convert_to_tensor(
        params,
        dtype=tf.float32,
    )

    if (
        params.shape.rank is not None
        and params.shape[-1] != OBSERVATION_PARAM_CHANNELS
    ):
        raise ValueError(
            "expected final parameter dimension "
            f"{OBSERVATION_PARAM_CHANNELS}; got {params.shape}"
        )

    tf.debugging.assert_equal(
        tf.shape(params)[-1],
        OBSERVATION_PARAM_CHANNELS,
        message=(
            "censored-logistic parameter tensor must have "
            "nine channels"
        ),
    )

    if float(min_log_scale) >= float(max_log_scale):
        raise ValueError(
            "min_log_scale must be smaller than max_log_scale"
        )

    base_mu = params[..., 0:3]

    log_scale = tf.clip_by_value(
        params[..., 3:6],
        clip_value_min=float(
            min_log_scale
        ),
        clip_value_max=float(
            max_log_scale
        ),
    )

    scale = tf.exp(
        log_scale
    )

    alpha = tf.math.tanh(
        params[..., 6:9]
    )

    tf.debugging.assert_all_finite(
        base_mu,
        "non-finite censored-logistic base locations",
    )

    tf.debugging.assert_all_finite(
        scale,
        "non-finite censored-logistic scales",
    )

    tf.debugging.assert_positive(
        scale,
        message="censored-logistic scales must be positive",
    )

    tf.debugging.assert_all_finite(
        alpha,
        "non-finite censored-logistic RGB couplings",
    )

    return (
        base_mu,
        scale,
        alpha,
    )


def coupled_rgb_locations(
    base_mu: tf.Tensor,
    alpha: tf.Tensor,
    rgb_context: tf.Tensor,
) -> tf.Tensor:
    """Construct effective R -> G -> B conditional locations.

    During likelihood evaluation, ``rgb_context`` is the observed target RGB
    vector for the current spatial pixel.

    During sampling, the caller supplies the already-generated red and green
    channel values in the same tensor positions.
    """
    base_mu = tf.convert_to_tensor(
        base_mu,
        dtype=tf.float32,
    )

    alpha = tf.convert_to_tensor(
        alpha,
        dtype=tf.float32,
    )

    rgb_context = tf.convert_to_tensor(
        rgb_context,
        dtype=tf.float32,
    )

    for name, tensor in (
        (
            "base_mu",
            base_mu,
        ),
        (
            "alpha",
            alpha,
        ),
        (
            "rgb_context",
            rgb_context,
        ),
    ):
        if (
            tensor.shape.rank is not None
            and tensor.shape[-1] != RGB_CHANNELS
        ):
            raise ValueError(
                f"{name} must have final dimension 3; "
                f"got {tensor.shape}"
            )

    red = rgb_context[..., 0]
    green = rgb_context[..., 1]

    alpha_rg = alpha[..., 0]
    alpha_br = alpha[..., 1]
    alpha_bg = alpha[..., 2]

    mu_r = base_mu[..., 0]

    mu_g = (
        base_mu[..., 1]
        + alpha_rg
        * (
            red
            - 0.5
        )
    )

    mu_b = (
        base_mu[..., 2]
        + alpha_br
        * (
            red
            - 0.5
        )
        + alpha_bg
        * (
            green
            - 0.5
        )
    )

    locations = tf.stack(
        (
            mu_r,
            mu_g,
            mu_b,
        ),
        axis=-1,
    )

    tf.debugging.assert_all_finite(
        locations,
        "non-finite coupled RGB locations",
    )

    return locations


def censored_logistic_log_prob(
    x: tf.Tensor,
    mu: tf.Tensor,
    scale: tf.Tensor,
) -> tf.Tensor:
    """Evaluate governed scalar censored-logistic log probability.

    Exact zero and exact one are boundary atoms.

    Values strictly inside (0,1) use the ordinary logistic density.
    """
    x = tf.convert_to_tensor(
        x,
        dtype=tf.float32,
    )

    mu = tf.convert_to_tensor(
        mu,
        dtype=tf.float32,
    )

    scale = tf.convert_to_tensor(
        scale,
        dtype=tf.float32,
    )

    tf.debugging.assert_all_finite(
        x,
        "non-finite censored-logistic observations",
    )

    tf.debugging.assert_all_finite(
        mu,
        "non-finite censored-logistic locations",
    )

    tf.debugging.assert_all_finite(
        scale,
        "non-finite censored-logistic scales",
    )

    tf.debugging.assert_positive(
        scale,
        message="censored-logistic scale must be positive",
    )

    tf.debugging.assert_greater_equal(
        x,
        tf.constant(
            0.0,
            dtype=tf.float32,
        ),
        message="censored-logistic observation below zero",
    )

    tf.debugging.assert_less_equal(
        x,
        tf.constant(
            1.0,
            dtype=tf.float32,
        ),
        message="censored-logistic observation above one",
    )

    z = (
        x
        - mu
    ) / scale

    log_scale = tf.math.log(
        scale
    )

    interior = (
        -tf.nn.softplus(
            z
        )
        -tf.nn.softplus(
            -z
        )
        -log_scale
    )

    lower_z = (
        -mu
    ) / scale

    lower_atom = tf.math.log_sigmoid(
        lower_z
    )

    upper_z = (
        mu
        - 1.0
    ) / scale

    upper_atom = tf.math.log_sigmoid(
        upper_z
    )

    result = tf.where(
        tf.equal(
            x,
            0.0,
        ),
        lower_atom,
        tf.where(
            tf.equal(
                x,
                1.0,
            ),
            upper_atom,
            interior,
        ),
    )

    tf.debugging.assert_all_finite(
        result,
        "non-finite censored-logistic log probability",
    )

    return result


def coupled_rgb_censored_logistic_nll(
    target_rgb: tf.Tensor,
    params: tf.Tensor,
    *,
    min_log_scale: float = -7.0,
    max_log_scale: float = 3.0,
) -> tf.Tensor:
    """Return mean negative log likelihood per RGB spatial pixel."""
    target_rgb = tf.convert_to_tensor(
        target_rgb,
        dtype=tf.float32,
    )

    if (
        target_rgb.shape.rank is not None
        and target_rgb.shape[-1] != RGB_CHANNELS
    ):
        raise ValueError(
            "target_rgb must have final dimension 3; "
            f"got {target_rgb.shape}"
        )

    tf.debugging.assert_equal(
        tf.shape(target_rgb)[-1],
        RGB_CHANNELS,
        message="target_rgb must have exactly three channels",
    )

    base_mu, scale, alpha = (
        split_censored_logistic_parameters(
            params,
            min_log_scale=min_log_scale,
            max_log_scale=max_log_scale,
        )
    )

    locations = coupled_rgb_locations(
        base_mu,
        alpha,
        target_rgb,
    )

    channel_log_prob = censored_logistic_log_prob(
        target_rgb,
        locations,
        scale,
    )

    pixel_log_prob = tf.reduce_sum(
        channel_log_prob,
        axis=-1,
    )

    nll = -tf.reduce_mean(
        pixel_log_prob
    )

    tf.debugging.assert_all_finite(
        nll,
        "non-finite coupled RGB censored-logistic NLL",
    )

    return nll


def _machine_open_uniform(
    shape: tf.Tensor,
    *,
    seed: tf.Tensor,
) -> tf.Tensor:
    """Draw reproducible float32 uniforms protected from exact 0 and 1."""
    values = tf.random.stateless_uniform(
        shape=shape,
        seed=seed,
        minval=0.0,
        maxval=1.0,
        dtype=tf.float32,
    )

    lower = np.nextafter(
        np.float32(
            0.0
        ),
        np.float32(
            1.0
        ),
        dtype=np.float32,
    )

    upper = np.nextafter(
        np.float32(
            1.0
        ),
        np.float32(
            0.0
        ),
        dtype=np.float32,
    )

    return tf.clip_by_value(
        values,
        clip_value_min=float(
            lower
        ),
        clip_value_max=float(
            upper
        ),
    )


def _sample_scalar_censored_logistic(
    mu: tf.Tensor,
    scale: tf.Tensor,
    *,
    temperature: float,
    seed: tf.Tensor,
) -> tf.Tensor:
    """Sample one governed scalar-channel tensor."""
    temperature = float(
        temperature
    )

    if temperature < 0.0:
        raise ValueError(
            "temperature must be non-negative"
        )

    mu = tf.convert_to_tensor(
        mu,
        dtype=tf.float32,
    )

    scale = tf.convert_to_tensor(
        scale,
        dtype=tf.float32,
    )

    if temperature == 0.0:
        return tf.clip_by_value(
            mu,
            0.0,
            1.0,
        )

    uniform = _machine_open_uniform(
        tf.shape(
            mu
        ),
        seed=seed,
    )

    logistic_noise = (
        tf.math.log(
            uniform
        )
        - tf.math.log1p(
            -uniform
        )
    )

    latent = (
        mu
        + temperature
        * scale
        * logistic_noise
    )

    sample = tf.clip_by_value(
        latent,
        0.0,
        1.0,
    )

    tf.debugging.assert_all_finite(
        sample,
        "non-finite censored-logistic sample",
    )

    return tf.cast(
        sample,
        tf.float32,
    )


def sample_coupled_rgb_from_parameters(
    params: tf.Tensor,
    *,
    temperature: float = 1.0,
    seed: int = 42,
    min_log_scale: float = -7.0,
    max_log_scale: float = 3.0,
) -> tf.Tensor:
    """Sample coupled RGB values directly in canonical [0,1] space."""
    params = tf.convert_to_tensor(
        params,
        dtype=tf.float32,
    )

    base_mu, scale, alpha = (
        split_censored_logistic_parameters(
            params,
            min_log_scale=min_log_scale,
            max_log_scale=max_log_scale,
        )
    )

    alpha_rg = alpha[..., 0]
    alpha_br = alpha[..., 1]
    alpha_bg = alpha[..., 2]

    red = _sample_scalar_censored_logistic(
        base_mu[..., 0],
        scale[..., 0],
        temperature=temperature,
        seed=tf.constant(
            [
                int(
                    seed
                ),
                0,
            ],
            dtype=tf.int32,
        ),
    )

    mu_green = (
        base_mu[..., 1]
        + alpha_rg
        * (
            red
            - 0.5
        )
    )

    green = _sample_scalar_censored_logistic(
        mu_green,
        scale[..., 1],
        temperature=temperature,
        seed=tf.constant(
            [
                int(
                    seed
                ),
                1,
            ],
            dtype=tf.int32,
        ),
    )

    mu_blue = (
        base_mu[..., 2]
        + alpha_br
        * (
            red
            - 0.5
        )
        + alpha_bg
        * (
            green
            - 0.5
        )
    )

    blue = _sample_scalar_censored_logistic(
        mu_blue,
        scale[..., 2],
        temperature=temperature,
        seed=tf.constant(
            [
                int(
                    seed
                ),
                2,
            ],
            dtype=tf.int32,
        ),
    )

    rgb = tf.stack(
        (
            red,
            green,
            blue,
        ),
        axis=-1,
    )

    tf.debugging.assert_all_finite(
        rgb,
        "non-finite coupled censored-logistic RGB sample",
    )

    tf.debugging.assert_greater_equal(
        rgb,
        tf.constant(
            0.0,
            dtype=tf.float32,
        ),
    )

    tf.debugging.assert_less_equal(
        rgb,
        tf.constant(
            1.0,
            dtype=tf.float32,
        ),
    )

    return tf.cast(
        rgb,
        tf.float32,
    )


__all__ = [
    "OBSERVATION_PARAM_CHANNELS",
    "RGB_CHANNELS",
    "censored_logistic_log_prob",
    "coupled_rgb_censored_logistic_nll",
    "coupled_rgb_locations",
    "sample_coupled_rgb_from_parameters",
    "split_censored_logistic_parameters",
]
