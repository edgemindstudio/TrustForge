"""Sampling primitives for the governed revision-2026 Conditional PixelCNN.

Sampling is sequential over spatial pixels and vectorized over the sample
batch. Historical autoregressive sampling code remains unchanged.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from .autoregressive_observation import (
    OBSERVATION_PARAM_CHANNELS,
    sample_coupled_rgb_from_parameters,
)


def _validate_class_ids(
    class_ids: np.ndarray,
    *,
    num_classes: int,
) -> np.ndarray:
    """Validate and normalize requested class identifiers."""
    ids = np.asarray(class_ids)

    if ids.ndim != 1:
        raise ValueError(
            f"class_ids must have shape (B,); got {ids.shape}"
        )

    if ids.size == 0:
        raise ValueError("class_ids must not be empty")

    if not np.issubdtype(ids.dtype, np.integer):
        raise TypeError(
            "class_ids must contain integer class identifiers"
        )

    ids = ids.astype(np.int32, copy=False)

    if np.any(ids < 0) or np.any(ids >= int(num_classes)):
        raise ValueError(
            f"class_ids must lie in [0, {int(num_classes) - 1}]"
        )

    return ids


def _sample_pixel_rgb(
    pixel_params: tf.Tensor,
    *,
    temperature: float,
    min_log_scale: float,
    max_log_scale: float,
    seed: int,
    pixel_index: int,
) -> tf.Tensor:
    """Sample one batch of governed RGB pixel vectors reproducibly."""

    mixed_seed = (
        int(seed)
        + int(pixel_index)
        * 1_000_003
    )

    rgb = sample_coupled_rgb_from_parameters(
        pixel_params,
        temperature=temperature,
        seed=mixed_seed,
        min_log_scale=min_log_scale,
        max_log_scale=max_log_scale,
    )

    tf.debugging.assert_all_finite(
        rgb,
        "non-finite autoregressive RGB pixel sample",
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


def sample_revision_pixelcnn(
    model: tf.keras.Model,
    *,
    class_ids: np.ndarray,
    img_shape: tuple[int, int, int],
    num_classes: int,
    temperature: float = 1.0,
    min_log_scale: float = -7.0,
    max_log_scale: float = 3.0,
    seed: int = 42,
) -> np.ndarray:
    """Generate canonical RGB images with raster-order autoregressive sampling.

    Returns:
        float32 array with shape (B,H,W,3) and values in [0,1].
    """
    height, width, channels = tuple(
        int(v)
        for v in img_shape
    )

    if channels != 3:
        raise ValueError(
            "revision-2026 autoregressive sampler "
            "requires canonical C=3 input"
        )

    if int(num_classes) <= 0:
        raise ValueError(
            "num_classes must be positive"
        )

    temperature = float(temperature)

    if temperature < 0.0:
        raise ValueError(
            "temperature must be non-negative"
        )

    if float(min_log_scale) >= float(max_log_scale):
        raise ValueError(
            "min_log_scale must be smaller than max_log_scale"
        )

    ids = _validate_class_ids(
        class_ids,
        num_classes=num_classes,
    )

    batch_size = int(ids.shape[0])

    onehot = tf.one_hot(
        ids,
        depth=int(num_classes),
        dtype=tf.float32,
    )

    images = tf.zeros(
        (batch_size, height, width, channels),
        dtype=tf.float32,
    )

    expected_param_channels = OBSERVATION_PARAM_CHANNELS

    for row in range(height):
        for col in range(width):
            params = model(
                [images, onehot],
                training=False,
            )

            tf.debugging.assert_rank(
                params,
                4,
                message="PixelCNN output must be rank-4 NHWC",
            )

            tf.debugging.assert_equal(
                tf.shape(params)[-1],
                expected_param_channels,
                message=(
                    "revision-2026 PixelCNN must output "
                    "nine governed observation parameters per pixel"
                ),
            )

            pixel_params = params[
                :,
                row,
                col,
                :,
            ]

            linear_index = row * width + col

            pixel_rgb = _sample_pixel_rgb(
                pixel_params,
                temperature=temperature,
                min_log_scale=min_log_scale,
                max_log_scale=max_log_scale,
                seed=int(seed),
                pixel_index=linear_index,
            )

            indices = tf.stack(
                [
                    tf.range(
                        batch_size,
                        dtype=tf.int32,
                    ),
                    tf.fill(
                        [batch_size],
                        tf.cast(row, tf.int32),
                    ),
                    tf.fill(
                        [batch_size],
                        tf.cast(col, tf.int32),
                    ),
                ],
                axis=1,
            )

            images = tf.tensor_scatter_nd_update(
                images,
                indices,
                pixel_rgb,
            )

    tf.debugging.assert_all_finite(
        images,
        "non-finite complete autoregressive sample",
    )

    tf.debugging.assert_greater_equal(
        images,
        tf.constant(0.0, dtype=tf.float32),
    )

    tf.debugging.assert_less_equal(
        images,
        tf.constant(1.0, dtype=tf.float32),
    )

    result = images.numpy().astype(
        np.float32,
        copy=False,
    )

    if result.shape != (
        batch_size,
        height,
        width,
        channels,
    ):
        raise RuntimeError(
            "autoregressive sampler produced unexpected shape: "
            f"{result.shape}"
        )

    return result


__all__ = [
    "sample_revision_pixelcnn",
]
