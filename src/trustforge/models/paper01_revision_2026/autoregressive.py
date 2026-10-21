"""Governed revision-2026 conditional PixelCNN primitives.

This module implements only the model and probability primitives for the
revised Paper 1 autoregressive family.

Historical implementations under ``autoregressive/`` remain unchanged.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import initializers, layers, models

from .autoregressive_observation import (
    OBSERVATION_PARAM_CHANNELS,
)


RGB_CHANNELS = 3


def _make_causal_mask(
    kh: int,
    kw: int,
    in_ch: int,
    out_ch: int,
    mask_type: str,
) -> np.ndarray:
    """Create a spatial PixelCNN mask for a convolution kernel."""
    mask_type = str(mask_type).upper()
    if mask_type not in {"A", "B"}:
        raise ValueError("mask_type must be 'A' or 'B'")

    mask = np.ones((kh, kw, in_ch, out_ch), dtype=np.float32)
    center_h = kh // 2
    center_w = kw // 2

    mask[
        center_h,
        center_w + (1 if mask_type == "B" else 0) :,
        :,
        :,
    ] = 0.0
    mask[center_h + 1 :, :, :, :] = 0.0

    return mask


class MaskedConv2D(layers.Layer):
    """Spatially causal masked convolution used by the revised PixelCNN."""

    def __init__(
        self,
        filters: int,
        kernel_size: int | Tuple[int, int],
        mask_type: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.filters = int(filters)
        self.kernel_size = (
            (kernel_size, kernel_size)
            if isinstance(kernel_size, int)
            else tuple(kernel_size)
        )
        self.mask_type = str(mask_type).upper()

        if self.mask_type not in {"A", "B"}:
            raise ValueError("mask_type must be 'A' or 'B'")

    def build(self, input_shape):
        kh, kw = self.kernel_size
        in_ch = int(input_shape[-1])

        mask = _make_causal_mask(
            kh,
            kw,
            in_ch,
            self.filters,
            self.mask_type,
        )

        self.kernel = self.add_weight(
            name="kernel",
            shape=(kh, kw, in_ch, self.filters),
            initializer=initializers.GlorotUniform(),
            trainable=True,
        )
        self.bias = self.add_weight(
            name="bias",
            shape=(self.filters,),
            initializer="zeros",
            trainable=True,
        )
        self._mask = tf.constant(mask, dtype=self.dtype or tf.float32)

        super().build(input_shape)

    def call(self, inputs):
        kernel = self.kernel * self._mask
        output = tf.nn.conv2d(
            inputs,
            kernel,
            strides=1,
            padding="SAME",
        )
        return tf.nn.bias_add(output, self.bias)


def _broadcast_class_condition(
    y_onehot: tf.Tensor,
    *,
    height: int,
    width: int,
    filters: int,
    name: str,
) -> tf.Tensor:
    """Project class labels into an unmasked spatial conditioning bias."""
    cond = layers.Dense(
        filters,
        activation=None,
        name=f"{name}_dense",
    )(y_onehot)
    cond = layers.Reshape(
        (1, 1, filters),
        name=f"{name}_reshape",
    )(cond)
    return layers.Lambda(
        lambda t: tf.tile(t, [1, height, width, 1]),
        name=f"{name}_broadcast",
    )(cond)


def build_revision_pixelcnn(
    img_shape: Tuple[int, int, int],
    num_classes: int,
    *,
    filters: int = 64,
    masked_layers: int = 6,
) -> tf.keras.Model:
    """Build the governed continuous conditional PixelCNN.

    The model predicts nine governed observation parameters per spatial position:

    - 3 RGB base locations
    - 3 RGB raw log scales
    - 3 RGB channel-coupling parameters

    Class information is injected as an unmasked conditioning bias and is
    therefore available even at the first raster-order pixel.
    """
    height, width, channels = tuple(int(v) for v in img_shape)

    if channels != RGB_CHANNELS:
        raise ValueError(
            "revision-2026 PixelCNN requires canonical RGB input with C=3; "
            f"got img_shape={img_shape}"
        )

    if int(num_classes) <= 0:
        raise ValueError("num_classes must be positive")

    if int(filters) <= 0:
        raise ValueError("filters must be positive")

    if int(masked_layers) < 1:
        raise ValueError("masked_layers must be at least 1")

    image_in = layers.Input(
        shape=(height, width, channels),
        name="image",
    )
    class_in = layers.Input(
        shape=(int(num_classes),),
        name="onehot",
    )

    hidden = MaskedConv2D(
        filters,
        kernel_size=7,
        mask_type="A",
        name="maskedA_7x7",
    )(image_in)

    class_bias = _broadcast_class_condition(
        class_in,
        height=height,
        width=width,
        filters=filters,
        name="class_condition_input",
    )

    hidden = layers.Add(name="condition_input")(
        [hidden, class_bias]
    )
    hidden = layers.ReLU(name="relu_input")(hidden)

    for index in range(1, int(masked_layers) + 1):
        residual = MaskedConv2D(
            filters,
            kernel_size=3,
            mask_type="B",
            name=f"maskedB_{index}",
        )(hidden)

        class_bias = _broadcast_class_condition(
            class_in,
            height=height,
            width=width,
            filters=filters,
            name=f"class_condition_{index}",
        )

        hidden = layers.Add(name=f"condition_{index}")(
            [residual, class_bias]
        )
        hidden = layers.ReLU(name=f"relu_{index}")(hidden)

    hidden = layers.Conv2D(
        filters,
        kernel_size=1,
        padding="same",
        activation="relu",
        name="proj_1x1",
    )(hidden)

    params = layers.Conv2D(
        OBSERVATION_PARAM_CHANNELS,
        kernel_size=1,
        padding="same",
        activation=None,
        name="observation_params",
    )(hidden)

    return models.Model(
        [image_in, class_in],
        params,
        name="Revision2026ConditionalPixelCNN",
    )




__all__ = [
    "OBSERVATION_PARAM_CHANNELS",
    "MaskedConv2D",
    "RGB_CHANNELS",
    "build_revision_pixelcnn",
]
