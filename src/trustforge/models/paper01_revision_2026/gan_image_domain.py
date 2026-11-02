"""Strict canonical/GAN image-domain conversion for Paper 1."""

from __future__ import annotations

import numpy as np


IMAGE_SHAPE = (40, 40, 3)


def _validate(images, *, name, low, high):
    if not isinstance(images, np.ndarray):
        raise TypeError(f"{name}: expected numpy.ndarray")

    if images.dtype != np.float32:
        raise TypeError(f"{name}: expected float32")

    if images.ndim != 4 or images.shape[1:] != IMAGE_SHAPE:
        raise ValueError(
            f"{name}: expected (N,40,40,3)"
        )

    if images.shape[0] <= 0:
        raise ValueError(f"{name}: empty image batch")

    if not np.isfinite(images).all():
        raise ValueError(f"{name}: nonfinite image values")

    if np.any(images < low) or np.any(images > high):
        raise ValueError(
            f"{name}: values outside [{low},{high}]"
        )


def canonical_to_gan(images: np.ndarray) -> np.ndarray:
    """Map canonical [0,1] float32 images to GAN [-1,1]."""

    _validate(
        images,
        name="canonical_images",
        low=0.0,
        high=1.0,
    )

    result = images * np.float32(2.0) - np.float32(1.0)

    _validate(
        result,
        name="gan_images",
        low=-1.0,
        high=1.0,
    )

    return result


def gan_to_canonical(images: np.ndarray) -> np.ndarray:
    """Map GAN [-1,1] float32 images to canonical [0,1]."""

    _validate(
        images,
        name="gan_images",
        low=-1.0,
        high=1.0,
    )

    result = (images + np.float32(1.0)) * np.float32(0.5)

    _validate(
        result,
        name="canonical_images",
        low=0.0,
        high=1.0,
    )

    return result
