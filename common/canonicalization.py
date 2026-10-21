"""
TrustForge canonical image preprocessing for Paper 1 revision 2026.

Canonical model-facing representation:
    shape: (40, 40, 3)
    layout: NHWC
    dtype: float32
    range: [0, 1]

This module does not infer source numeric ranges heuristically.
Callers must explicitly declare the source representation.
"""

from __future__ import annotations

from enum import Enum
from typing import Tuple

import numpy as np
import tensorflow as tf


CANONICAL_HEIGHT = 40
CANONICAL_WIDTH = 40
CANONICAL_CHANNELS = 3
CANONICAL_SHAPE = (
    CANONICAL_HEIGHT,
    CANONICAL_WIDTH,
    CANONICAL_CHANNELS,
)


class SourceNumericRange(str, Enum):
    """Explicit supported source numeric conventions."""

    ZERO_ONE = "zero_one"
    MINUS_ONE_ONE = "minus_one_one"
    UINT8_255 = "uint8_255"


def normalize_source_numeric(
    x: np.ndarray,
    source_range: SourceNumericRange,
) -> np.ndarray:
    """
    Convert an explicitly declared source numeric representation to float32
    in [0,1].

    Supported source conventions:
    - ZERO_ONE:      source is already in [0,1]
    - MINUS_ONE_ONE: source is in [-1,1]
    - UINT8_255:     source represents byte intensities in [0,255]
    """
    arr = np.asarray(x)

    if source_range == SourceNumericRange.ZERO_ONE:
        out = arr.astype(np.float32, copy=False)

    elif source_range == SourceNumericRange.MINUS_ONE_ONE:
        out = arr.astype(np.float32, copy=False)
        out = (out + 1.0) / 2.0

    elif source_range == SourceNumericRange.UINT8_255:
        out = arr.astype(np.float32, copy=False) / 255.0

    else:
        raise ValueError(
            f"Unsupported source numeric range: {source_range!r}"
        )

    return np.clip(out, 0.0, 1.0)


def _ensure_nhwc(x: np.ndarray) -> np.ndarray:
    """
    Normalize supported image-array layouts to NHWC.

    Supported:
    - (H,W)
    - (H,W,1)
    - (H,W,3)
    - (N,H,W)
    - (N,H,W,1)
    - (N,H,W,3)
    """
    arr = np.asarray(x)

    if arr.ndim == 2:
        arr = arr[None, ..., None]

    elif arr.ndim == 3:
        if arr.shape[-1] in (1, 3):
            arr = arr[None, ...]
        else:
            arr = arr[..., None]

    elif arr.ndim == 4:
        if arr.shape[-1] not in (1, 3):
            raise ValueError(
                f"Expected NHWC with 1 or 3 channels, got shape {arr.shape}"
            )

    else:
        raise ValueError(
            f"Unsupported image-array shape {arr.shape}"
        )

    if arr.shape[-1] not in (1, 3):
        raise ValueError(
            f"Expected 1 or 3 channels, got shape {arr.shape}"
        )

    return arr


def canonicalize_images(
    x: np.ndarray,
    *,
    source_range: SourceNumericRange,
) -> np.ndarray:
    """
    Convert source images to the TrustForge canonical representation.

    Processing order:
    1. explicit source numeric normalization to float32 [0,1]
    2. convert supported shapes to NHWC
    3. resize spatially to 40x40 using bilinear interpolation with antialiasing
       when spatial resizing is required
    4. replicate grayscale channel to RGB
    5. verify final shape
    6. clip defensively to [0,1]
    """
    arr = normalize_source_numeric(x, source_range)
    arr = _ensure_nhwc(arr)

    if tuple(arr.shape[1:3]) != (CANONICAL_HEIGHT, CANONICAL_WIDTH):
        arr = tf.image.resize(
            arr,
            size=(CANONICAL_HEIGHT, CANONICAL_WIDTH),
            method="bilinear",
            antialias=True,
        ).numpy()

    arr = arr.astype(np.float32, copy=False)

    if arr.shape[-1] == 1:
        arr = np.repeat(arr, CANONICAL_CHANNELS, axis=-1)

    if arr.shape[-1] != CANONICAL_CHANNELS:
        raise ValueError(
            f"Canonical channel count must be 3, got shape {arr.shape}"
        )

    expected_tail: Tuple[int, int, int] = CANONICAL_SHAPE

    if tuple(arr.shape[1:]) != expected_tail:
        raise ValueError(
            f"Canonical shape mismatch: expected (N,{expected_tail}), "
            f"got {arr.shape}"
        )

    return np.clip(arr, 0.0, 1.0).astype(np.float32, copy=False)
