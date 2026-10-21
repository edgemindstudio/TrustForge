"""Tests for the TrustForge Paper 1 revision canonical representation."""

import numpy as np

from common.canonicalization import (
    CANONICAL_CHANNELS,
    CANONICAL_HEIGHT,
    CANONICAL_SHAPE,
    CANONICAL_WIDTH,
    SourceNumericRange,
    canonicalize_images,
    normalize_source_numeric,
)


def test_canonical_constants():
    assert CANONICAL_HEIGHT == 40
    assert CANONICAL_WIDTH == 40
    assert CANONICAL_CHANNELS == 3
    assert CANONICAL_SHAPE == (40, 40, 3)


def test_zero_one_normalization_preserves_values():
    x = np.array([0.0, 0.25, 0.5, 1.0], dtype=np.float64)

    result = normalize_source_numeric(
        x,
        SourceNumericRange.ZERO_ONE,
    )

    assert result.dtype == np.float32
    np.testing.assert_allclose(
        result,
        np.array([0.0, 0.25, 0.5, 1.0], dtype=np.float32),
    )


def test_minus_one_one_maps_exactly_to_zero_one():
    x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)

    result = normalize_source_numeric(
        x,
        SourceNumericRange.MINUS_ONE_ONE,
    )

    expected = np.array(
        [0.0, 0.25, 0.5, 0.75, 1.0],
        dtype=np.float32,
    )

    np.testing.assert_allclose(result, expected)


def test_uint8_maps_exactly_to_zero_one():
    x = np.array([0, 127, 255], dtype=np.uint8)

    result = normalize_source_numeric(
        x,
        SourceNumericRange.UINT8_255,
    )

    expected = np.array(
        [0.0, 127.0 / 255.0, 1.0],
        dtype=np.float32,
    )

    np.testing.assert_allclose(result, expected)


def test_40x40_grayscale_is_not_spatially_changed():
    rng = np.random.default_rng(42)
    x = rng.random((2, 40, 40, 1)).astype(np.float32)

    result = canonicalize_images(
        x,
        source_range=SourceNumericRange.ZERO_ONE,
    )

    assert result.shape == (2, 40, 40, 3)
    assert result.dtype == np.float32

    np.testing.assert_allclose(result[..., 0], x[..., 0])
    np.testing.assert_allclose(result[..., 1], x[..., 0])
    np.testing.assert_allclose(result[..., 2], x[..., 0])


def test_grayscale_resize_produces_three_identical_channels():
    rng = np.random.default_rng(42)
    x = rng.random((3, 12, 12, 1)).astype(np.float32)

    result = canonicalize_images(
        x,
        source_range=SourceNumericRange.ZERO_ONE,
    )

    assert result.shape == (3, 40, 40, 3)

    np.testing.assert_allclose(result[..., 0], result[..., 1])
    np.testing.assert_allclose(result[..., 1], result[..., 2])


def test_rgb_resize_preserves_channel_distinction():
    x = np.zeros((1, 256, 256, 3), dtype=np.uint8)
    x[..., 0] = 255
    x[..., 1] = 128
    x[..., 2] = 0

    result = canonicalize_images(
        x,
        source_range=SourceNumericRange.UINT8_255,
    )

    assert result.shape == (1, 40, 40, 3)
    assert result.dtype == np.float32

    assert not np.allclose(result[..., 0], result[..., 1])
    assert not np.allclose(result[..., 1], result[..., 2])

    np.testing.assert_allclose(result[..., 0], 1.0, atol=1e-6)
    np.testing.assert_allclose(result[..., 1], 128.0 / 255.0, atol=1e-6)
    np.testing.assert_allclose(result[..., 2], 0.0, atol=1e-6)


def test_single_2d_grayscale_image_is_supported():
    x = np.full((256, 256), 127, dtype=np.uint8)

    result = canonicalize_images(
        x,
        source_range=SourceNumericRange.UINT8_255,
    )

    assert result.shape == (1, 40, 40, 3)

    np.testing.assert_allclose(result[..., 0], result[..., 1])
    np.testing.assert_allclose(result[..., 1], result[..., 2])


def test_output_is_bounded_zero_one():
    x = np.array(
        [[[-2.0], [2.0]], [[0.0], [1.0]]],
        dtype=np.float32,
    )

    result = canonicalize_images(
        x,
        source_range=SourceNumericRange.ZERO_ONE,
    )

    assert float(result.min()) >= 0.0
    assert float(result.max()) <= 1.0
