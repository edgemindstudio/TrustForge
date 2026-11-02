"""Regression tests for revision-2026 GAN image domains."""

import numpy as np
import pytest

from trustforge.models.paper01_revision_2026.gan_image_domain import (
    canonical_to_gan,
    gan_to_canonical,
)


def test_canonical_endpoints():
    images = np.zeros((2, 40, 40, 3), dtype=np.float32)
    images[1] = 1.0

    converted = canonical_to_gan(images)

    assert np.all(converted[0] == -1.0)
    assert np.all(converted[1] == 1.0)
    assert converted.dtype == np.float32


def test_gan_endpoints():
    images = np.zeros((2, 40, 40, 3), dtype=np.float32)
    images[0] = -1.0
    images[1] = 1.0

    converted = gan_to_canonical(images)

    assert np.all(converted[0] == 0.0)
    assert np.all(converted[1] == 1.0)
    assert converted.dtype == np.float32


def test_midpoint():
    images = np.full(
        (1, 40, 40, 3), 0.5, dtype=np.float32
    )

    converted = canonical_to_gan(images)

    assert np.all(converted == 0.0)


def test_round_trip():
    rng = np.random.default_rng(2026)
    images = rng.uniform(
        0.0, 1.0, size=(7, 40, 40, 3)
    ).astype(np.float32)

    restored = gan_to_canonical(canonical_to_gan(images))

    assert restored.shape == images.shape
    assert restored.dtype == np.float32

    np.testing.assert_allclose(
        restored,
        images,
        rtol=0.0,
        atol=1e-7,
    )


@pytest.mark.parametrize(
    "shape",
    [
        (40, 40, 3),
        (1, 40, 40, 1),
        (1, 40, 40, 4),
        (1, 32, 32, 3),
        (0, 40, 40, 3),
    ],
)
def test_invalid_shapes_rejected(shape):
    images = np.zeros(shape, dtype=np.float32)

    with pytest.raises(ValueError):
        canonical_to_gan(images)


@pytest.mark.parametrize(
    "dtype",
    [np.float64, np.float16, np.uint8],
)
def test_invalid_dtypes_rejected(dtype):
    images = np.zeros((1, 40, 40, 3), dtype=dtype)

    with pytest.raises(TypeError):
        canonical_to_gan(images)


@pytest.mark.parametrize(
    "value",
    [np.nan, np.inf, -np.inf, -0.01, 1.01],
)
def test_invalid_canonical_values_rejected(value):
    images = np.zeros((1, 40, 40, 3), dtype=np.float32)
    images[0, 0, 0, 0] = value

    with pytest.raises(ValueError):
        canonical_to_gan(images)


@pytest.mark.parametrize(
    "value",
    [np.nan, np.inf, -np.inf, -1.01, 1.01],
)
def test_invalid_gan_values_rejected(value):
    images = np.zeros((1, 40, 40, 3), dtype=np.float32)
    images[0, 0, 0, 0] = value

    with pytest.raises(ValueError):
        gan_to_canonical(images)


def test_inputs_are_not_modified():
    images = np.full(
        (1, 40, 40, 3), 0.25, dtype=np.float32
    )
    before = images.copy()

    canonical_to_gan(images)

    np.testing.assert_array_equal(images, before)


def test_nonarray_input_rejected():
    with pytest.raises(TypeError):
        canonical_to_gan([[0.5]])
