"""Configuration-only tests for revision-2026 GAN model factory."""

import pytest

from trustforge.models.paper01_revision_2026.gan_model_factory import (
    GANModelConfig,
    GAN_IMAGE_SHAPE,
    GAN_LATENT_DIM,
    GAN_LEARNING_RATE,
    GAN_BETA_1,
)


@pytest.mark.parametrize("classes", [2, 5, 9, 47])
def test_supported_class_counts(classes):
    config = GANModelConfig(num_classes=classes)

    assert config.num_classes == classes
    assert config.latent_dim == 100
    assert config.learning_rate == 0.0002
    assert config.beta_1 == 0.5
    assert GAN_IMAGE_SHAPE == (40, 40, 3)


@pytest.mark.parametrize(
    "classes",
    [0, -1, True, 2.5, "47"],
)
def test_invalid_class_counts_rejected(classes):
    with pytest.raises(ValueError):
        GANModelConfig(num_classes=classes)


@pytest.mark.parametrize(
    "latent",
    [0, 64, 128, True, 100.0],
)
def test_invalid_latent_dimensions_rejected(latent):
    with pytest.raises(ValueError):
        GANModelConfig(num_classes=47, latent_dim=latent)


@pytest.mark.parametrize(
    "lr",
    [-1.0, 0.0, 0.0001, float("nan"), True],
)
def test_invalid_learning_rates_rejected(lr):
    with pytest.raises(ValueError):
        GANModelConfig(num_classes=47, learning_rate=lr)


@pytest.mark.parametrize(
    "beta",
    [-0.1, 0.9, 1.0, float("nan"), True],
)
def test_invalid_beta_1_values_rejected(beta):
    with pytest.raises(ValueError):
        GANModelConfig(num_classes=47, beta_1=beta)


def test_configuration_is_immutable():
    config = GANModelConfig(num_classes=9)

    with pytest.raises(AttributeError):
        config.num_classes = 47


def test_governed_constants():
    assert GAN_LATENT_DIM == 100
    assert GAN_LEARNING_RATE == 2e-4
    assert GAN_BETA_1 == 0.5
