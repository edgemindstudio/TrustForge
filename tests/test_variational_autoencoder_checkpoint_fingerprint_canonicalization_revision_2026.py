"""VAE architecture fingerprint canonicalization regressions.

Disposable, untrained networks only.
No real dataset access, training, or checkpoint publication.
"""

import copy

import optree
import keras
import pytest
import tensorflow as tf

from trustforge.models.paper01_revision_2026 import (
    variational_autoencoder_model as model,
    variational_autoencoder_checkpoint_fingerprint as fingerprint,
    variational_autoencoder_checkpoint_fingerprint_registry as registry,
)


DATASETS = {
    "malex": 2,
    "cicmaldroid2020": 5,
    "ustc_tfc2016": 9,
    "malnet_image": 47,
}


@pytest.mark.parametrize("dataset_id", sorted(DATASETS))
def test_consecutive_constructions_have_same_fingerprint(dataset_id):
    class_count = DATASETS[dataset_id]

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(42)

    first_encoder, first_decoder = model.build_models(
        latent_dim=100,
        num_classes=class_count,
    )

    second_encoder, second_decoder = model.build_models(
        latent_dim=100,
        num_classes=class_count,
    )

    first = fingerprint.fingerprint_vae_architecture(
        encoder=first_encoder,
        decoder=first_decoder,
        class_count=class_count,
    )

    second = fingerprint.fingerprint_vae_architecture(
        encoder=second_encoder,
        decoder=second_decoder,
        class_count=class_count,
    )

    expected = registry.expected_vae_architecture_fingerprint(
        dataset_id=dataset_id,
    )

    assert first == second == expected


@pytest.mark.parametrize("dataset_id", sorted(DATASETS))
def test_canonical_descriptor_idempotent(dataset_id):
    class_count = DATASETS[dataset_id]

    tf.keras.backend.clear_session()

    # Consume default Keras naming counters.
    model.build_models(
        latent_dim=100,
        num_classes=class_count,
    )

    encoder, decoder = model.build_models(
        latent_dim=100,
        num_classes=class_count,
    )

    descriptor = fingerprint.describe_vae_architecture(
        encoder=encoder,
        decoder=decoder,
        class_count=class_count,
    )

    original = copy.deepcopy(
        descriptor["encoder"]["keras_configuration"]
    )

    normalized = fingerprint._canonicalize_encoder_configuration(
        original
    )

    assert normalized == original
    assert descriptor["encoder"]["keras_configuration"] == original


def test_canonicalization_preserves_computational_drift_detection():
    tf.keras.backend.clear_session()

    encoder, decoder = model.build_models(
        latent_dim=100,
        num_classes=9,
    )

    def alter_layer(layer):
        configuration = layer.get_config()

        if layer.name == "encoder_conv1":
            configuration["activation"] = "linear"

        return layer.__class__.from_config(configuration)

    altered_encoder = tf.keras.models.clone_model(
        encoder,
        clone_function=alter_layer,
    )

    original = fingerprint.fingerprint_vae_architecture(
        encoder=encoder,
        decoder=decoder,
        class_count=9,
    )

    altered = fingerprint.fingerprint_vae_architecture(
        encoder=altered_encoder,
        decoder=decoder,
        class_count=9,
    )

    assert original != altered


def test_canonicalization_does_not_mutate_input_configuration():
    tf.keras.backend.clear_session()

    encoder, _ = model.build_models(
        latent_dim=100,
        num_classes=9,
    )

    configuration = encoder.get_config()
    original = copy.deepcopy(configuration)

    fingerprint._canonicalize_encoder_configuration(
        configuration
    )

    assert configuration == original
