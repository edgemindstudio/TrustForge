"""Candidate paired-VAE architecture fingerprint regression tests.

Synthetic, untrained models only. No checkpoint publication,
real-dataset access, or training.
"""

from copy import deepcopy
import hashlib
import json

import optree
import keras
import pytest
import tensorflow as tf

from trustforge.models.paper01_revision_2026 import (
    variational_autoencoder_model as model,
    variational_autoencoder_checkpoint_fingerprint as fingerprint,
)


DATASETS = {
    "malex": 2,
    "cicmaldroid2020": 5,
    "ustc_tfc2016": 9,
    "malnet_image": 47,
}


def make_models(class_count, seed=42):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)

    return model.build_models(
        latent_dim=100,
        num_classes=class_count,
    )


@pytest.mark.parametrize(
    "class_count",
    [2, 5, 9, 47],
)
def test_fingerprint_stable_across_seeds(class_count):
    first_encoder, first_decoder = make_models(
        class_count,
        seed=42,
    )

    first = fingerprint.fingerprint_vae_architecture(
        encoder=first_encoder,
        decoder=first_decoder,
        class_count=class_count,
    )

    second_encoder, second_decoder = make_models(
        class_count,
        seed=43,
    )

    second = fingerprint.fingerprint_vae_architecture(
        encoder=second_encoder,
        decoder=second_decoder,
        class_count=class_count,
    )

    assert first == second
    assert len(first) == 64
    int(first, 16)


@pytest.mark.parametrize(
    "class_count",
    [2, 5, 9, 47],
)
def test_descriptor_contains_complete_variable_inventory(class_count):
    encoder, decoder = make_models(class_count)

    descriptor = fingerprint.describe_vae_architecture(
        encoder=encoder,
        decoder=decoder,
        class_count=class_count,
    )

    assert descriptor["class_count"] == class_count
    assert descriptor["latent_dim"] == 100
    assert (
        descriptor["fingerprint_schema_version"]
        == fingerprint.FINGERPRINT_SCHEMA_VERSION
    )

    assert len(
        descriptor["encoder"]["variable_specification"]
    ) == 16

    assert len(
        descriptor["decoder"]["variable_specification"]
    ) == 8

    assert (
        descriptor["encoder"]["keras_configuration"]["name"]
        == encoder.name
    )

    assert (
        descriptor["decoder"]["keras_configuration"]["name"]
        == decoder.name
    )

    json.dumps(
        descriptor,
        sort_keys=True,
        allow_nan=False,
    )


def test_different_class_counts_have_different_fingerprints():
    fingerprints = set()

    for count in (2, 5, 9, 47):
        encoder, decoder = make_models(count)

        digest = fingerprint.fingerprint_vae_architecture(
            encoder=encoder,
            decoder=decoder,
            class_count=count,
        )

        fingerprints.add(digest)

    assert len(fingerprints) == 4


def test_encoder_configuration_change_changes_digest():
    encoder, decoder = make_models(9)

    original = fingerprint.describe_vae_architecture(
        encoder=encoder,
        decoder=decoder,
        class_count=9,
    )

    modified = deepcopy(original)

    layers = modified["encoder"]["keras_configuration"]["layers"]

    matches = [
        layer
        for layer in layers
        if layer["config"]["name"] == "encoder_conv1"
    ]

    assert len(matches) == 1
    matches[0]["config"]["activation"] = "linear"

    assert fingerprint._canonical_json_bytes(
        original
    ) != fingerprint._canonical_json_bytes(
        modified
    )


def test_decoder_configuration_change_changes_digest():
    encoder, decoder = make_models(9)

    original = fingerprint.describe_vae_architecture(
        encoder=encoder,
        decoder=decoder,
        class_count=9,
    )

    modified = deepcopy(original)

    layers = modified["decoder"]["keras_configuration"]["layers"]

    matches = [
        layer
        for layer in layers
        if layer["config"]["name"] == "decoder_mean"
    ]

    assert len(matches) == 1
    matches[0]["config"]["activation"] = "sigmoid"

    assert fingerprint._canonical_json_bytes(
        original
    ) != fingerprint._canonical_json_bytes(
        modified
    )


def test_variable_specification_change_changes_digest():
    encoder, decoder = make_models(9)

    original = fingerprint.describe_vae_architecture(
        encoder=encoder,
        decoder=decoder,
        class_count=9,
    )

    modified = deepcopy(original)

    modified["encoder"]["variable_specification"][0][
        "trainable"
    ] = False

    assert fingerprint._canonical_json_bytes(
        original
    ) != fingerprint._canonical_json_bytes(
        modified
    )


@pytest.mark.parametrize(
    "invalid_class_count",
    [0, -1, True, 1.5, "9"],
)
def test_invalid_class_count_rejected(invalid_class_count):
    encoder, decoder = make_models(9)

    with pytest.raises(ValueError):
        fingerprint.describe_vae_architecture(
            encoder=encoder,
            decoder=decoder,
            class_count=invalid_class_count,
        )


@pytest.mark.parametrize(
    "invalid_latent_dim",
    [64, 0, True, "100"],
)
def test_invalid_latent_dimension_rejected(invalid_latent_dim):
    encoder, decoder = make_models(9)

    with pytest.raises(ValueError):
        fingerprint.describe_vae_architecture(
            encoder=encoder,
            decoder=decoder,
            class_count=9,
            latent_dim=invalid_latent_dim,
        )
