"""Negative architecture fingerprint boundary regression tests.

Uses untrained, disposable VAE architectures only.
No real dataset access, training, or checkpoint publication.
"""

from unittest.mock import patch

import optree
import keras
import pytest
import tensorflow as tf

from trustforge.models.paper01_revision_2026 import (
    variational_autoencoder_model as model,
    variational_autoencoder_checkpoint_fingerprint_registry as registry,
)


def make_models(class_count):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(42)

    return model.build_models(
        latent_dim=100,
        num_classes=class_count,
    )


@pytest.mark.parametrize(
    "dataset_id,encoder_classes,decoder_classes",
    [
        ("ustc_tfc2016", 9, 5),
        ("cicmaldroid2020", 5, 9),
        ("malex", 2, 47),
        ("malnet_image", 47, 2),
    ],
)
def test_mixed_encoder_decoder_pair_rejected(
    dataset_id,
    encoder_classes,
    decoder_classes,
):
    encoder, _ = make_models(encoder_classes)
    _, decoder = make_models(decoder_classes)

    with pytest.raises(
        ValueError,
        match="VAE architecture fingerprint mismatch",
    ):
        registry.validate_vae_architecture_fingerprint(
            dataset_id=dataset_id,
            encoder=encoder,
            decoder=decoder,
        )


def test_incomplete_reference_registry_rejected():
    encoder, decoder = make_models(9)

    incomplete = {
        key: value
        for key, value in registry.REFERENCE_FINGERPRINTS.items()
        if key != "malex"
    }

    with patch.object(
        registry,
        "REFERENCE_FINGERPRINTS",
        incomplete,
    ):
        with pytest.raises(
            RuntimeError,
            match="architecture fingerprint registry incomplete",
        ):
            registry.validate_vae_architecture_fingerprint(
                dataset_id="ustc_tfc2016",
                encoder=encoder,
                decoder=decoder,
            )


@pytest.mark.parametrize(
    "invalid_digest",
    [
        "",
        "not-a-sha256-digest",
        "0" * 63,
        "G" * 64,
        None,
        123,
    ],
)
def test_malformed_reference_digest_rejected(invalid_digest):
    encoder, decoder = make_models(9)

    modified = dict(registry.REFERENCE_FINGERPRINTS)
    modified["ustc_tfc2016"] = invalid_digest

    with patch.object(
        registry,
        "REFERENCE_FINGERPRINTS",
        modified,
    ):
        with pytest.raises(
            RuntimeError,
            match="invalid pinned architecture fingerprint",
        ):
            registry.validate_vae_architecture_fingerprint(
                dataset_id="ustc_tfc2016",
                encoder=encoder,
                decoder=decoder,
            )


def test_registry_restored_after_negative_checks():
    assert set(registry.REFERENCE_FINGERPRINTS) == {
        "malex",
        "cicmaldroid2020",
        "ustc_tfc2016",
        "malnet_image",
    }

    encoder, decoder = make_models(9)

    observed = registry.validate_vae_architecture_fingerprint(
        dataset_id="ustc_tfc2016",
        encoder=encoder,
        decoder=decoder,
    )

    assert observed == registry.expected_vae_architecture_fingerprint(
        dataset_id="ustc_tfc2016",
    )
