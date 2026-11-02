"""Candidate VAE architecture fingerprint registry tests.

Synthetic untrained architectures only.
No checkpoint publication, dataset reads, or training.
"""

import optree
import keras
import pytest
import tensorflow as tf

from trustforge.models.paper01_revision_2026 import (
    variational_autoencoder_model as model,
    variational_autoencoder_checkpoint_fingerprint_registry as registry,
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


@pytest.mark.parametrize("dataset_id", sorted(DATASETS))
def test_governed_architecture_accepted(dataset_id):
    encoder, decoder = make_models(DATASETS[dataset_id])

    actual = registry.validate_vae_architecture_fingerprint(
        dataset_id=dataset_id,
        encoder=encoder,
        decoder=decoder,
    )

    assert actual == registry.expected_vae_architecture_fingerprint(
        dataset_id=dataset_id,
    )


@pytest.mark.parametrize("dataset_id", sorted(DATASETS))
def test_pinned_reference_independent_of_initialization_seed(
    dataset_id,
):
    encoder, decoder = make_models(
        DATASETS[dataset_id],
        seed=43,
    )

    actual = registry.validate_vae_architecture_fingerprint(
        dataset_id=dataset_id,
        encoder=encoder,
        decoder=decoder,
    )

    assert actual == registry.REFERENCE_FINGERPRINTS[dataset_id]


@pytest.mark.parametrize(
    "dataset_id,wrong_classes",
    [
        ("malex", 9),
        ("cicmaldroid2020", 2),
        ("ustc_tfc2016", 47),
        ("malnet_image", 5),
    ],
)
def test_cross_class_architecture_rejected(
    dataset_id,
    wrong_classes,
):
    encoder, decoder = make_models(wrong_classes)

    with pytest.raises(
        ValueError,
        match="VAE architecture fingerprint mismatch",
    ):
        registry.validate_vae_architecture_fingerprint(
            dataset_id=dataset_id,
            encoder=encoder,
            decoder=decoder,
        )


def test_modified_pinned_reference_rejected(monkeypatch):
    encoder, decoder = make_models(9)

    monkeypatch.setitem(
        registry.REFERENCE_FINGERPRINTS,
        "ustc_tfc2016",
        "0" * 64,
    )

    with pytest.raises(
        ValueError,
        match="VAE architecture fingerprint mismatch",
    ):
        registry.validate_vae_architecture_fingerprint(
            dataset_id="ustc_tfc2016",
            encoder=encoder,
            decoder=decoder,
        )


@pytest.mark.parametrize(
    "dataset_id",
    ["unknown", "", "candidate_v1"],
)
def test_unknown_dataset_rejected(dataset_id):
    with pytest.raises(ValueError):
        registry.expected_vae_architecture_fingerprint(
            dataset_id=dataset_id,
        )


@pytest.mark.parametrize("dataset_id", [None, 9, True])
def test_nonstring_dataset_rejected(dataset_id):
    with pytest.raises(TypeError):
        registry.expected_vae_architecture_fingerprint(
            dataset_id=dataset_id,
        )


def test_registry_covers_governed_datasets():
    assert set(registry.REFERENCE_FINGERPRINTS) == set(DATASETS)


def test_registry_digest_format():
    for dataset_id in DATASETS:
        digest = registry.expected_vae_architecture_fingerprint(
            dataset_id=dataset_id,
        )

        assert len(digest) == 64
        int(digest, 16)
