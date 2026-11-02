"""Full-model NPZ regression tests for the revised Paper 1 VAE.

All models are untrained. No production datasets, checkpoint
files, or training operations are involved.
"""

import optree
import keras
import numpy as np
import pytest
import tensorflow as tf

from trustforge.models.paper01_revision_2026 import (
    variational_autoencoder_model as model,
    variational_autoencoder_checkpoint_state as state_module,
    variational_autoencoder_checkpoint_npz as codec,
)


DATASETS = {
    "malex": 2,
    "cicmaldroid2020": 5,
    "ustc_tfc2016": 9,
    "malnet_image": 47,
}


@pytest.mark.parametrize(
    "dataset_id",
    sorted(DATASETS),
)
def test_complete_model_npz_round_trip(dataset_id):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(42)

    class_count = DATASETS[dataset_id]

    encoder, decoder = model.build_models(
        latent_dim=100,
        num_classes=class_count,
    )

    original = state_module.capture_vae_state(
        encoder=encoder,
        decoder=decoder,
    )

    for role, network, expected_count in (
        ("encoder", encoder, 16),
        ("decoder", decoder, 8),
    ):
        specification = state_module.variable_specification(
            network,
            role=role,
        )

        assert len(specification) == expected_count

        raw = codec.encode_network_state(
            role=role,
            state=original[role],
            specification=specification,
        )

        assert 0 < len(raw) <= codec.MAX_ARCHIVE_BYTES

        decoded = codec.decode_network_state(
            role=role,
            raw=raw,
            specification=specification,
        )

        assert set(decoded) == set(original[role])

        for key in original[role]:
            np.testing.assert_array_equal(
                decoded[key],
                original[role][key],
            )


@pytest.mark.parametrize(
    "dataset_id",
    sorted(DATASETS),
)
def test_complete_model_npz_encoding_deterministic(dataset_id):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(42)

    encoder, decoder = model.build_models(
        latent_dim=100,
        num_classes=DATASETS[dataset_id],
    )

    original = state_module.capture_vae_state(
        encoder=encoder,
        decoder=decoder,
    )

    for role, network in (
        ("encoder", encoder),
        ("decoder", decoder),
    ):
        specification = state_module.variable_specification(
            network,
            role=role,
        )

        first = codec.encode_network_state(
            role=role,
            state=original[role],
            specification=specification,
        )

        second = codec.encode_network_state(
            role=role,
            state=original[role],
            specification=specification,
        )

        assert first == second
