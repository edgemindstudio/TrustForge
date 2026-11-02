"""Governed VAE loader synthetic round-trip and fingerprint tests.

Uses disposable untrained models and temporary checkpoint artifacts.
Does not establish training authenticity or authorize real-data training.
"""

import hashlib

import optree
import keras
import numpy as np
import pytest
import tensorflow as tf

from trustforge.provenance import canonical_json_bytes

from trustforge.models.paper01_revision_2026 import (
    variational_autoencoder_model as model,
    variational_autoencoder_checkpoint_state as state_module,
    variational_autoencoder_checkpoint_npz as npz,
    variational_autoencoder_checkpoint_loader as loader,
    variational_autoencoder_checkpoint_compatibility as compatibility,
    variational_autoencoder_checkpoint_restore as restore_module,
)

from tests.test_variational_autoencoder_checkpoint_artifact_revision_2026 import (
    make_manifest,
)

from tests.test_variational_autoencoder_checkpoint_provenance_revision_2026 import (
    evidence_for,
)


DATASET_ID = "ustc_tfc2016"
SEED = 42
CLASS_COUNT = 9
CLASS_MAPPING = [f"class_{i}" for i in range(CLASS_COUNT)]


@pytest.fixture
def synthetic_bundle(tmp_path, monkeypatch):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)

    encoder, decoder = model.build_models(
        latent_dim=100,
        num_classes=CLASS_COUNT,
    )

    expected_state = state_module.capture_vae_state(
        encoder=encoder,
        decoder=decoder,
    )

    archives = {}

    for role, network in (
        ("encoder", encoder),
        ("decoder", decoder),
    ):
        specification = state_module.variable_specification(
            network,
            role=role,
        )

        archives[f"vae_{role}_state.npz"] = (
            npz.encode_network_state(
                role=role,
                state=expected_state[role],
                specification=specification,
            )
        )

    provenance = evidence_for(DATASET_ID)
    provenance_bytes = canonical_json_bytes(provenance)

    metadata = make_manifest()
    metadata["architecture_version"] = (
        compatibility.CANDIDATE_ARCHITECTURE_VERSION
    )
    metadata["state_files"] = {
        filename: hashlib.sha256(raw).hexdigest()
        for filename, raw in archives.items()
    }
    metadata["dataset_provenance_sha256"] = (
        hashlib.sha256(provenance_bytes).hexdigest()
    )

    (tmp_path / loader.MANIFEST_FILENAME).write_bytes(
        canonical_json_bytes(metadata) + b"\n"
    )
    (tmp_path / loader.VAE_PROVENANCE_FILENAME).write_bytes(
        provenance_bytes
    )

    for filename, raw in archives.items():
        (tmp_path / filename).write_bytes(raw)

    monkeypatch.setenv(
        "SLURM_JOB_ID",
        "synthetic-test-only",
    )

    return {
        "directory": tmp_path,
        "metadata": metadata,
        "provenance": provenance,
        "expected_state": expected_state,
    }


def load_bundle(bundle):
    return loader.load_governed_vae_checkpoint(
        bundle["directory"],
        requested_dataset_id=DATASET_ID,
        requested_experimental_seed=SEED,
        requested_class_mapping=CLASS_MAPPING,
    )


def test_synthetic_checkpoint_exact_roundtrip(synthetic_bundle):
    restored = load_bundle(synthetic_bundle)

    assert restored["metadata"] == synthetic_bundle["metadata"]
    assert (
        restored["dataset_provenance"]
        == synthetic_bundle["provenance"]
    )

    actual_state = state_module.capture_vae_state(
        encoder=restored["encoder"],
        decoder=restored["decoder"],
    )

    expected_state = synthetic_bundle["expected_state"]

    assert set(actual_state) == {"encoder", "decoder"}

    for role in ("encoder", "decoder"):
        assert set(actual_state[role]) == set(expected_state[role])

        for key, expected in expected_state[role].items():
            np.testing.assert_array_equal(
                actual_state[role][key],
                expected,
            )


def test_fingerprint_mismatch_blocks_decode_and_assignment(
    synthetic_bundle,
    monkeypatch,
):
    fingerprint_calls = []

    def reject_fingerprint(*, dataset_id, encoder, decoder):
        fingerprint_calls.append(dataset_id)
        raise ValueError(
            "VAE architecture fingerprint mismatch"
        )

    def forbidden_decode(**kwargs):
        raise AssertionError(
            "NPZ DECODING MUST NOT OCCUR"
        )

    def forbidden_prepare(**kwargs):
        raise AssertionError(
            "STATE PREPARATION MUST NOT OCCUR"
        )

    monkeypatch.setattr(
        loader,
        "validate_vae_architecture_fingerprint",
        reject_fingerprint,
    )

    monkeypatch.setattr(
        loader,
        "decode_network_state",
        forbidden_decode,
    )

    monkeypatch.setattr(
        restore_module,
        "prepare_vae_checkpoint_restoration",
        forbidden_prepare,
    )

    with pytest.raises(
        ValueError,
        match="VAE architecture fingerprint mismatch",
    ):
        load_bundle(synthetic_bundle)

    assert fingerprint_calls == [DATASET_ID]
