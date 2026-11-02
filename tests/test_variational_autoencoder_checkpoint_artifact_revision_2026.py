"""Synthetic VAE checkpoint byte-integrity regression tests.

These tests use small temporary byte artifacts, not production
checkpoints or datasets. NPZ decoding is outside this test scope.
"""

from copy import deepcopy
import hashlib
import os

import pytest

from trustforge.models.paper01_revision_2026 import (
    variational_autoencoder_checkpoint_artifact as artifact,
    variational_autoencoder_checkpoint_manifest as manifest,
)


ENCODER_NAME = "vae_encoder_state.npz"
DECODER_NAME = "vae_decoder_state.npz"

ENCODER_BYTES = b"synthetic-encoder-checkpoint-v1"
DECODER_BYTES = b"synthetic-decoder-checkpoint-v1"


def make_manifest():
    return {
        "schema_version": manifest.SCHEMA_VERSION,
        "schema_type": manifest.SCHEMA_TYPE,
        "paper_revision": manifest.PAPER_REVISION,
        "model_family": manifest.MODEL_FAMILY,
        "dataset_id": "ustc_tfc2016",
        "experimental_seed": 42,
        "class_mapping": [f"class_{i}" for i in range(9)],
        "class_count": 9,
        "image_shape": [40, 40, 3],
        "latent_dim": 100,
        "encoder_dense_units": 256,
        "decoder_feature_channels": 64,
        "architecture_version": "candidate_v1",
        "observation_version": "candidate_v1",
        "observation_variance": 1.0,
        "kl_weight": 1.0,
        "kl_annealing": False,
        "completed_epochs": 1,
        "completed_updates": 10,
        "checkpoint_role": "last",
        "selection_policy_id": "candidate_unapproved_v1",
        "validation_criterion": {
            "metric": "validation_objective",
            "direction": "minimize",
            "value": 1.0,
        },
        "validation_seed_policy_id": "candidate_unapproved_v1",
        "effective_configuration": {"batch_size": 8},
        "state_files": {
            ENCODER_NAME: hashlib.sha256(ENCODER_BYTES).hexdigest(),
            DECODER_NAME: hashlib.sha256(DECODER_BYTES).hexdigest(),
        },
        "dataset_provenance_file": "vae_dataset_provenance.json",
        "dataset_provenance_sha256": "c" * 64,
    }


def make_bundle(tmp_path):
    (tmp_path / ENCODER_NAME).write_bytes(ENCODER_BYTES)
    (tmp_path / DECODER_NAME).write_bytes(DECODER_BYTES)
    return make_manifest()


def verify(tmp_path, metadata):
    return artifact.verify_vae_checkpoint_state_artifacts(
        tmp_path,
        metadata,
    )


def test_complete_state_pair_accepted(tmp_path):
    metadata = make_bundle(tmp_path)
    verified = verify(tmp_path, metadata)

    assert verified == {
        ENCODER_NAME: ENCODER_BYTES,
        DECODER_NAME: DECODER_BYTES,
    }


@pytest.mark.parametrize("name", [ENCODER_NAME, DECODER_NAME])
def test_missing_state_file_rejected(tmp_path, name):
    metadata = make_bundle(tmp_path)
    (tmp_path / name).unlink()

    with pytest.raises(ValueError):
        verify(tmp_path, metadata)


@pytest.mark.parametrize("name", [ENCODER_NAME, DECODER_NAME])
def test_modified_state_file_rejected(tmp_path, name):
    metadata = make_bundle(tmp_path)
    (tmp_path / name).write_bytes(b"corrupted-checkpoint")

    with pytest.raises(ValueError, match="SHA-256"):
        verify(tmp_path, metadata)


@pytest.mark.parametrize("name", [ENCODER_NAME, DECODER_NAME])
def test_empty_state_file_rejected(tmp_path, name):
    metadata = make_bundle(tmp_path)
    (tmp_path / name).write_bytes(b"")

    with pytest.raises(ValueError):
        verify(tmp_path, metadata)


def test_unexpected_state_file_rejected(tmp_path):
    metadata = make_bundle(tmp_path)
    (tmp_path / "vae_extra_state.npz").write_bytes(b"extra")

    with pytest.raises(ValueError, match="inventory"):
        verify(tmp_path, metadata)


def test_publication_lock_rejected(tmp_path):
    metadata = make_bundle(tmp_path)
    (tmp_path / ".vae_checkpoint_publication.lock").mkdir()

    with pytest.raises(RuntimeError, match="incomplete"):
        verify(tmp_path, metadata)


@pytest.mark.parametrize("name", [ENCODER_NAME, DECODER_NAME])
def test_symlink_state_file_rejected(tmp_path, name):
    metadata = make_bundle(tmp_path)

    external = tmp_path / "external.bin"
    external.write_bytes(
        ENCODER_BYTES if name == ENCODER_NAME else DECODER_BYTES
    )

    (tmp_path / name).unlink()
    (tmp_path / name).symlink_to(external)

    with pytest.raises(ValueError):
        verify(tmp_path, metadata)


def test_symlink_checkpoint_directory_rejected(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    metadata = make_bundle(real)

    link = tmp_path / "linked"
    link.symlink_to(real, target_is_directory=True)

    with pytest.raises(ValueError):
        verify(link, metadata)


def test_manifest_digest_mismatch_rejected(tmp_path):
    metadata = make_bundle(tmp_path)
    bad = deepcopy(metadata)
    bad["state_files"][ENCODER_NAME] = "0" * 64

    with pytest.raises(ValueError, match="SHA-256"):
        verify(tmp_path, bad)


def test_invalid_manifest_rejected_before_file_reads(tmp_path):
    metadata = make_bundle(tmp_path)
    metadata["dataset_id"] = "unsupported"

    with pytest.raises(ValueError):
        verify(tmp_path, metadata)


def test_non_directory_rejected(tmp_path):
    metadata = make_manifest()
    missing = tmp_path / "missing"

    with pytest.raises(ValueError):
        verify(missing, metadata)


def test_oversized_file_rejected(tmp_path, monkeypatch):
    metadata = make_bundle(tmp_path)

    monkeypatch.setattr(
        artifact,
        "MAX_STATE_FILE_BYTES",
        8,
    )

    with pytest.raises(ValueError):
        verify(tmp_path, metadata)


def test_returned_bytes_match_verified_hashes(tmp_path):
    metadata = make_bundle(tmp_path)
    verified = verify(tmp_path, metadata)

    for name, raw in verified.items():
        assert hashlib.sha256(raw).hexdigest() == (
            metadata["state_files"][name]
        )


def test_unexpected_non_state_file_not_claimed_verified(tmp_path):
    metadata = make_bundle(tmp_path)
    (tmp_path / "unrelated.txt").write_text("unverified")

    verified = verify(tmp_path, metadata)

    assert set(verified) == {
        ENCODER_NAME,
        DECODER_NAME,
    }
    assert "unrelated.txt" not in verified
