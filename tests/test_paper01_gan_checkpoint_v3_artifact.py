"""Artifact-backed checkpoint v3 provenance verifier tests."""

from copy import deepcopy
from hashlib import sha256
import json

import pytest

from trustforge.provenance import canonical_json_bytes
from trustforge.models.paper01_revision_2026.gan_checkpoint_v3_artifact import (
    MAX_PROVENANCE_BYTES,
    verify_gan_checkpoint_v3_provenance_artifact,
)
from trustforge.models.paper01_revision_2026.gan_checkpoint_v3_provenance import (
    SCHEMA_TYPE,
    SCHEMA_VERSION,
    USTC_LIMITATIONS,
    USTC_PARTITION_POLICY,
)
from trustforge.models.paper01_revision_2026.gan_dataset_provenance import (
    USTC_NCHW_SOURCE_SHA256,
)


def valid_evidence():
    return {
        "schema_type": SCHEMA_TYPE,
        "schema_version": SCHEMA_VERSION,
        "dataset_id": "ustc_tfc2016",
        "source_contract": "ustc_nchw_revision_2026",
        "source_fingerprints": dict(USTC_NCHW_SOURCE_SHA256),
        "partition_policy": dict(USTC_PARTITION_POLICY),
        "verification_method":
            "sha256_full_file_stream_on_compute_node",
        "verification_scope": "four_original_nchw_numpy_files",
        "verification_limitations": list(USTC_LIMITATIONS),
    }


def valid_metadata(digest):
    return {
        "schema_version": "3.0",
        "schema_type": "paper01_revision2026_conditional_gan",
        "paper_revision": "paper01_revision_2026",
        "model_family": "conditional_gan",
        "dataset_id": "ustc_tfc2016",
        "experimental_seed": 42,
        "class_mapping": [str(i) for i in range(9)],
        "class_count": 9,
        "generator_iteration": 10,
        "discriminator_updates": 20,
        "planned_generator_iterations": 10,
        "checkpoint_selection": "terminal_governed_iteration",
        "image_shape": [40, 40, 3],
        "latent_dim": 100,
        "learning_rate": 0.0002,
        "beta_1": 0.5,
        "class_sampling_alpha": 0.85,
        "batch_size": 256,
        "state_files": {
            "model_state.ckpt.index": "a" * 64,
            "model_state.ckpt.data-00000-of-00001": "b" * 64,
        },
        "exposure_evidence_file": "gan_exposure_evidence.json",
        "exposure_evidence_sha256": "c" * 64,
        "dataset_provenance_file": "gan_dataset_provenance.json",
        "dataset_provenance_sha256": digest,
    }


def fixture(tmp_path):
    evidence = valid_evidence()
    raw = canonical_json_bytes(evidence)
    path = tmp_path / "gan_dataset_provenance.json"
    path.write_bytes(raw)
    return path, evidence, valid_metadata(sha256(raw).hexdigest())


def test_valid_artifact(tmp_path):
    path, evidence, metadata = fixture(tmp_path)
    assert verify_gan_checkpoint_v3_provenance_artifact(
        tmp_path, metadata
    ) == evidence


def test_input_metadata_not_modified(tmp_path):
    path, evidence, metadata = fixture(tmp_path)
    before = deepcopy(metadata)
    verify_gan_checkpoint_v3_provenance_artifact(tmp_path, metadata)
    assert metadata == before


def test_missing_file_rejected(tmp_path):
    path, evidence, metadata = fixture(tmp_path)
    path.unlink()
    with pytest.raises(ValueError, match="missing or unsafe"):
        verify_gan_checkpoint_v3_provenance_artifact(
            tmp_path, metadata
        )


def test_symlink_rejected(tmp_path):
    path, evidence, metadata = fixture(tmp_path)
    target = tmp_path / "target.json"
    path.rename(target)
    path.symlink_to(target)
    with pytest.raises(ValueError, match="missing or unsafe"):
        verify_gan_checkpoint_v3_provenance_artifact(
            tmp_path, metadata
        )


def test_modified_bytes_rejected(tmp_path):
    path, evidence, metadata = fixture(tmp_path)
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_gan_checkpoint_v3_provenance_artifact(
            tmp_path, metadata
        )


def test_substituted_digest_rejected(tmp_path):
    path, evidence, metadata = fixture(tmp_path)
    metadata["dataset_provenance_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_gan_checkpoint_v3_provenance_artifact(
            tmp_path, metadata
        )


def test_wrong_dataset_identity_rejected(tmp_path):
    path, evidence, metadata = fixture(tmp_path)
    metadata["dataset_id"] = "cicmaldroid2020"
    metadata["class_mapping"] = [str(i) for i in range(5)]
    metadata["class_count"] = 5
    with pytest.raises(ValueError, match="identity mismatch"):
        verify_gan_checkpoint_v3_provenance_artifact(
            tmp_path, metadata
        )


def test_noncanonical_json_rejected(tmp_path):
    path, evidence, metadata = fixture(tmp_path)
    raw = json.dumps(evidence, indent=2).encode("utf-8")
    path.write_bytes(raw)
    metadata["dataset_provenance_sha256"] = sha256(raw).hexdigest()
    with pytest.raises(ValueError, match="not canonical"):
        verify_gan_checkpoint_v3_provenance_artifact(
            tmp_path, metadata
        )


def test_fingerprint_tampering_rejected_even_with_updated_digest(tmp_path):
    path, evidence, metadata = fixture(tmp_path)
    evidence["source_fingerprints"]["train_data.npy"] = "0" * 64
    raw = canonical_json_bytes(evidence)
    path.write_bytes(raw)
    metadata["dataset_provenance_sha256"] = sha256(raw).hexdigest()
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        verify_gan_checkpoint_v3_provenance_artifact(
            tmp_path, metadata
        )


def test_removed_limitations_rejected_with_updated_digest(tmp_path):
    path, evidence, metadata = fixture(tmp_path)
    evidence["verification_limitations"] = []
    raw = canonical_json_bytes(evidence)
    path.write_bytes(raw)
    metadata["dataset_provenance_sha256"] = sha256(raw).hexdigest()
    with pytest.raises(ValueError, match="limitations mismatch"):
        verify_gan_checkpoint_v3_provenance_artifact(
            tmp_path, metadata
        )


def test_empty_artifact_rejected(tmp_path):
    path, evidence, metadata = fixture(tmp_path)
    path.write_bytes(b"")
    with pytest.raises(ValueError, match="size"):
        verify_gan_checkpoint_v3_provenance_artifact(
            tmp_path, metadata
        )


def test_oversized_artifact_rejected(tmp_path):
    path, evidence, metadata = fixture(tmp_path)
    path.write_bytes(b"x" * (MAX_PROVENANCE_BYTES + 1))
    with pytest.raises(ValueError, match="size"):
        verify_gan_checkpoint_v3_provenance_artifact(
            tmp_path, metadata
        )


def test_v2_manifest_rejected(tmp_path):
    path, evidence, metadata = fixture(tmp_path)
    metadata["schema_version"] = "2.0"
    with pytest.raises(ValueError, match="schema version"):
        verify_gan_checkpoint_v3_provenance_artifact(
            tmp_path, metadata
        )
