"""Checkpoint v3 bundle-verification regression tests."""

from hashlib import sha256

import pytest

from trustforge.provenance import canonical_json_bytes
from trustforge.models.paper01_revision_2026 import (
    gan_checkpoint_persistence as v2_persistence,
)
from trustforge.models.paper01_revision_2026.gan_checkpoint_v3_bundle import (
    verify_gan_checkpoint_v3_bundle,
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


def make_bundle(tmp_path):
    evidence = {
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

    provenance_bytes = canonical_json_bytes(evidence)
    (tmp_path / "gan_dataset_provenance.json").write_bytes(
        provenance_bytes
    )

    state = {
        "model_state.ckpt.index": b"fixture-index",
        "model_state.ckpt.data-00000-of-00001": b"fixture-shard",
    }

    for name, raw in state.items():
        (tmp_path / name).write_bytes(raw)

    metadata = {
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
            name: sha256(raw).hexdigest()
            for name, raw in state.items()
        },
        "exposure_evidence_file": "gan_exposure_evidence.json",
        "exposure_evidence_sha256": "c" * 64,
        "dataset_provenance_file": "gan_dataset_provenance.json",
        "dataset_provenance_sha256":
            sha256(provenance_bytes).hexdigest(),
    }

    return metadata, evidence


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    metadata, evidence = make_bundle(tmp_path)

    # Only exposure validation is replaced by a fixture.
    # Actual state-file and provenance hashes are still checked.
    monkeypatch.setattr(
        v2_persistence,
        "verify_gan_checkpoint_evidence",
        lambda directory, metadata: {"fixture": True},
    )

    return tmp_path, metadata, evidence


def test_valid_bundle(prepared):
    root, metadata, evidence = prepared
    result = verify_gan_checkpoint_v3_bundle(root, metadata)
    assert result["dataset_provenance"] == evidence
    assert result["exposure_evidence"] == {"fixture": True}


def test_corrupted_state_rejected(prepared):
    root, metadata, _ = prepared
    with (root / "model_state.ckpt.index").open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="integrity failure"):
        verify_gan_checkpoint_v3_bundle(root, metadata)


def test_missing_state_rejected(prepared):
    root, metadata, _ = prepared
    (root / "model_state.ckpt.index").unlink()
    with pytest.raises(ValueError):
        verify_gan_checkpoint_v3_bundle(root, metadata)


def test_extra_state_rejected(prepared):
    root, metadata, _ = prepared
    (root / "model_state.ckpt.extra").write_bytes(b"extra")
    with pytest.raises(ValueError, match="inventory mismatch"):
        verify_gan_checkpoint_v3_bundle(root, metadata)


def test_corrupted_provenance_rejected(prepared):
    root, metadata, _ = prepared
    path = root / "gan_dataset_provenance.json"
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_gan_checkpoint_v3_bundle(root, metadata)


def test_missing_provenance_rejected(prepared):
    root, metadata, _ = prepared
    (root / "gan_dataset_provenance.json").unlink()
    with pytest.raises(ValueError, match="missing or unsafe"):
        verify_gan_checkpoint_v3_bundle(root, metadata)


def test_publication_lock_rejected(prepared):
    root, metadata, _ = prepared
    (root / ".gan_checkpoint_publication.lock").mkdir()
    with pytest.raises(RuntimeError, match="incomplete"):
        verify_gan_checkpoint_v3_bundle(root, metadata)


def test_v2_schema_rejected(prepared):
    root, metadata, _ = prepared
    metadata["schema_version"] = "2.0"
    with pytest.raises(ValueError, match="schema version"):
        verify_gan_checkpoint_v3_bundle(root, metadata)


def test_wrong_dataset_rejected(prepared):
    root, metadata, _ = prepared
    metadata["dataset_id"] = "cicmaldroid2020"
    metadata["class_count"] = 5
    metadata["class_mapping"] = [str(i) for i in range(5)]
    with pytest.raises(ValueError, match="identity mismatch"):
        verify_gan_checkpoint_v3_bundle(root, metadata)
