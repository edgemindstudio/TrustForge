"""Governed VAE checkpoint loader pre-restoration rejection tests.

All checkpoint files are synthetic and disposable.
No real datasets, training, or checkpoint publication.
"""

import hashlib

import optree
import keras
import tensorflow as tf
import pytest

from trustforge.provenance import canonical_json_bytes

from trustforge.models.paper01_revision_2026 import (
    variational_autoencoder_checkpoint_loader as loader,
    variational_autoencoder_checkpoint_compatibility as compatibility,
)

from tests import (
    test_variational_autoencoder_checkpoint_artifact_revision_2026
    as artifact_fixtures,
)

from tests import (
    test_variational_autoencoder_checkpoint_provenance_revision_2026
    as provenance_fixtures,
)


DATASET_ID = "ustc_tfc2016"
SEED = 42
CLASS_MAPPING = [f"class_{index}" for index in range(9)]


@pytest.fixture
def checkpoint_bundle(tmp_path, monkeypatch):
    metadata = artifact_fixtures.make_manifest()
    metadata["architecture_version"] = (
        compatibility.CANDIDATE_ARCHITECTURE_VERSION
    )

    provenance = provenance_fixtures.evidence_for(DATASET_ID)
    provenance_bytes = canonical_json_bytes(provenance)

    metadata["dataset_provenance_sha256"] = (
        hashlib.sha256(provenance_bytes).hexdigest()
    )

    (tmp_path / loader.MANIFEST_FILENAME).write_bytes(
        canonical_json_bytes(metadata) + b"\n"
    )

    (tmp_path / loader.VAE_PROVENANCE_FILENAME).write_bytes(
        provenance_bytes
    )

    (tmp_path / artifact_fixtures.ENCODER_NAME).write_bytes(
        artifact_fixtures.ENCODER_BYTES
    )

    (tmp_path / artifact_fixtures.DECODER_NAME).write_bytes(
        artifact_fixtures.DECODER_BYTES
    )

    monkeypatch.setenv("SLURM_JOB_ID", "synthetic-test-only")

    def forbidden_restoration(**kwargs):
        raise AssertionError(
            "MODEL RESTORATION MUST NOT BE REACHED"
        )

    monkeypatch.setattr(
        loader,
        "restore_vae_state_into_isolated_models",
        forbidden_restoration,
    )

    return tmp_path


def invoke(root, *, dataset_id=DATASET_ID,
           seed=SEED, class_mapping=None):
    if class_mapping is None:
        class_mapping = CLASS_MAPPING

    return loader.load_governed_vae_checkpoint(
        root,
        requested_dataset_id=dataset_id,
        requested_experimental_seed=seed,
        requested_class_mapping=class_mapping,
    )


def test_dataset_identity_mismatch(checkpoint_bundle):
    with pytest.raises(ValueError, match="dataset identity mismatch"):
        invoke(checkpoint_bundle, dataset_id="malex")


def test_seed_identity_mismatch(checkpoint_bundle):
    with pytest.raises(ValueError, match="experimental seed mismatch"):
        invoke(checkpoint_bundle, seed=43)


def test_ordered_class_mapping_mismatch(checkpoint_bundle):
    reversed_mapping = list(reversed(CLASS_MAPPING))

    with pytest.raises(ValueError, match="ordered class mapping mismatch"):
        invoke(checkpoint_bundle, class_mapping=reversed_mapping)


def test_noncanonical_manifest_rejected(checkpoint_bundle):
    path = checkpoint_bundle / loader.MANIFEST_FILENAME
    raw = path.read_bytes()
    assert raw.endswith(b"\n")

    path.write_bytes(raw[:-1])

    with pytest.raises(ValueError, match="noncanonical"):
        invoke(checkpoint_bundle)


def test_noncanonical_provenance_rejected(checkpoint_bundle):
    path = checkpoint_bundle / loader.VAE_PROVENANCE_FILENAME
    path.write_bytes(path.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="noncanonical"):
        invoke(checkpoint_bundle)


def test_provenance_digest_mismatch(checkpoint_bundle):
    path = checkpoint_bundle / loader.VAE_PROVENANCE_FILENAME
    raw = path.read_bytes()

    path.write_bytes(raw.replace(
        b"ustc_tfc2016",
        b"cicmaldroid2020",
        1,
    ))

    with pytest.raises(ValueError, match="provenance SHA-256 mismatch"):
        invoke(checkpoint_bundle)


@pytest.mark.parametrize("filename", [
    artifact_fixtures.ENCODER_NAME,
    artifact_fixtures.DECODER_NAME,
])
def test_corrupted_archive_digest_rejected(checkpoint_bundle, filename):
    path = checkpoint_bundle / filename
    path.write_bytes(path.read_bytes() + b"corruption")

    with pytest.raises((ValueError, RuntimeError)):
        invoke(checkpoint_bundle)


def test_publication_lock_rejected(checkpoint_bundle):
    (checkpoint_bundle / loader.PUBLICATION_LOCK).write_text(
        "incomplete",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="publication incomplete"):
        invoke(checkpoint_bundle)


def test_missing_slurm_context_rejected(checkpoint_bundle, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)

    with pytest.raises(RuntimeError, match="requires Slurm"):
        invoke(checkpoint_bundle)


def test_unsupported_architecture_rejected(checkpoint_bundle):
    path = checkpoint_bundle / loader.MANIFEST_FILENAME

    import json
    metadata = json.loads(path.read_bytes())
    metadata["architecture_version"] = "unsupported_architecture"

    path.write_bytes(
        canonical_json_bytes(metadata) + b"\n"
    )

    with pytest.raises(ValueError):
        invoke(checkpoint_bundle)
