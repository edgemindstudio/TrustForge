"""Read-only committed VAE generation verifier regressions.

Constructs synthetic generation fixtures in disposable test directories.
Does not execute the governed publisher, train models, or access datasets.
"""

import json

import optree
import keras
import pytest
import tensorflow as tf

from trustforge.provenance import canonical_json_bytes

from trustforge.models.paper01_revision_2026 import (
    variational_autoencoder_checkpoint_committed_generation as verifier,
    variational_autoencoder_checkpoint_commit_record as commit,
)

from tests.test_variational_autoencoder_checkpoint_loader_roundtrip_revision_2026 import (
    synthetic_bundle,
    DATASET_ID,
    SEED,
    CLASS_MAPPING,
)


GENERATION_ID = "0123456789abcdef0123456789abcdef"
OTHER_ID = "fedcba9876543210fedcba9876543210"


@pytest.fixture
def committed_fixture(synthetic_bundle, tmp_path):
    root = tmp_path / "publication"
    generations = root / "generations"
    commits = root / "commits"

    generations.mkdir(parents=True)
    commits.mkdir()

    generation = generations / GENERATION_ID
    generation.mkdir()

    source_directory = synthetic_bundle["directory"]

    for filename in (
        "vae_checkpoint_manifest.json",
        "vae_dataset_provenance.json",
        "vae_encoder_state.npz",
        "vae_decoder_state.npz",
    ):
        (generation / filename).write_bytes(
            (source_directory / filename).read_bytes()
        )

    manifest_bytes = (
        generation / "vae_checkpoint_manifest.json"
    ).read_bytes()

    record = commit.build_vae_commit_record(
        generation_id=GENERATION_ID,
        manifest_bytes=manifest_bytes,
    )

    marker = commits / (GENERATION_ID + ".commit.json")
    marker.write_bytes(commit.encode_vae_commit_record(record))

    return {
        "root": root,
        "generation": generation,
        "marker": marker,
        "record": record,
        "metadata": synthetic_bundle["metadata"],
    }


def verify(fixture, *, generation_id=GENERATION_ID):
    return verifier.verify_committed_vae_generation(
        fixture["root"],
        generation_id=generation_id,
        requested_dataset_id=DATASET_ID,
        requested_experimental_seed=SEED,
        requested_class_mapping=CLASS_MAPPING,
    )


def test_committed_generation_verified(committed_fixture):
    result = verify(committed_fixture)

    assert result["generation_id"] == GENERATION_ID
    assert result["commit_record"] == committed_fixture["record"]
    assert result["metadata"] == committed_fixture["metadata"]
    assert set(result["verified_state_bytes"]) == {
        "vae_encoder_state.npz",
        "vae_decoder_state.npz",
    }


def test_missing_commit_marker_rejected(committed_fixture):
    committed_fixture["marker"].unlink()

    with pytest.raises(ValueError):
        verify(committed_fixture)


def test_malformed_commit_json_rejected(committed_fixture):
    committed_fixture["marker"].write_bytes(b"not-json")

    with pytest.raises(ValueError, match="commit record JSON"):
        verify(committed_fixture)


def test_noncanonical_commit_json_rejected(committed_fixture):
    record = committed_fixture["record"]

    committed_fixture["marker"].write_bytes(
        json.dumps(record, indent=2).encode("utf-8")
    )

    with pytest.raises(ValueError, match="noncanonical"):
        verify(committed_fixture)


def test_commit_generation_identity_mismatch_rejected(
    committed_fixture,
):
    record = dict(committed_fixture["record"])
    record["generation_id"] = OTHER_ID

    committed_fixture["marker"].write_bytes(
        commit.encode_vae_commit_record(record)
    )

    with pytest.raises(ValueError, match="generation identity mismatch"):
        verify(committed_fixture)


def test_altered_manifest_rejected(committed_fixture):
    path = (
        committed_fixture["generation"]
        / "vae_checkpoint_manifest.json"
    )
    path.write_bytes(path.read_bytes() + b"modified")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify(committed_fixture)


def test_symlinked_commit_marker_rejected(committed_fixture):
    marker = committed_fixture["marker"]
    saved = marker.with_name("saved_commit_record.json")

    marker.rename(saved)
    marker.symlink_to(saved.name)

    with pytest.raises(ValueError):
        verify(committed_fixture)


def test_uncommitted_other_generation_rejected(committed_fixture):
    other = (
        committed_fixture["root"]
        / "generations"
        / OTHER_ID
    )
    other.mkdir()

    with pytest.raises(ValueError):
        verify(committed_fixture, generation_id=OTHER_ID)


def test_symlinked_manifest_rejected(committed_fixture):
    manifest = (
        committed_fixture["generation"]
        / "vae_checkpoint_manifest.json"
    )
    saved = manifest.with_name("saved_manifest.json")

    manifest.rename(saved)
    manifest.symlink_to(saved.name)

    with pytest.raises(ValueError):
        verify(committed_fixture)
