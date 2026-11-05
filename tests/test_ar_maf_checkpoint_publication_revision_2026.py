from copy import deepcopy

import numpy as np
import pytest

from trustforge.models.paper01_revision_2026 import ar_maf_checkpoint_publication as publication
from trustforge.models.paper01_revision_2026 import ar_maf_checkpoint_manifest as manifest
from trustforge.models.paper01_revision_2026.ar_maf_checkpoint_bundle import (
    read_checkpoint_bundle,
)
from trustforge.models.paper01_revision_2026.ar_maf_checkpoint_identity import (
    canonical_json_bytes,
)
from trustforge.models.paper01_revision_2026.ar_maf_checkpoint_npz import (
    encode_network_state,
)


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    identity = {
        "family": "ar", "dataset_id": "malex",
        "experimental_seed": 42,
        "class_mapping": ["benign", "malware"],
        "dataset_provenance": {"fixture": True},
    }
    monkeypatch.setattr(
        manifest, "build_checkpoint_identity",
        lambda **kwargs: deepcopy(identity),
    )
    specification = [{
        "key": "model_0000", "shape": [2],
        "dtype": "float32", "trainable": True,
    }]
    raw = encode_network_state(
        role="model", specification=specification,
        state={"model_0000": np.ones(2, dtype=np.float32)},
    )
    history = [{
        "epoch_index": epoch, "train_nll": 3.0 - epoch,
        "validation_nll": 2.0 - epoch,
        "training_examples": 65, "validation_examples": 3,
        "training_batches": 2, "validation_batches": 1,
        "completed_training_steps": 2 * (epoch + 1),
    } for epoch in range(2)]
    document = manifest.prepare_checkpoint_manifest(
        identity=identity, recipe_raw=b"fixture", history=history,
        specification=specification, best_raw=raw, last_raw=raw,
        source_commit="a" * 40, source_files={"src/model.py": "b" * 64},
    )
    parent = tmp_path / "private"
    parent.mkdir(mode=0o700)
    requested = {
        "recipe_raw": b"fixture",
        "expected_identity": deepcopy(identity),
        "expected_source_commit": "a" * 40,
        "expected_source_files": {"src/model.py": "b" * 64},
    }
    prepared = {
        **requested,
        "manifest_raw": canonical_json_bytes(document) + b"\n",
        "best_raw": raw, "last_raw": raw,
    }
    return parent, prepared, requested


def test_publish_then_load_exact_bundle(inputs):
    parent, prepared, requested = inputs
    result = publication.publish_checkpoint_bundle(
        parent, "attempt01", **prepared,
    )
    loaded = read_checkpoint_bundle(parent / "attempt01", **requested)
    assert loaded["manifest"] == result["manifest"]
    assert loaded["best_raw"] == prepared["best_raw"]
    assert result["publication_status"] == "WRITTEN_VERIFIED_AND_SYNCED"
    assert (parent / "attempt01").stat().st_mode & 0o777 == 0o700


def test_existing_destination_never_overwritten(inputs):
    parent, prepared, _ = inputs
    publication.publish_checkpoint_bundle(parent, "attempt01", **prepared)
    root = parent / "attempt01"
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    with pytest.raises(FileExistsError):
        publication.publish_checkpoint_bundle(parent, "attempt01", **prepared)
    assert {p.name: p.read_bytes() for p in root.iterdir()} == before


def test_invalid_bundle_rejected_before_directory_creation(inputs):
    parent, prepared, _ = inputs
    prepared["best_raw"] = b"invalid"
    with pytest.raises(ValueError):
        publication.publish_checkpoint_bundle(parent, "attempt01", **prepared)
    assert list(parent.iterdir()) == []


@pytest.mark.parametrize("name", ["", ".", "..", "../escape", "a/b", "a\\b"])
def test_invalid_generation_name_rejected(inputs, name):
    parent, prepared, _ = inputs
    with pytest.raises(ValueError):
        publication.publish_checkpoint_bundle(parent, name, **prepared)
    assert list(parent.iterdir()) == []


def test_symlink_destination_rejected_without_changing_target(inputs):
    parent, prepared, _ = inputs
    target = parent / "target"
    target.mkdir()
    (parent / "attempt01").symlink_to(target, target_is_directory=True)
    with pytest.raises(FileExistsError):
        publication.publish_checkpoint_bundle(parent, "attempt01", **prepared)
    assert list(target.iterdir()) == []


def test_state_write_failure_retains_incomplete_attempt(inputs, monkeypatch):
    parent, prepared, requested = inputs
    original = publication.write_exclusive_vae_artifact

    def fail_second_state(fd, name, raw):
        if name == "last_state.npz":
            raise OSError("injected state failure")
        return original(fd, name, raw)

    monkeypatch.setattr(
        publication, "write_exclusive_vae_artifact", fail_second_state,
    )
    with pytest.raises(OSError, match="injected state failure"):
        publication.publish_checkpoint_bundle(parent, "attempt01", **prepared)
    root = parent / "attempt01"
    assert (root / "best_state.npz").is_file()
    assert not (root / "manifest.json").exists()
    with pytest.raises(ValueError, match="inventory"):
        read_checkpoint_bundle(root, **requested)


def test_failure_after_manifest_write_is_indeterminate(inputs, monkeypatch):
    parent, prepared, requested = inputs
    original = publication.write_exclusive_vae_artifact

    def fail_after_manifest(fd, name, raw):
        result = original(fd, name, raw)
        if name == "manifest.json":
            raise OSError("injected post-manifest failure")
        return result

    monkeypatch.setattr(
        publication, "write_exclusive_vae_artifact", fail_after_manifest,
    )
    with pytest.raises(publication.IndeterminateCheckpointPublicationError):
        publication.publish_checkpoint_bundle(parent, "attempt01", **prepared)
    # An indeterminate outcome may already contain a valid bundle.
    loaded = read_checkpoint_bundle(parent / "attempt01", **requested)
    assert loaded["best_raw"] == prepared["best_raw"]
