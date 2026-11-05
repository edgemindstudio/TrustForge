from copy import deepcopy

import numpy as np
import pytest

from trustforge.models.paper01_revision_2026 import ar_maf_checkpoint_bundle as bundle
from trustforge.models.paper01_revision_2026 import ar_maf_checkpoint_manifest as manifest
from trustforge.models.paper01_revision_2026.ar_maf_checkpoint_identity import (
    canonical_json_bytes,
)
from trustforge.models.paper01_revision_2026.ar_maf_checkpoint_npz import (
    encode_network_state,
)


@pytest.fixture
def fixture(tmp_path, monkeypatch):
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
        source_commit="a" * 40,
        source_files={"src/model.py": "b" * 64},
    )
    root = tmp_path / "checkpoint"
    root.mkdir()
    (root / "manifest.json").write_bytes(
        canonical_json_bytes(document) + b"\n"
    )
    (root / "best_state.npz").write_bytes(raw)
    (root / "last_state.npz").write_bytes(raw)
    requested = {
        "recipe_raw": b"fixture",
        "expected_identity": deepcopy(identity),
        "expected_source_commit": "a" * 40,
        "expected_source_files": {"src/model.py": "b" * 64},
    }
    return root, requested, document, raw


def test_complete_bundle_returns_verified_bytes(fixture):
    root, requested, document, raw = fixture
    result = bundle.read_checkpoint_bundle(root, **requested)
    assert result["manifest"] == document
    assert result["best_raw"] == raw
    assert result["last_raw"] == raw


@pytest.mark.parametrize("member", [
    "manifest.json", "best_state.npz", "last_state.npz",
])
def test_missing_member_rejected(fixture, member):
    root, requested, _, _ = fixture
    (root / member).unlink()
    with pytest.raises(ValueError, match="inventory"):
        bundle.read_checkpoint_bundle(root, **requested)


def test_unexpected_member_rejected(fixture):
    root, requested, _, _ = fixture
    (root / "unexpected").write_bytes(b"x")
    with pytest.raises(ValueError, match="inventory"):
        bundle.read_checkpoint_bundle(root, **requested)


@pytest.mark.parametrize("field", ["scientific", "commit", "source_files"])
def test_requested_identity_mismatch_rejected(fixture, field):
    root, requested, _, _ = fixture
    if field == "scientific":
        requested["expected_identity"]["experimental_seed"] = 43
    elif field == "commit":
        requested["expected_source_commit"] = "c" * 40
    else:
        requested["expected_source_files"]["src/model.py"] = "c" * 64
    with pytest.raises(ValueError, match="identity mismatch"):
        bundle.read_checkpoint_bundle(root, **requested)


@pytest.mark.parametrize("member", ["best_state.npz", "last_state.npz"])
def test_changed_valid_archive_rejected(fixture, member):
    root, requested, document, _ = fixture
    changed = encode_network_state(
        role="model", specification=document["tensor_specification"],
        state={"model_0000": np.zeros(2, dtype=np.float32)},
    )
    (root / member).write_bytes(changed)
    with pytest.raises(ValueError):
        bundle.read_checkpoint_bundle(root, **requested)


def test_tampered_derived_manifest_field_rejected(fixture):
    root, requested, document, _ = fixture
    document["best_epoch_index"] = 0
    (root / "manifest.json").write_bytes(
        canonical_json_bytes(document) + b"\n"
    )
    with pytest.raises(ValueError, match="verified artifact evidence"):
        bundle.read_checkpoint_bundle(root, **requested)


def test_noncanonical_manifest_rejected(fixture):
    root, requested, _, _ = fixture
    path = root / "manifest.json"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="noncanonical"):
        bundle.read_checkpoint_bundle(root, **requested)


@pytest.mark.parametrize("member", [
    "manifest.json", "best_state.npz", "last_state.npz",
])
def test_symlink_member_rejected(fixture, member):
    root, requested, _, _ = fixture
    original = root / member
    external = root.parent / ("external_" + member)
    original.rename(external)
    original.symlink_to(external)
    with pytest.raises(ValueError):
        bundle.read_checkpoint_bundle(root, **requested)


def test_symlink_directory_rejected(fixture):
    root, requested, _, _ = fixture
    link = root.parent / "linked_checkpoint"
    link.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError):
        bundle.read_checkpoint_bundle(link, **requested)


def test_manifest_size_limit_enforced(fixture, monkeypatch):
    root, requested, _, _ = fixture
    monkeypatch.setattr(bundle, "MAX_MANIFEST_BYTES", 1)
    with pytest.raises(ValueError):
        bundle.read_checkpoint_bundle(root, **requested)
