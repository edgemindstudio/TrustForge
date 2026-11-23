"""Fixture tests for durable Diffusion checkpoint bundles.

No TensorFlow training, inference, synthesis or real-data access occurs.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from trustforge.models.paper01_revision_2026.diffusion_checkpoint_bundle import (
    DiffusionBundleError,
    load_diffusion_checkpoint_bundle,
    publish_diffusion_checkpoint_bundle,
)
from trustforge.models.paper01_revision_2026.diffusion_checkpoint_state import (
    prepare_diffusion_checkpoint_states,
    restore_diffusion_checkpoint_state,
)
from trustforge.models.paper01_revision_2026.synthesis_linkage import (
    SynthesisRequest,
    load_synthesis_recipe,
    restore_diffusion_model_for_synthesis,
)


REPO = Path(__file__).resolve().parents[1]
DATASET = "malex"
MAPPING = ("benign", "malware")
SOURCE = "a" * 40
PROVENANCE = {"synthetic": True}


class DummyModel:
    def __init__(self):
        self.values = [np.zeros((2, 3), dtype=np.float32)]

    def get_weights(self):
        return [value.copy() for value in self.values]

    def set_weights(self, weights):
        self.values = [value.copy() for value in weights]


def governed_fixture():
    configuration = {
        "image_shape": [40, 40, 3], "depth": 2,
        "base_filters": 64, "time_embedding_dim": 128,
        "timesteps": 1000, "schedule": "linear",
        "beta_start": 1e-4, "beta_end": 2e-2,
        "epochs": 2, "batch_size": 64,
        "learning_rate": 2e-4, "beta_1": 0.9,
    }
    history = (
        {
            "epoch": 0, "training_examples": 3,
            "validation_examples": 2, "completed_updates": 1,
            "train_mse": 1.2, "validation_mse": 0.7,
        },
        {
            "epoch": 1, "training_examples": 3,
            "validation_examples": 2, "completed_updates": 2,
            "train_mse": 1.0, "validation_mse": 0.8,
        },
    )
    return {
        "dataset_id": DATASET,
        "experimental_seed": 42,
        "class_count": 2,
        "dataset_provenance": PROVENANCE,
        "scientific_configuration": configuration,
        "checkpoint_status": "NOT_PREPARED",
        "publication_status": "NOT_PUBLISHED",
        "training_result": {
            "history": history, "best_epoch": 0,
            "best_validation_mse": 0.7,
            "best_weights": (np.full((2, 3), 1, dtype=np.float32),),
            "last_weights": (np.full((2, 3), 2, dtype=np.float32),),
            "completed_epochs": 2, "completed_updates": 2,
            "train_size": 3, "validation_size": 2,
        },
    }


def prepared():
    return prepare_diffusion_checkpoint_states(
        governed_result=governed_fixture(), class_mapping=MAPPING,
    )


def publish(tmp_path, states=None):
    destination = tmp_path / "bundle"
    result = publish_diffusion_checkpoint_bundle(
        prepared=prepared() if states is None else states,
        destination=destination,
        source_commit=SOURCE,
        dataset_provenance=PROVENANCE,
    )
    return destination, result


def load(directory, manifest_sha, **overrides):
    arguments = {
        "expected_manifest_sha256": manifest_sha,
        "requested_dataset_id": DATASET,
        "requested_experimental_seed": 42,
        "requested_class_mapping": MAPPING,
        "expected_source_commit": SOURCE,
    }
    arguments.update(overrides)
    return load_diffusion_checkpoint_bundle(directory, **arguments)


def rewrite_manifest(directory, manifest):
    raw = (
        json.dumps(
            manifest, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode()
        + b"\n"
    )
    (directory / "manifest.json").write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def test_best_and_last_round_trip(tmp_path):
    directory, result = publish(tmp_path)
    loaded = load(directory, result["manifest_sha256"])
    assert set(loaded) == {"best", "last"}
    assert loaded["best"]["selected_epoch"] == 0
    assert loaded["last"]["selected_epoch"] == 1
    np.testing.assert_array_equal(
        loaded["best"]["weights"][0],
        np.full((2, 3), 1, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        loaded["last"]["weights"][0],
        np.full((2, 3), 2, dtype=np.float32),
    )
    assert loaded["best"]["state_sha256"] != loaded["last"]["state_sha256"]


def test_loaded_best_restores_into_compatible_model(tmp_path):
    directory, result = publish(tmp_path)
    loaded = load(
        directory, result["manifest_sha256"], requested_role="best",
    )
    model = restore_diffusion_checkpoint_state(
        prepared=loaded["best"], model_factory=DummyModel,
        requested_dataset_id=DATASET, requested_experimental_seed=42,
        requested_class_mapping=MAPPING, requested_role="best",
    )
    np.testing.assert_array_equal(
        model.get_weights()[0],
        np.full((2, 3), 1, dtype=np.float32),
    )


def test_loaded_best_is_accepted_by_synthesis_restoration(tmp_path):
    directory, result = publish(tmp_path)
    loaded = load(directory, result["manifest_sha256"])["best"]
    recipe = load_synthesis_recipe(REPO)
    request = SynthesisRequest(
        dataset_id=DATASET, family="diffusion", experimental_seed=42,
        class_id=0, sample_indices=(0,), class_mapping=MAPPING,
        scope="fixture",
    )
    identity = {
        "dataset_id": DATASET, "experimental_seed": 42,
        "class_mapping": list(MAPPING), "checkpoint_role": "best",
        "checkpoint_sha256": loaded["state_sha256"],
    }
    model = restore_diffusion_model_for_synthesis(
        prepared=loaded, model_factory=DummyModel, request=request,
        recipe=recipe, checkpoint_identity=identity,
    )
    np.testing.assert_array_equal(
        model.get_weights()[0],
        np.full((2, 3), 1, dtype=np.float32),
    )


def test_roles_are_not_interchanged(tmp_path):
    directory, result = publish(tmp_path)
    loaded = load(directory, result["manifest_sha256"])
    assert not np.array_equal(
        loaded["best"]["weights"][0], loaded["last"]["weights"][0],
    )
    assert loaded["best"]["checkpoint_role"] == "best"
    assert loaded["last"]["checkpoint_role"] == "last"


@pytest.mark.parametrize("overrides", [
    {"requested_dataset_id": "cicmaldroid2020"},
    {"requested_experimental_seed": 43},
    {"requested_class_mapping": ("malware", "benign")},
    {"requested_role": "other"},
])
def test_wrong_identity_rejected(tmp_path, overrides):
    directory, result = publish(tmp_path)
    with pytest.raises(DiffusionBundleError):
        load(directory, result["manifest_sha256"], **overrides)


def test_wrong_manifest_digest_rejected(tmp_path):
    directory, _ = publish(tmp_path)
    with pytest.raises(DiffusionBundleError, match="manifest digest"):
        load(directory, "0" * 64)


def test_wrong_source_commit_rejected(tmp_path):
    directory, result = publish(tmp_path)
    with pytest.raises(DiffusionBundleError, match="source commit"):
        load(
            directory, result["manifest_sha256"],
            expected_source_commit="b" * 40,
        )


def test_manifest_configuration_tamper_rejected(tmp_path):
    directory, result = publish(tmp_path)
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest["scientific_configuration"]["epochs"] = 3
    digest = rewrite_manifest(directory, manifest)
    with pytest.raises(DiffusionBundleError, match="identity mismatch"):
        load(directory, digest)


def test_corrupted_tensor_rejected(tmp_path):
    directory, result = publish(tmp_path)
    path = directory / "best" / "w000.npy"
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 0xFF
    path.write_bytes(bytes(raw))
    with pytest.raises(DiffusionBundleError, match="tensor integrity"):
        load(directory, result["manifest_sha256"])


def test_missing_tensor_rejected(tmp_path):
    directory, result = publish(tmp_path)
    (directory / "best" / "w000.npy").unlink()
    with pytest.raises(DiffusionBundleError):
        load(directory, result["manifest_sha256"])


def test_state_digest_contradiction_rejected(tmp_path):
    directory, result = publish(tmp_path)
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest["states"]["best"]["state_sha256"] = "0" * 64
    digest = rewrite_manifest(directory, manifest)
    with pytest.raises(DiffusionBundleError, match="digest mismatch"):
        load(directory, digest)


def test_existing_destination_not_overwritten(tmp_path):
    directory, _ = publish(tmp_path)
    marker = directory / "manifest.json"
    before = marker.read_bytes()
    with pytest.raises(FileExistsError):
        publish_diffusion_checkpoint_bundle(
            prepared=prepared(), destination=directory,
            source_commit=SOURCE, dataset_provenance=PROVENANCE,
        )
    assert marker.read_bytes() == before


def test_partial_publication_marker_rejected(tmp_path):
    directory = tmp_path / "partial"
    directory.mkdir()
    (directory / ".bundle.pending").write_bytes(b"")
    with pytest.raises(DiffusionBundleError, match="incomplete"):
        load(directory, "0" * 64)


def test_unexpected_bundle_entry_rejected(tmp_path):
    directory, result = publish(tmp_path)
    (directory / "extra.txt").write_text("unexpected")
    with pytest.raises(DiffusionBundleError, match="entries"):
        load(directory, result["manifest_sha256"])


def test_tampered_prepared_state_rejected(tmp_path):
    states = prepared()
    states["best"]["weights"][0][0, 0] = 9.0
    with pytest.raises(DiffusionBundleError, match="digest mismatch"):
        publish(tmp_path, states=copy.deepcopy(states))


def test_invalid_source_commit_rejected(tmp_path):
    with pytest.raises(DiffusionBundleError, match="source commit"):
        publish_diffusion_checkpoint_bundle(
            prepared=prepared(), destination=tmp_path / "bundle",
            source_commit="not-a-commit", dataset_provenance=PROVENANCE,
        )
