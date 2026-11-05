from copy import deepcopy
import hashlib

import numpy as np
import pytest

from trustforge.models.paper01_revision_2026 import ar_maf_checkpoint_manifest as manifest
from trustforge.models.paper01_revision_2026.ar_maf_checkpoint_npz import (
    encode_network_state,
)


@pytest.fixture
def inputs(monkeypatch):
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

    def archive(value):
        return encode_network_state(
            role="model", specification=specification,
            state={"model_0000": np.full(2, value, dtype=np.float32)},
        )

    history = [{
        "epoch_index": epoch,
        "train_nll": 3.0 - epoch,
        "validation_nll": 2.0 - epoch,
        "training_examples": 65,
        "validation_examples": 3,
        "training_batches": 2,
        "validation_batches": 1,
        "completed_training_steps": 2 * (epoch + 1),
    } for epoch in range(2)]
    return {
        "identity": deepcopy(identity),
        "recipe_raw": b"fixture",
        "history": history,
        "specification": specification,
        "best_raw": archive(2),
        "last_raw": archive(2),
        "source_commit": "a" * 40,
        "source_files": {"src/model.py": "b" * 64},
    }, archive


def test_manifest_binds_selection_artifacts_and_independent_records(inputs):
    values, _ = inputs
    result = manifest.prepare_checkpoint_manifest(**values)
    assert result["best_epoch_index"] == 1
    assert result["completed_training_steps"] == 4
    assert result["artifacts"]["best"]["epoch_index"] == 1
    assert result["artifacts"]["last"]["filename"] == "last_state.npz"
    assert result["artifacts"]["best"]["sha256"] == hashlib.sha256(
        values["best_raw"]
    ).hexdigest()
    values["history"][0]["train_nll"] = 99
    values["specification"][0]["shape"][0] = 99
    assert result["history"][0]["train_nll"] == 3.0
    assert result["tensor_specification"][0]["shape"] == [2]


@pytest.mark.parametrize("losses,selected", [
    ((1.0, 1.0), 0),
    ((1.0, 1.0 - 5e-7), 1),
    ((1.0, 2.0), 0),
])
def test_strict_minimum_and_earliest_tie(inputs, losses, selected):
    values, archive = inputs
    for row, loss in zip(values["history"], losses):
        row["validation_nll"] = loss
    values["best_raw"] = archive(selected + 1)
    assert manifest.prepare_checkpoint_manifest(
        **values
    )["best_epoch_index"] == selected


@pytest.mark.parametrize("defect", [
    "missing_epoch", "extra_field", "epoch_order", "nan_loss",
    "bool_loss", "float_count", "wrong_batches", "wrong_steps",
    "changed_split", "zero_validation",
])
def test_invalid_history_rejected(inputs, defect):
    values, _ = inputs
    rows = values["history"]
    if defect == "missing_epoch":
        rows.pop()
    elif defect == "extra_field":
        rows[0]["unexpected"] = 1
    elif defect == "epoch_order":
        rows[0]["epoch_index"] = 1
    elif defect == "nan_loss":
        rows[0]["validation_nll"] = float("nan")
    elif defect == "bool_loss":
        rows[0]["train_nll"] = True
    elif defect == "float_count":
        rows[0]["training_examples"] = 65.0
    elif defect == "wrong_batches":
        rows[0]["training_batches"] = 1
    elif defect == "wrong_steps":
        rows[1]["completed_training_steps"] = 5
    elif defect == "changed_split":
        rows[1]["training_examples"] = 66
    else:
        rows[0]["validation_examples"] = 0
    with pytest.raises((ValueError, TypeError)):
        manifest.prepare_checkpoint_manifest(**values)


def test_same_epoch_best_last_disagreement_rejected(inputs):
    values, archive = inputs
    values["best_raw"] = archive(99)
    with pytest.raises(ValueError, match="best and last disagree"):
        manifest.prepare_checkpoint_manifest(**values)


@pytest.mark.parametrize("role", ["best_raw", "last_raw"])
def test_corrupt_archive_rejected(inputs, role):
    values, _ = inputs
    values[role] = values[role][:-20]
    with pytest.raises(ValueError):
        manifest.prepare_checkpoint_manifest(**values)


def test_modified_identity_rejected(inputs):
    values, _ = inputs
    values["identity"]["experimental_seed"] = 43
    with pytest.raises(ValueError, match="rebuilt contract"):
        manifest.prepare_checkpoint_manifest(**values)


@pytest.mark.parametrize("name", [
    "/absolute.py", "../escape.py", "src/../model.py",
    "src//model.py", "src\\model.py", "",
])
def test_invalid_source_path_rejected(inputs, name):
    values, _ = inputs
    values["source_files"] = {name: "b" * 64}
    with pytest.raises(ValueError):
        manifest.prepare_checkpoint_manifest(**values)


@pytest.mark.parametrize("field,value", [
    ("source_commit", "invalid"),
    ("source_files", {}),
    ("source_files", {"src/model.py": "invalid"}),
])
def test_invalid_source_identity_rejected(inputs, field, value):
    values, _ = inputs
    values[field] = value
    with pytest.raises(ValueError):
        manifest.prepare_checkpoint_manifest(**values)
