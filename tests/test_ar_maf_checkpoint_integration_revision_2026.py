"""Checkpoint integration using constructed contract records.

No production data is read and no real training is claimed.
All checkpoint and provenance validators remain active.
"""

from pathlib import Path

import numpy as np
import pytest

from test_variational_autoencoder_checkpoint_provenance_revision_2026 import (
    evidence_for,
)
from trustforge.models.paper01_revision_2026.ar_maf_execution_inputs import (
    DATASET_CLASSES,
)
from trustforge.models.paper01_revision_2026.ar_maf_checkpoint_identity import (
    build_checkpoint_identity, canonical_json_bytes,
)
from trustforge.models.paper01_revision_2026.ar_maf_checkpoint_manifest import (
    prepare_checkpoint_manifest,
)
from trustforge.models.paper01_revision_2026.ar_maf_checkpoint_state import (
    encode_model_snapshot, restore_model_snapshot,
)
from trustforge.models.paper01_revision_2026.ar_maf_checkpoint_publication import (
    publish_checkpoint_bundle,
)
from trustforge.models.paper01_revision_2026.ar_maf_checkpoint_bundle import (
    read_checkpoint_bundle,
)


class Variable:
    trainable = True

    def __init__(self, value):
        self.value = np.full(2, value, dtype=np.float32)

    def numpy(self):
        return self.value


class Model:
    def __init__(self, value):
        self.weights = [Variable(value)]

    def set_weights(self, values):
        assert len(values) == 1
        self.weights[0].value = np.array(values[0], copy=True)


def recipe_bytes():
    return (
        Path(__file__).resolve().parents[1]
        / "studies/paper01_benchmark/revision_2026/"
          "ar_maf_integration_recipe.json"
    ).read_bytes()


def make_identity(dataset_id, family, provenance):
    # Numeric placeholders are test mapping labels, not semantic-name claims.
    return build_checkpoint_identity(
        family=family, dataset_id=dataset_id, experimental_seed=42,
        class_mapping=[
            str(i) for i in range(DATASET_CLASSES[dataset_id])
        ],
        recipe_raw=recipe_bytes(), dataset_provenance=provenance,
    )


@pytest.mark.parametrize("dataset_id", sorted(DATASET_CLASSES))
@pytest.mark.parametrize("family", ["ar", "maf"])
def test_full_checkpoint_path_without_validator_stubs(
    tmp_path, dataset_id, family,
):
    identity = make_identity(dataset_id, family, evidence_for(dataset_id))
    model = Model(0)
    specification, best_raw = encode_model_snapshot(
        model=model, weights=[np.ones(2, dtype=np.float32)],
    )
    _, last_raw = encode_model_snapshot(
        model=model, weights=[np.full(2, 2, dtype=np.float32)],
    )
    history = [{
        "epoch_index": epoch, "train_nll": 3.0 - epoch,
        "validation_nll": 1.0 + epoch,
        "training_examples": 65, "validation_examples": 3,
        "training_batches": 2, "validation_batches": 1,
        "completed_training_steps": 2 * (epoch + 1),
    } for epoch in range(2)]
    document = prepare_checkpoint_manifest(
        identity=identity, recipe_raw=recipe_bytes(), history=history,
        specification=specification,
        best_raw=best_raw, last_raw=last_raw,
        source_commit="a" * 40,
        source_files={"fixture/model.py": "b" * 64},
    )
    requested = {
        "recipe_raw": recipe_bytes(),
        "expected_identity": identity,
        "expected_source_commit": "a" * 40,
        "expected_source_files": {"fixture/model.py": "b" * 64},
    }
    parent = tmp_path / "private"
    parent.mkdir(mode=0o700)
    publish_checkpoint_bundle(
        parent, "fixture01",
        manifest_raw=canonical_json_bytes(document) + b"\n",
        best_raw=best_raw, last_raw=last_raw, **requested,
    )
    loaded = read_checkpoint_bundle(parent / "fixture01", **requested)
    assert loaded["manifest"]["best_epoch_index"] == 0

    for role, expected in (("best", 1), ("last", 2)):
        target = Model(99)
        restore_model_snapshot(
            model=target,
            specification=loaded["manifest"]["tensor_specification"],
            raw=loaded[role + "_raw"],
        )
        np.testing.assert_array_equal(
            target.weights[0].numpy(),
            np.full(2, expected, dtype=np.float32),
        )


@pytest.mark.parametrize("dataset_id", sorted(DATASET_CLASSES))
def test_invalid_provenance_blocks_publication_before_creation(
    tmp_path, dataset_id,
):
    identity = make_identity(dataset_id, "ar", evidence_for(dataset_id))
    model = Model(1)
    specification, raw = encode_model_snapshot(
        model=model, weights=[model.weights[0].numpy()],
    )
    history = [{
        "epoch_index": epoch, "train_nll": 1.0,
        "validation_nll": 1.0,
        "training_examples": 1, "validation_examples": 1,
        "training_batches": 1, "validation_batches": 1,
        "completed_training_steps": epoch + 1,
    } for epoch in range(2)]
    document = prepare_checkpoint_manifest(
        identity=identity, recipe_raw=recipe_bytes(), history=history,
        specification=specification, best_raw=raw, last_raw=raw,
        source_commit="a" * 40,
        source_files={"fixture/model.py": "b" * 64},
    )
    document["identity"]["dataset_provenance"]["source_fingerprints"] = {}
    parent = tmp_path / "private"
    parent.mkdir(mode=0o700)

    with pytest.raises(ValueError):
        publish_checkpoint_bundle(
            parent, "rejected",
            manifest_raw=canonical_json_bytes(document) + b"\n",
            best_raw=raw, last_raw=raw,
            recipe_raw=recipe_bytes(),
            expected_identity=document["identity"],
            expected_source_commit="a" * 40,
            expected_source_files={"fixture/model.py": "b" * 64},
        )
    assert list(parent.iterdir()) == []
