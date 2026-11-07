"""Execution-envelope component checks using real v1 validators.

Checkpoint tensors and scaler inputs are synthetic. Source/context fields
are declarations; no source files or real dataset bytes are authenticated.
"""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from trustforge.data.paper01_revision_2026.cic_fitted_state import (
    capture_fitted_state,
)
from trustforge.models.paper01_revision_2026.ar_maf_checkpoint_identity import (
    build_checkpoint_identity, canonical_json_bytes,
)
from trustforge.models.paper01_revision_2026.ar_maf_checkpoint_manifest import (
    prepare_checkpoint_manifest,
)
from trustforge.models.paper01_revision_2026.ar_maf_checkpoint_npz import (
    encode_network_state,
)
from trustforge.models.paper01_revision_2026.gan_v3_provenance_adapter import (
    build_gan_v3_provenance,
)
from trustforge.models.paper01_revision_2026 import (
    ar_maf_cic_execution_manifest as execution,
)


@pytest.fixture(scope="module")
def fixture():
    recipe = (
        Path(__file__).resolve().parents[1]
        / "studies/paper01_benchmark/revision_2026/ar_maf_integration_recipe.json"
    ).read_bytes()
    preflight = {
        "dataset_id": "cicmaldroid2020",
        "source_contract": "cic_authoritative_source_revision_2026",
        "integrity_settings_verified": True,
        "verified_sha256": {
            "source_csv":
                "3415babdd76a2b77c400a9ec3a8a6057705d5a57a9e91753191ecc2946df7c00",
            "feature_columns_path":
                "7a6045fe10bfe27c1faee98d729de470401b7cf733ccccc107499f11dea8fc8f",
        },
    }
    provenance = build_gan_v3_provenance(
        dataset_id="cicmaldroid2020", verified_preflight=preflight,
    )
    identity = build_checkpoint_identity(
        family="ar", dataset_id="cicmaldroid2020",
        experimental_seed=42, class_mapping=[str(i) for i in range(5)],
        recipe_raw=recipe, dataset_provenance=provenance,
    )
    specification = [{
        "key": "model_0000", "shape": [2],
        "dtype": "float32", "trainable": True,
    }]
    state = encode_network_state(
        role="model", specification=specification,
        state={"model_0000": np.ones(2, dtype=np.float32)},
    )
    history = [{
        "epoch_index": epoch,
        "train_nll": 3.0 - epoch,
        "validation_nll": 2.0 - epoch,
        "training_examples": 8117,
        "validation_examples": 1158,
        "training_batches": 127,
        "validation_batches": 19,
        "completed_training_steps": 127 * (epoch + 1),
    } for epoch in range(2)]
    commit = "a" * 40
    sources = {"src/fixture.py": "b" * 64}
    manifest = prepare_checkpoint_manifest(
        identity=identity, recipe_raw=recipe, history=history,
        specification=specification, best_raw=state, last_raw=state,
        source_commit=commit, source_files=sources,
    )
    scaler = StandardScaler().fit(np.zeros((8117, 139), dtype=np.float32))
    indices = np.arange(8117, dtype=np.int64)
    context = {
        "source_preflight": preflight,
        "configuration_sha256": "c" * 64,
        "feature_columns": [f"fixture_feature_{i}" for i in range(139)],
        "partition": {"seed": 42, "ratios": [0.7, 0.1, 0.2]},
        "implementation": {
            "commit": commit, "tracked_source_files": sources,
        },
        "environment": {
            "python": "fixture", "numpy": "fixture",
            "pandas": "fixture", "scikit_learn": "fixture",
            "tensorflow": "fixture", "byteorder": "little",
        },
    }
    fitted = capture_fitted_state(
        scaler=scaler, training_indices=indices, context=context,
    )
    arguments = {
        "checkpoint_manifest_raw": canonical_json_bytes(manifest) + b"\n",
        "fitted_state_raw": canonical_json_bytes(fitted) + b"\n",
        "best_raw": state,
        "last_raw": state,
        "recipe_raw": recipe,
        "expected_identity": identity,
        "expected_source_commit": commit,
        "expected_source_files": sources,
        "expected_fitted_state_sha256": fitted["sha256"],
    }
    return arguments, scaler, indices, context, manifest


def verify_arguments(arguments, raw):
    requested = {
        key: value for key, value in arguments.items()
        if key not in ("checkpoint_manifest_raw", "fitted_state_raw")
    }
    return execution.verify_cic_execution_manifest(
        execution_manifest_raw=raw, **requested,
    )


def test_complete_envelope_roundtrip(fixture):
    arguments, _, _, _, manifest = fixture
    raw = execution.prepare_cic_execution_manifest(**arguments)
    result = verify_arguments(arguments, raw)
    assert result["manifest"] == manifest
    assert result["best_raw"] == arguments["best_raw"]
    assert result["fitted_state_raw"] == arguments["fitted_state_raw"]


def test_independent_expected_fitted_identity_required(fixture):
    arguments = dict(fixture[0])
    arguments["expected_fitted_state_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="fitted-state identity mismatch"):
        execution.prepare_cic_execution_manifest(**arguments)


@pytest.mark.parametrize("field", ["commit", "tracked_source_files"])
def test_source_binding_mismatch(fixture, field):
    original, scaler, indices, context, _ = fixture
    arguments = dict(original)
    context = deepcopy(context)
    context["implementation"][field] = (
        "d" * 40 if field == "commit" else {"src/other.py": "e" * 64}
    )
    record = capture_fitted_state(
        scaler=scaler, training_indices=indices, context=context,
    )
    arguments["fitted_state_raw"] = canonical_json_bytes(record) + b"\n"
    arguments["expected_fitted_state_sha256"] = record["sha256"]
    with pytest.raises(ValueError, match="source identities differ"):
        execution.prepare_cic_execution_manifest(**arguments)


def test_corrupt_checkpoint_state_rejected(fixture):
    arguments = dict(fixture[0])
    arguments["best_raw"] = b"not an NPZ archive"
    with pytest.raises(Exception):
        execution.prepare_cic_execution_manifest(**arguments)


def test_payload_tampering_rejected(fixture):
    import json
    arguments = dict(fixture[0])
    record = json.loads(arguments["fitted_state_raw"])
    record["payload"]["declared_context"]["configuration_sha256"] = "e" * 64
    arguments["fitted_state_raw"] = canonical_json_bytes(record) + b"\n"
    with pytest.raises(ValueError, match="digest mismatch"):
        execution.prepare_cic_execution_manifest(**arguments)


@pytest.mark.parametrize("fault", ["missing", "digest", "extra", "noncanonical"])
def test_envelope_tampering_rejected(fixture, fault):
    import json
    arguments = fixture[0]
    raw = execution.prepare_cic_execution_manifest(**arguments)
    envelope = json.loads(raw)
    if fault == "missing":
        del envelope["fitted_state"]
    elif fault == "digest":
        envelope["checkpoint_manifest_sha256"] = "e" * 64
    elif fault == "extra":
        envelope["unexpected"] = True
    else:
        with pytest.raises(ValueError):
            verify_arguments(arguments, raw + b"\n")
        return
    changed = canonical_json_bytes(envelope) + b"\n"
    with pytest.raises(ValueError):
        verify_arguments(arguments, changed)
