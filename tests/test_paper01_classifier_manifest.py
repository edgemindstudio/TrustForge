from copy import deepcopy
import hashlib
from pathlib import Path
import numpy as np
import pytest

from trustforge.evaluation.paper01_classifier_control import state_digest
from trustforge.evaluation.paper01_classifier_state import (
    classifier_specification, encode_classifier_weights,
)
from trustforge.evaluation.paper01_classifier_manifest import (
    prepare_classifier_manifest, verify_classifier_manifest, canonical_manifest,
)

RECIPE = (
    Path(__file__).resolve().parents[1]
    / "studies/paper01_benchmark/revision_2026/classifier_recipe.json"
).read_bytes()


def fixture(losses=None):
    losses = [float(20 - i) for i in range(20)] if losses is None else losses
    weights = [
        np.full(entry["shape"], index / 10, dtype=np.float32)
        for index, entry in enumerate(classifier_specification(
            dataset_id="cicmaldroid2020", recipe_raw=RECIPE,
        ))
    ]
    selected_digest = state_digest(weights)
    state_raw = encode_classifier_weights(
        weights=weights, dataset_id="cicmaldroid2020", recipe_raw=RECIPE,
        expected_selected_weights_sha256=selected_digest,
    )
    context = {
        "dataset_id": "cicmaldroid2020", "evaluator_seed": 42,
        "arm": "real_only", "training_sample_count": 129,
        "validation_sample_count": 5,
        "training_identity_sha256": "a" * 64,
        "validation_identity_sha256": "b" * 64,
        "source_commit": "c" * 40,
        "source_files_sha256": "d" * 64,
        "environment_sha256": "e" * 64,
    }
    best = min(range(len(losses)), key=lambda i: losses[i])
    wait = len(losses) - 1 - best
    report = {
        "recipe_sha256": hashlib.sha256(RECIPE).hexdigest(),
        "evaluator_seed": 42,
        "initial_weights_sha256": "f" * 64,
        "selected_weights_sha256": selected_digest,
        "selected_epoch_index": best,
        "selected_validation_loss": losses[best],
        "completed_epochs": len(losses),
        "optimizer_updates": 2 * len(losses),
        "stopping_outcome": "patience" if wait >= 5 else "maximum_epochs",
        "history": [
            {
                "epoch_index": i, "training_loss": 1.0,
                "validation_loss": loss, "training_examples": 129,
                "validation_examples": 5, "optimizer_updates": 2 * (i + 1),
            }
            for i, loss in enumerate(losses)
        ],
    }
    return context, report, state_raw


def prepare(context, report, state_raw):
    return prepare_classifier_manifest(
        context=context, training_report=report,
        state_raw=state_raw, recipe_raw=RECIPE,
    )


def verify(manifest, state_raw, context, selected_digest, **overrides):
    raw = canonical_manifest(manifest)
    args = {
        "manifest_raw": raw, "state_raw": state_raw, "recipe_raw": RECIPE,
        "expected_manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "expected_context": context,
        "expected_selected_weights_sha256": selected_digest,
    }
    args.update(overrides)
    return verify_classifier_manifest(**args)


def test_roundtrip_and_independent_manifest_representation():
    context, report, state = fixture()
    manifest = prepare(context, report, state)
    assert verify(
        manifest, state, context, report["selected_weights_sha256"],
    ) == manifest
    assert manifest["optimizer_state_included"] is False
    manifest["context"]["arm"] = "real_plus_synthetic"
    assert context["arm"] == "real_only"


def test_patience_and_earliest_tied_minimum():
    context, report, state = fixture([1.0] * 6)
    manifest = prepare(context, report, state)
    assert manifest["training_report"]["selected_epoch_index"] == 0
    assert manifest["training_report"]["stopping_outcome"] == "patience"


@pytest.mark.parametrize("losses", [[1.0] * 5, [1.0] * 7])
def test_premature_stop_or_history_after_patience_rejected(losses):
    with pytest.raises(ValueError):
        prepare(*fixture(losses))


@pytest.mark.parametrize("field,value", [
    ("selected_epoch_index", 0),
    ("selected_epoch_index", True),
    ("selected_validation_loss", 999.0),
    ("optimizer_updates", 39),
    ("completed_epochs", 19),
    ("stopping_outcome", "patience"),
    ("evaluator_seed", 43),
    ("selected_weights_sha256", "0" * 64),
])
def test_invalid_training_selection_rejected(field, value):
    context, report, state = fixture()
    report[field] = value
    with pytest.raises(ValueError):
        prepare(context, report, state)


@pytest.mark.parametrize("field,value", [
    ("epoch_index", True),
    ("training_examples", 128),
    ("validation_examples", 4),
    ("optimizer_updates", 1),
    ("validation_loss", float("nan")),
    ("training_loss", True),
])
def test_invalid_history_rejected(field, value):
    context, report, state = fixture()
    report["history"][0][field] = value
    with pytest.raises(ValueError):
        prepare(context, report, state)


@pytest.mark.parametrize("fault", ["hash", "shape", "extra"])
def test_rehashed_manifest_tampering_rejected(fault):
    context, report, state = fixture()
    manifest = prepare(context, report, state)
    if fault == "hash":
        manifest["state_file_sha256"] = "0" * 64
    elif fault == "shape":
        manifest["tensor_specification"][0]["shape"][0] = 1
    else:
        manifest["complete_evaluation"] = True
    with pytest.raises(ValueError):
        verify(manifest, state, context, report["selected_weights_sha256"])


def test_external_context_mismatch_rejected():
    context, report, state = fixture()
    manifest = prepare(context, report, state)
    changed = deepcopy(context)
    changed["environment_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        verify(manifest, state, changed, report["selected_weights_sha256"])


def test_external_selected_weight_mismatch_rejected():
    context, report, state = fixture()
    manifest = prepare(context, report, state)
    with pytest.raises(ValueError):
        verify(manifest, state, context, "0" * 64)


def test_noncanonical_manifest_rejected_even_with_matching_hash():
    context, report, state = fixture()
    manifest = prepare(context, report, state)
    raw = canonical_manifest(manifest) + b" "
    with pytest.raises(ValueError):
        verify(
            manifest, state, context, report["selected_weights_sha256"],
            manifest_raw=raw,
            expected_manifest_sha256=hashlib.sha256(raw).hexdigest(),
        )
