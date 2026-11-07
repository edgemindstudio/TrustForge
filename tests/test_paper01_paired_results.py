from copy import deepcopy
import hashlib
from pathlib import Path
import numpy as np
import pytest

from trustforge.evaluation.paper01_classifier_control import state_digest
from trustforge.evaluation.paper01_downstream_metrics import downstream_metrics
from trustforge.evaluation.paper01_paired_results import link_paired_results

BASE = (
    Path(__file__).resolve().parents[1]
    / "studies/paper01_benchmark/revision_2026"
)
CLASSIFIER = (BASE / "classifier_recipe.json").read_bytes()
METRIC = (BASE / "downstream_metric_recipe.json").read_bytes()
INITIAL = "a" * 64
TEST = {
    "dataset_id": "cicmaldroid2020",
    "split": "test",
    "size": 10,
    "class_count": 5,
    "declared_provenance_sha256": "b" * 64,
}


def arm(perfect, selected):
    labels = np.tile(np.arange(5, dtype=np.int64), 2)
    predicted = labels if perfect else (labels + 1) % 5
    probabilities = np.eye(5, dtype=np.float32)[predicted]
    metrics = downstream_metrics(
        labels=labels, probabilities=probabilities,
        class_count=5, recipe_raw=METRIC,
    )
    return {
        "training": {
            "evaluator_seed": 42,
            "recipe_sha256": hashlib.sha256(CLASSIFIER).hexdigest(),
            "initial_weights_sha256": INITIAL,
            "selected_weights_sha256": selected,
        },
        "predictions": {
            "labels": labels,
            "probabilities": probabilities,
            "report": {
                "schema": "trustforge.paper01.test-predictions.v1",
                "test_identity": deepcopy(TEST),
                "selected_weights_sha256": selected,
                "inference_state_unchanged": True,
                "position_order":
                    "complete_logical_positions_zero_through_size_minus_one",
                "sample_count": 10,
                "maximum_image_batch": 10,
                "ordered_prediction_state_sha256":
                    state_digest([labels, probabilities]),
                "metrics": metrics,
                "complete_evaluation": False,
            },
        },
    }


def arms():
    return {
        "real_only": arm(False, "c" * 64),
        "real_plus_synthetic": arm(True, "d" * 64),
    }


def run(records=None, **overrides):
    kwargs = {
        "arms": arms() if records is None else records,
        "evaluator_seed": 42,
        "expected_test_identity": deepcopy(TEST),
        "expected_initial_weights_sha256": INITIAL,
        "classifier_recipe_raw": CLASSIFIER,
        "metric_recipe_raw": METRIC,
    }
    kwargs.update(overrides)
    return link_paired_results(**kwargs)


def test_linkage_difference_and_no_complete_evaluation_claim():
    result = run()
    assert result["macro_f1_difference"] == 1
    assert result["metric_replay_verified"] is True
    assert result["complete_evaluation"] is False
    assert result["status"] == "LINKED_NOT_COMPLETE_EVALUATION"
    assert result["test_identity"] == TEST


@pytest.mark.parametrize("field,value", [
    ("evaluator_seed", 43),
    ("evaluator_seed", True),
    ("recipe_sha256", "0" * 64),
    ("initial_weights_sha256", "0" * 64),
    ("selected_weights_sha256", "0" * 64),
])
def test_incompatible_training_rejected(field, value):
    records = arms()
    records["real_plus_synthetic"]["training"][field] = value
    with pytest.raises(ValueError):
        run(records)


@pytest.mark.parametrize("field,value", [
    ("selected_weights_sha256", "0" * 64),
    ("inference_state_unchanged", False),
    ("position_order", "shuffled"),
    ("sample_count", 9),
    ("maximum_image_batch", 129),
    ("ordered_prediction_state_sha256", "0" * 64),
])
def test_prediction_linkage_failures_rejected(field, value):
    records = arms()
    records["real_plus_synthetic"]["predictions"]["report"][field] = value
    with pytest.raises(ValueError):
        run(records)


def test_test_identity_mismatch_rejected():
    records = arms()
    records["real_plus_synthetic"]["predictions"]["report"][
        "test_identity"
    ]["declared_provenance_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        run(records)


def test_rehashed_different_ordered_labels_rejected():
    records = arms()
    collected = records["real_plus_synthetic"]["predictions"]
    collected["labels"] = collected["labels"][::-1].copy()
    collected["report"]["ordered_prediction_state_sha256"] = state_digest([
        collected["labels"], collected["probabilities"],
    ])
    with pytest.raises(ValueError, match="paired test labels"):
        run(records)


def test_altered_saved_metric_rejected():
    records = arms()
    records["real_plus_synthetic"]["predictions"]["report"][
        "metrics"
    ]["macro_f1"] = 0.5
    with pytest.raises(ValueError, match="replay"):
        run(records)


def test_altered_probabilities_rejected():
    records = arms()
    records["real_plus_synthetic"]["predictions"]["probabilities"][0] = 0.2
    with pytest.raises(ValueError):
        run(records)


@pytest.mark.parametrize("name", ["classifier_recipe_raw", "metric_recipe_raw"])
def test_changed_recipe_rejected(name):
    raw = CLASSIFIER if name == "classifier_recipe_raw" else METRIC
    with pytest.raises(ValueError):
        run(**{name: raw + b" "})


def test_missing_arm_rejected():
    records = arms()
    del records["real_only"]
    with pytest.raises(ValueError):
        run(records)


def test_output_does_not_alias_input_reports():
    records = arms()
    result = run(records)
    result["arms"]["real_only"]["training"]["evaluator_seed"] = 48
    assert records["real_only"]["training"]["evaluator_seed"] == 42
