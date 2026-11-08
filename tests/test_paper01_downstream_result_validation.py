import hashlib
import json
from pathlib import Path
import numpy as np
import pytest

from trustforge.evaluation.paper01_classifier_control import state_digest
from trustforge.evaluation.paper01_downstream_metrics import downstream_metrics
from trustforge.evaluation.paper01_downstream_result_validation import (
    verify_downstream_result,
)

RECIPE = (
    Path(__file__).resolve().parents[1]
    / "studies/paper01_benchmark/revision_2026/downstream_metric_recipe.json"
).read_bytes()


def fixture():
    labels = np.tile(np.arange(5, dtype=np.int64), 2)
    probabilities = np.eye(5, dtype=np.float32)[labels]
    result = downstream_metrics(
        labels=labels, probabilities=probabilities,
        class_count=5, recipe_raw=RECIPE,
    )
    return labels, probabilities, result


def verify(labels, probabilities, result, **overrides):
    raw = (json.dumps(result, sort_keys=True, allow_nan=False) + "\n").encode()
    arguments = {
        "dataset_id": "cicmaldroid2020",
        "labels": labels,
        "probabilities": probabilities,
        "expected_prediction_state_sha256": state_digest([labels, probabilities]),
        "result_raw": raw,
        "expected_result_file_sha256": hashlib.sha256(raw).hexdigest(),
        "metric_recipe_raw": RECIPE,
    }
    arguments.update(overrides)
    return verify_downstream_result(**arguments)


def test_full_report_replay_without_completion_claim():
    result = verify(*fixture())
    assert result["metric_replay_verified"] is True
    assert result["sample_count"] == 10
    assert result["class_count"] == 5
    assert result["provenance_validation"] == "NOT_PERFORMED"
    assert result["complete_evaluation"] is False


@pytest.mark.parametrize("fault", [
    "macro", "class", "bin", "coverage", "boolean", "extra",
])
def test_rehashed_semantic_tampering_rejected(fault):
    labels, probabilities, result = fixture()
    if fault == "macro":
        result["macro_f1"] = 0.5
    elif fault == "class":
        result["per_class"]["0"]["support"] += 1
    elif fault == "bin":
        result["ece_bins"][-1]["count"] += 1
    elif fault == "coverage":
        result["sample_count"] -= 1
    elif fault == "boolean":
        result["macro_f1"] = True
    else:
        result["undeclared"] = 1
    with pytest.raises(ValueError, match="semantics"):
        verify(labels, probabilities, result)


@pytest.mark.parametrize("field", [
    "expected_prediction_state_sha256", "expected_result_file_sha256",
])
def test_wrong_expected_identity_rejected(field):
    with pytest.raises(ValueError):
        verify(*fixture(), **{field: "0" * 64})


def test_changed_recipe_rejected():
    with pytest.raises(ValueError):
        verify(*fixture(), metric_recipe_raw=RECIPE + b" ")


def test_wrong_dataset_class_universe_rejected():
    with pytest.raises(ValueError):
        verify(*fixture(), dataset_id="malex")
