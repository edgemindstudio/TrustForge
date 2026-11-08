import hashlib
import json
import numpy as np
import pytest

from trustforge.evaluation.paper01_classifier_control import state_digest
from trustforge.evaluation.paper01_ms_ssim import SETTINGS
from trustforge.evaluation.paper01_ms_ssim_pairs import select_ms_ssim_pairs
from trustforge.evaluation.paper01_ms_ssim_result_validation import (
    verify_ms_ssim_result,
)


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def fixture():
    labels = np.repeat(np.arange(5, dtype=np.int64), 2000)
    selection = select_ms_ssim_pairs(
        labels=labels, class_count=5, manifest_sha256="a" * 64,
    )
    selection_raw = canonical(selection) + b"\n"
    scores = np.linspace(0, 1, 1000, dtype=np.float32).reshape(5, 200)
    scorer = {**SETTINGS, "tensorflow": "fixture"}
    per_class = {
        str(c): {
            "pair_count": 200,
            "scores": scores[c].tolist(),
            "mean": float(np.mean(scores[c].tolist(), dtype=np.float64)),
        }
        for c in range(5)
    }
    report = {
        "metric": SETTINGS["metric"],
        "dataset_id": "cicmaldroid2020",
        "manifest_sha256": "a" * 64,
        "selection_file_sha256": sha(selection_raw),
        "scorer": scorer,
        "class_count": 5,
        "per_class": per_class,
        "macro_mean": float(np.mean([
            per_class[str(c)]["mean"] for c in range(5)
        ], dtype=np.float64)),
        "pair_count": 1000,
        "class_coverage_complete": True,
        "complete_evaluation": False,
    }
    return {
        "report": report,
        "selection": selection,
        "arguments": {
            "dataset_id": "cicmaldroid2020",
            "selection_raw": selection_raw,
            "expected_selection_file_sha256": sha(selection_raw),
            "labels": labels,
            "expected_manifest_sha256": "a" * 64,
            "expected_scorer_identity": scorer,
            "pair_scores": scores,
            "expected_pair_scores_state_sha256": state_digest([scores]),
        },
    }


def verify(data):
    raw = canonical(data["report"]) + b"\n"
    return verify_ms_ssim_result(
        result_raw=raw, expected_result_file_sha256=sha(raw),
        **data["arguments"],
    )


def test_complete_selection_and_aggregation_replay():
    result = verify(fixture())
    assert result["pair_selection_replay_verified"] is True
    assert result["score_coverage_and_aggregation_verified"] is True
    assert result["pair_count"] == 1000
    assert result["image_score_recomputation"] == "NOT_PERFORMED"
    assert result["complete_evaluation"] is False


@pytest.mark.parametrize("fault", [
    "macro", "class_mean", "missing_class", "missing_score",
    "score", "pair_count", "coverage", "complete_flag",
])
def test_rehashed_report_tampering_rejected(fault):
    data = fixture()
    report = data["report"]
    if fault == "macro":
        report["macro_mean"] += 0.1
    elif fault == "class_mean":
        report["per_class"]["0"]["mean"] += 0.1
    elif fault == "missing_class":
        del report["per_class"]["0"]
    elif fault == "missing_score":
        report["per_class"]["0"]["scores"].pop()
    elif fault == "score":
        report["per_class"]["0"]["scores"][0] += 0.1
    elif fault == "pair_count":
        report["pair_count"] -= 1
    elif fault == "coverage":
        report["class_coverage_complete"] = False
    else:
        report["complete_evaluation"] = True
    with pytest.raises(ValueError):
        verify(data)


def test_rehashed_pair_selection_tampering_rejected():
    data = fixture()
    pairs = data["selection"]["classes"]["0"]["manifest_position_pairs"]
    pairs[0], pairs[1] = pairs[1], pairs[0]
    raw = canonical(data["selection"]) + b"\n"
    data["arguments"]["selection_raw"] = raw
    data["arguments"]["expected_selection_file_sha256"] = sha(raw)
    data["report"]["selection_file_sha256"] = sha(raw)
    with pytest.raises(ValueError, match="selection"):
        verify(data)


@pytest.mark.parametrize("fault", ["changed", "dtype", "shape", "nonfinite", "order"])
def test_pair_score_array_failures_rejected(fault):
    data = fixture()
    scores = data["arguments"]["pair_scores"]
    if fault == "changed":
        scores[0, 0] += 0.1
    elif fault == "dtype":
        data["arguments"]["pair_scores"] = scores.astype(np.float64)
    elif fault == "shape":
        data["arguments"]["pair_scores"] = scores[:, :-1]
    elif fault == "nonfinite":
        scores[0, 0] = np.nan
    else:
        data["arguments"]["pair_scores"] = scores[:, ::-1].copy()
    with pytest.raises(ValueError):
        verify(data)


def test_rehashed_changed_scores_still_require_matching_report():
    data = fixture()
    scores = data["arguments"]["pair_scores"]
    scores[0, 0] += 0.1
    data["arguments"]["expected_pair_scores_state_sha256"] = state_digest([scores])
    with pytest.raises(ValueError, match="semantic"):
        verify(data)


def test_wrong_manifest_expectation_rejected():
    data = fixture()
    data["arguments"]["expected_manifest_sha256"] = "b" * 64
    with pytest.raises(ValueError):
        verify(data)


@pytest.mark.parametrize("field,value", [
    ("filter_size", 11),
    ("tensorflow", ""),
    ("max_val", True),
])
def test_invalid_scorer_contract_rejected(field, value):
    data = fixture()
    data["arguments"]["expected_scorer_identity"][field] = value
    with pytest.raises(ValueError):
        verify(data)
