"""Verify saved downstream metric bytes against ordered prediction replay.

Prediction provenance, selected-model linkage and test-membership authenticity
remain outer requirements. This verifier does not grant complete evaluation.
"""

import re
import numpy as np

from .paper01_inputs import CLASS_COUNTS
from .paper01_classifier_control import state_digest
from .paper01_downstream_metrics import (
    downstream_metrics, selected_metric_recipe, RECIPE_SHA256,
)
from .paper01_result_artifacts import _verified_json


def verify_downstream_result(
    *, dataset_id, labels, probabilities, expected_prediction_state_sha256,
    result_raw, expected_result_file_sha256, metric_recipe_raw,
):
    if dataset_id not in CLASS_COUNTS:
        raise ValueError("authoritative dataset required")
    for value in (
        expected_prediction_state_sha256, expected_result_file_sha256,
    ):
        if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError("explicit expected SHA-256 required")
    selected_metric_recipe(metric_recipe_raw)
    if (
        not isinstance(labels, np.ndarray) or labels.dtype != np.int64
        or not isinstance(probabilities, np.ndarray)
        or probabilities.dtype != np.float32
    ):
        raise ValueError("collector-format labels and probabilities required")
    if state_digest([labels, probabilities]) != expected_prediction_state_sha256:
        raise ValueError("ordered prediction identity mismatch")

    saved = _verified_json(result_raw, expected_result_file_sha256)
    replayed = downstream_metrics(
        labels=labels, probabilities=probabilities,
        class_count=CLASS_COUNTS[dataset_id], recipe_raw=metric_recipe_raw,
    )
    # Type-sensitive comparison: booleans must not stand in for numbers.
    def identical(first, second):
        if type(first) is not type(second):
            return False
        if type(first) is dict:
            return first.keys() == second.keys() and all(
                identical(first[key], second[key]) for key in first
            )
        if type(first) is list:
            return len(first) == len(second) and all(
                identical(a, b) for a, b in zip(first, second)
            )
        return first == second

    if not identical(saved, replayed):
        raise ValueError("saved downstream semantics differ from exact replay")
    return {
        "schema": "trustforge.paper01.downstream-result-verification.v1",
        "dataset_id": dataset_id,
        "result_file_sha256": expected_result_file_sha256,
        "metric_recipe_sha256": RECIPE_SHA256,
        "ordered_prediction_state_sha256": expected_prediction_state_sha256,
        "sample_count": len(labels),
        "class_count": CLASS_COUNTS[dataset_id],
        "metric_replay_verified": True,
        "provenance_validation": "NOT_PERFORMED",
        "complete_evaluation": False,
    }
