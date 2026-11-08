"""Verify MS-SSIM report selection, pair-score identities and aggregation.

Does not authenticate image provenance or recompute TensorFlow image scores.
Pair-score array identities must come from the outer execution evidence.
"""

import hashlib
import json
import re
import numpy as np

from .paper01_inputs import CLASS_COUNTS
from .paper01_ms_ssim import SETTINGS
from .paper01_ms_ssim_pairs import select_ms_ssim_pairs
from .paper01_classifier_control import state_digest
from .paper01_result_artifacts import _verified_json


def _canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def _digest(value):
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("explicit expected SHA-256 required")
    return value


def verify_ms_ssim_result(
    *, dataset_id, result_raw, expected_result_file_sha256,
    selection_raw, expected_selection_file_sha256,
    labels, expected_manifest_sha256, expected_scorer_identity,
    pair_scores, expected_pair_scores_state_sha256,
):
    if dataset_id not in CLASS_COUNTS:
        raise ValueError("authoritative dataset required")
    classes = CLASS_COUNTS[dataset_id]
    saved = _verified_json(result_raw, _digest(expected_result_file_sha256))
    selection = _verified_json(
        selection_raw, _digest(expected_selection_file_sha256),
    )
    manifest = _digest(expected_manifest_sha256)
    selected = select_ms_ssim_pairs(
        labels=labels, class_count=classes, manifest_sha256=manifest,
    )
    if _canonical(selection) != _canonical(selected):
        raise ValueError("saved pair selection differs from policy replay")

    scorer = expected_scorer_identity
    if (
        type(scorer) is not dict
        or set(scorer) != set(SETTINGS) | {"tensorflow"}
        or any(
            _canonical(scorer[key]) != _canonical(value)
            for key, value in SETTINGS.items()
        )
        or type(scorer["tensorflow"]) is not str
        or not scorer["tensorflow"].strip()
    ):
        raise ValueError("selected scorer identity required")
    if (
        not isinstance(pair_scores, np.ndarray)
        or pair_scores.dtype != np.float32
        or pair_scores.shape != (classes, 200)
        or not np.isfinite(pair_scores).all()
    ):
        raise ValueError("complete ordered float32 class-by-pair scores required")
    if state_digest([pair_scores]) != _digest(expected_pair_scores_state_sha256):
        raise ValueError("pair-score array identity mismatch")

    per_class = {}
    for class_id in range(classes):
        scores = pair_scores[class_id].tolist()
        per_class[str(class_id)] = {
            "pair_count": 200,
            "scores": scores,
            "mean": float(np.mean(scores, dtype=np.float64)),
        }
    replayed = {
        "metric": SETTINGS["metric"],
        "dataset_id": dataset_id,
        "manifest_sha256": manifest,
        "selection_file_sha256": expected_selection_file_sha256,
        "scorer": scorer,
        "class_count": classes,
        "per_class": per_class,
        "macro_mean": float(np.mean([
            per_class[str(c)]["mean"] for c in range(classes)
        ], dtype=np.float64)),
        "pair_count": classes * 200,
        "class_coverage_complete": True,
        "complete_evaluation": False,
    }
    if _canonical(saved) != _canonical(replayed):
        raise ValueError("saved MS-SSIM report differs from semantic replay")
    return {
        "schema": "trustforge.paper01.ms-ssim-result-verification.v1",
        "dataset_id": dataset_id,
        "result_file_sha256": expected_result_file_sha256,
        "selection_file_sha256": expected_selection_file_sha256,
        "manifest_sha256": manifest,
        "pair_scores_state_sha256": expected_pair_scores_state_sha256,
        "scorer_identity_sha256": hashlib.sha256(_canonical(scorer)).hexdigest(),
        "class_count": classes,
        "pair_count": classes * 200,
        "pair_selection_replay_verified": True,
        "score_coverage_and_aggregation_verified": True,
        "image_score_recomputation": "NOT_PERFORMED",
        "provenance_validation": "NOT_PERFORMED",
        "complete_evaluation": False,
    }
