"""Persisted-pair MS-SSIM orchestration with complete class coverage.

Checks declared identities and consumed batches. Dataset provenance, generator
lineage and persistence timing remain outer requirements.
"""

from copy import deepcopy
import hashlib
import json
import re
import numpy as np

from .paper01_inputs import CLASS_COUNTS, validate_batch
from .paper01_ms_ssim import SETTINGS
from .paper01_ms_ssim_pairs import select_ms_ssim_pairs


def _canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def evaluate_ms_ssim_pairs(
    *, selection_raw, selection_sha256, synthetic_view, labels, scorer,
):
    if (
        type(selection_raw) is not bytes
        or type(selection_sha256) is not str
        or re.fullmatch(r"[0-9a-f]{64}", selection_sha256) is None
        or hashlib.sha256(selection_raw).hexdigest() != selection_sha256
    ):
        raise ValueError("persisted pair-selection identity mismatch")

    identity = deepcopy(synthetic_view.identity)
    dataset = identity.get("dataset_id")
    if dataset not in CLASS_COUNTS:
        raise ValueError("unsupported dataset")
    classes = CLASS_COUNTS[dataset]
    expected = select_ms_ssim_pairs(
        labels=labels, class_count=classes,
        manifest_sha256=identity.get("manifest_sha256"),
    )
    if synthetic_view.size != len(labels):
        raise ValueError("pool size differs from ordered labels")
    if _canonical(json.loads(selection_raw)) != _canonical(expected):
        raise ValueError("persisted pairs differ from selected policy")
    labels = labels.copy()

    scorer_identity = deepcopy(scorer.identity)
    if any(
        scorer_identity.get(key) != value
        for key, value in SETTINGS.items()
    ) or not isinstance(scorer_identity.get("tensorflow"), str) \
            or not scorer_identity["tensorflow"].strip():
        raise ValueError("selected scorer contract required")

    def unchanged():
        if (
            synthetic_view.identity != identity
            or synthetic_view.size != len(labels)
            or scorer.identity != scorer_identity
        ):
            raise RuntimeError("pool or scorer identity changed")

    synthetic_view.preflight()
    unchanged()
    per_class = {}
    for class_id in range(classes):
        pairs = np.asarray(
            expected["classes"][str(class_id)]["manifest_position_pairs"],
            dtype=np.int64,
        )
        scores = []
        for start in range(0, 200, 16):
            unchanged()
            selected = pairs[start:start + 16]
            positions, inverse = np.unique(selected.ravel(), return_inverse=True)
            observed, images, one_hot = synthetic_view.batch(positions.copy())
            if not np.array_equal(observed, positions):
                raise RuntimeError("pool changed requested positions")
            validate_batch(
                images, one_hot, count=len(positions), class_count=classes,
            )
            if not np.array_equal(one_hot.argmax(axis=1), labels[positions]):
                raise RuntimeError("image labels differ from selection labels")
            paired = inverse.reshape(-1, 2)
            values = scorer.score_pairs(
                images[paired[:, 0]], images[paired[:, 1]],
            )
            if (
                not isinstance(values, np.ndarray)
                or values.shape != (len(selected),)
                or values.dtype != np.float32
                or not np.isfinite(values).all()
            ):
                raise ValueError("failed or incomplete pair scores")
            scores.extend(values.tolist())
            unchanged()
        if len(scores) != 200:
            raise RuntimeError("incomplete class pair coverage")
        per_class[str(class_id)] = {
            "pair_count": 200,
            "scores": scores,
            "mean": float(np.mean(scores, dtype=np.float64)),
        }
    unchanged()
    return {
        "metric": SETTINGS["metric"],
        "dataset_id": dataset,
        "manifest_sha256": identity["manifest_sha256"],
        "selection_file_sha256": selection_sha256,
        "scorer": scorer_identity,
        "class_count": classes,
        "per_class": per_class,
        "macro_mean": float(np.mean([
            per_class[str(c)]["mean"] for c in range(classes)
        ], dtype=np.float64)),
        "pair_count": classes * 200,
        "class_coverage_complete": True,
        "complete_evaluation": False,
    }
