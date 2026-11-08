"""Verify saved pooled FID/class-FID against KID-bound feature replay.

The supplied result object contains exactly pooled and class_fid records.
No feature extraction, provenance authentication or completion authorization.
"""

import hashlib
import json
import re
import numpy as np

from .paper01_kid_result_validation import verify_kid_result
from .paper01_fid import feature_fid, complete_class_fid
from .paper01_result_artifacts import _verified_json


def _canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def verify_fid_result(
    *, result_raw, expected_result_file_sha256, kid_verification_arguments,
):
    if (
        type(expected_result_file_sha256) is not str
        or re.fullmatch(r"[0-9a-f]{64}", expected_result_file_sha256) is None
    ):
        raise ValueError("explicit expected result SHA-256 required")
    if type(kid_verification_arguments) is not dict:
        raise TypeError("KID verification inputs required")
    saved = _verified_json(result_raw, expected_result_file_sha256)
    if set(saved) != {"pooled", "class_fid"}:
        raise ValueError("exact pooled and class-FID result records required")

    # Revalidate supplied evidence; do not trust a caller-provided receipt.
    kid_receipt = verify_kid_result(**kid_verification_arguments)
    real = kid_verification_arguments["real_features"]
    synthetic = kid_verification_arguments["synthetic_features"]
    classes = kid_receipt["class_count"]
    m = kid_receipt["samples_per_class"]

    # The verified reference selector orders rows class-major.
    labels = np.repeat(np.arange(classes, dtype=np.int64), m)
    replayed = {
        "pooled": feature_fid(real, synthetic),
        "class_fid": complete_class_fid(
            real, synthetic,
            real_labels=labels, synthetic_labels=labels,
            class_count=classes,
        ),
    }
    if _canonical(saved) != _canonical(replayed):
        raise ValueError("saved FID/class-FID differs from exact semantic replay")

    return {
        "schema": "trustforge.paper01.fid-result-verification.v1",
        "dataset_id": kid_receipt["dataset_id"],
        "result_file_sha256": expected_result_file_sha256,
        "kid_verification_sha256": hashlib.sha256(
            _canonical(kid_receipt)
        ).hexdigest(),
        "selection_file_sha256": kid_receipt["selection_file_sha256"],
        "real_features_sha256": kid_receipt["real_features_sha256"],
        "synthetic_features_sha256": kid_receipt["synthetic_features_sha256"],
        "class_count": classes,
        "samples_per_class": m,
        "pooled_fid_replay_verified": True,
        "all_class_fid_replay_verified": True,
        "feature_sets_shared_with_kid": True,
        "feature_extraction_authentication": "NOT_PERFORMED",
        "provenance_validation": "NOT_PERFORMED",
        "complete_evaluation": False,
    }
