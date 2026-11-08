"""Verify saved KID reports against selections and feature-array replay.

Does not authenticate original data, feature extraction, generator lineage or
selection persistence timing. Exact replay is environment-specific.
"""

import hashlib
import json
import re
import numpy as np

from .paper01_inputs import CLASS_COUNTS
from .paper01_reference_selection import select_balanced_references
from .paper01_inception import WEIGHTS_SHA256
from .paper01_kid import kid_subsets
from .paper01_result_artifacts import _verified_json


def _canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def _digest(value):
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("explicit expected SHA-256 required")
    return value


def verify_kid_result(
    *, result_raw, expected_result_file_sha256,
    selection_raw, expected_selection_file_sha256,
    expected_real_validation_identity, expected_synthetic_manifest_sha256,
    real_validation_labels, synthetic_labels,
    real_features, synthetic_features, expected_feature_extractor,
):
    saved = _verified_json(result_raw, _digest(expected_result_file_sha256))
    selection = _verified_json(
        selection_raw, _digest(expected_selection_file_sha256),
    )
    identity = expected_real_validation_identity
    if type(identity) is not dict:
        raise ValueError("explicit real-validation identity required")
    dataset = identity.get("dataset_id")
    if (
        dataset not in CLASS_COUNTS
        or identity.get("split") != "validation"
        or type(identity.get("class_count")) is not int
        or identity["class_count"] != CLASS_COUNTS[dataset]
        or type(identity.get("size")) is not int
        or identity["size"] != len(real_validation_labels)
    ):
        raise ValueError("incompatible real-validation identity")
    classes = CLASS_COUNTS[dataset]
    selected = select_balanced_references(
        real_validation_labels=real_validation_labels,
        synthetic_labels=synthetic_labels,
        class_count=classes,
        real_validation_identity_sha256=hashlib.sha256(
            _canonical(identity)
        ).hexdigest(),
        synthetic_manifest_sha256=_digest(expected_synthetic_manifest_sha256),
    )
    if _canonical(selection) != _canonical(selected):
        raise ValueError("saved selection differs from reference replay")

    extractor = expected_feature_extractor
    required = {
        "schema": "trustforge.paper01.inception-features.v1",
        "architecture": "Keras_InceptionV3",
        "include_top": False,
        "pooling": "avg",
        "weights_sha256": WEIGHTS_SHA256,
        "weights_bytes": 87910968,
        "input": "float32_NHWC_40x40x3_[0,1]",
        "resize": {
            "shape": [299, 299], "method": "bilinear", "antialias": False,
        },
        "preprocessing": "multiply_255_then_inception_v3_preprocess_input",
        "preserve_rgb": True,
        "training": False,
        "feature_dimension": 2048,
        "maximum_batch_size": 16,
    }
    if (
        type(extractor) is not dict
        or set(extractor) != set(required) | {"tensorflow", "keras"}
        or any(
            _canonical(extractor[key]) != _canonical(value)
            for key, value in required.items()
        )
        or any(
            type(extractor[key]) is not str or not extractor[key].strip()
            for key in ("tensorflow", "keras")
        )
    ):
        raise ValueError("selected extractor identity required")

    m = selected["real_reference"]["samples_per_class"]
    shape = (classes * m, 2048)
    for features in (real_features, synthetic_features):
        if (
            not isinstance(features, np.ndarray)
            or features.dtype != np.float32
            or features.shape != shape
        ):
            raise ValueError("selected float32 feature-array contract mismatch")
        for start in range(0, len(features), 200):
            if not np.isfinite(features[start:start + 200]).all():
                raise ValueError("nonfinite feature data")

    replayed = {
        "metric": "pooled_class_balanced_kid",
        "selection_file_sha256": expected_selection_file_sha256,
        "real_reference_sha256": selected["real_reference_sha256"],
        "synthetic_reference_sha256": selected["synthetic_reference_sha256"],
        "dataset_id": dataset,
        "class_count": classes,
        "samples_per_class": m,
        "feature_extractor": extractor,
        "feature_shape": list(shape),
        "real_features_sha256": hashlib.sha256(
            real_features.tobytes(order="C")
        ).hexdigest(),
        "synthetic_features_sha256": hashlib.sha256(
            synthetic_features.tobytes(order="C")
        ).hexdigest(),
        "kid": kid_subsets(real_features, synthetic_features),
        "complete_evaluation": False,
    }
    if _canonical(saved) != _canonical(replayed):
        raise ValueError("saved KID report differs from exact semantic replay")
    return {
        "schema": "trustforge.paper01.kid-result-verification.v1",
        "dataset_id": dataset,
        "result_file_sha256": expected_result_file_sha256,
        "selection_file_sha256": expected_selection_file_sha256,
        "real_features_sha256": replayed["real_features_sha256"],
        "synthetic_features_sha256": replayed["synthetic_features_sha256"],
        "class_count": classes,
        "samples_per_class": m,
        "reference_selection_replay_verified": True,
        "kid_replay_verified": True,
        "feature_extraction_authentication": "NOT_PERFORMED",
        "provenance_validation": "NOT_PERFORMED",
        "complete_evaluation": False,
    }
