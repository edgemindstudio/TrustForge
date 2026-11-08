import hashlib
import json
import numpy as np
import pytest

from trustforge.evaluation.paper01_inception import WEIGHTS_SHA256
from trustforge.evaluation.paper01_kid import kid_subsets
from trustforge.evaluation.paper01_reference_selection import select_balanced_references
from trustforge.evaluation.paper01_kid_result_validation import verify_kid_result


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def fixture():
    real_labels = np.repeat(np.arange(5, dtype=np.int64), 2)
    synthetic_labels = np.repeat(np.arange(5, dtype=np.int64), 2000)
    identity = {
        "dataset_id": "cicmaldroid2020", "split": "validation",
        "size": 10, "class_count": 5,
        "declared_provenance_sha256": "a" * 64,
    }
    selection = select_balanced_references(
        real_validation_labels=real_labels,
        synthetic_labels=synthetic_labels,
        class_count=5,
        real_validation_identity_sha256=sha(canonical(identity)),
        synthetic_manifest_sha256="b" * 64,
    )
    selection_raw = canonical(selection) + b"\n"
    extractor = {
        "schema": "trustforge.paper01.inception-features.v1",
        "architecture": "Keras_InceptionV3",
        "include_top": False, "pooling": "avg",
        "weights_sha256": WEIGHTS_SHA256, "weights_bytes": 87910968,
        "input": "float32_NHWC_40x40x3_[0,1]",
        "resize": {
            "shape": [299, 299], "method": "bilinear", "antialias": False,
        },
        "preprocessing": "multiply_255_then_inception_v3_preprocess_input",
        "preserve_rgb": True, "training": False,
        "feature_dimension": 2048, "maximum_batch_size": 16,
        "tensorflow": "fixture", "keras": "fixture",
    }
    rng = np.random.default_rng(91)
    real = rng.normal(size=(10, 2048)).astype(np.float32)
    synthetic = rng.normal(size=(10, 2048)).astype(np.float32)
    report = {
        "metric": "pooled_class_balanced_kid",
        "selection_file_sha256": sha(selection_raw),
        "real_reference_sha256": selection["real_reference_sha256"],
        "synthetic_reference_sha256": selection["synthetic_reference_sha256"],
        "dataset_id": "cicmaldroid2020", "class_count": 5,
        "samples_per_class": 2,
        "feature_extractor": extractor,
        "feature_shape": [10, 2048],
        "real_features_sha256": sha(real.tobytes(order="C")),
        "synthetic_features_sha256": sha(synthetic.tobytes(order="C")),
        "kid": kid_subsets(real, synthetic),
        "complete_evaluation": False,
    }
    return {
        "report": report,
        "selection": selection,
        "arguments": {
            "selection_raw": selection_raw,
            "expected_selection_file_sha256": sha(selection_raw),
            "expected_real_validation_identity": identity,
            "expected_synthetic_manifest_sha256": "b" * 64,
            "real_validation_labels": real_labels,
            "synthetic_labels": synthetic_labels,
            "real_features": real,
            "synthetic_features": synthetic,
            "expected_feature_extractor": extractor,
        },
    }


def verify(data):
    raw = canonical(data["report"]) + b"\n"
    return verify_kid_result(
        result_raw=raw, expected_result_file_sha256=sha(raw),
        **data["arguments"],
    )


def test_complete_reference_and_kid_replay():
    result = verify(fixture())
    assert result["reference_selection_replay_verified"] is True
    assert result["kid_replay_verified"] is True
    assert result["class_count"] == 5
    assert result["samples_per_class"] == 2
    assert result["feature_extraction_authentication"] == "NOT_PERFORMED"
    assert result["complete_evaluation"] is False


@pytest.mark.parametrize("fault", [
    "mean", "estimate", "missing_draw", "indices", "feature_hash",
    "class_count", "complete_flag", "extra_field",
])
def test_rehashed_report_tampering_rejected(fault):
    data = fixture()
    report = data["report"]
    if fault == "mean":
        report["kid"]["mean"] += 0.1
    elif fault == "estimate":
        report["kid"]["draws"][0]["estimate"] += 0.1
    elif fault == "missing_draw":
        report["kid"]["draws"].pop()
    elif fault == "indices":
        report["kid"]["draws"][0]["real_indices"].reverse()
    elif fault == "feature_hash":
        report["real_features_sha256"] = "0" * 64
    elif fault == "class_count":
        report["class_count"] = 4
    elif fault == "complete_flag":
        report["complete_evaluation"] = True
    else:
        report["extra"] = 1
    with pytest.raises(ValueError):
        verify(data)


def test_rehashed_selection_tampering_rejected():
    data = fixture()
    data["selection"]["real_reference"]["positions_by_class"][0].reverse()
    raw = canonical(data["selection"]) + b"\n"
    data["arguments"]["selection_raw"] = raw
    data["arguments"]["expected_selection_file_sha256"] = sha(raw)
    data["report"]["selection_file_sha256"] = sha(raw)
    with pytest.raises(ValueError, match="selection"):
        verify(data)


@pytest.mark.parametrize("fault", ["changed", "dtype", "shape", "nonfinite"])
def test_invalid_or_changed_features_rejected(fault):
    data = fixture()
    real = data["arguments"]["real_features"]
    if fault == "changed":
        real[0, 0] += 1
    elif fault == "dtype":
        data["arguments"]["real_features"] = real.astype(np.float64)
    elif fault == "shape":
        data["arguments"]["real_features"] = real[:-1]
    else:
        real[0, 0] = np.nan
    with pytest.raises(ValueError):
        verify(data)


@pytest.mark.parametrize("field,value", [
    ("weights_sha256", "0" * 64),
    ("preserve_rgb", False),
    ("tensorflow", ""),
    ("feature_dimension", 2047),
])
def test_extractor_mismatch_rejected(field, value):
    data = fixture()
    data["arguments"]["expected_feature_extractor"][field] = value
    with pytest.raises(ValueError):
        verify(data)


def test_wrong_manifest_expectation_rejected():
    data = fixture()
    data["arguments"]["expected_synthetic_manifest_sha256"] = "c" * 64
    with pytest.raises(ValueError):
        verify(data)


def test_non_validation_reference_rejected():
    data = fixture()
    data["arguments"]["expected_real_validation_identity"]["split"] = "test"
    with pytest.raises(ValueError):
        verify(data)
