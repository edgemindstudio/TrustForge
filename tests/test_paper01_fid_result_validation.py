from pathlib import Path
import runpy
import numpy as np
import pytest

from trustforge.evaluation.paper01_fid import feature_fid, complete_class_fid
from trustforge.evaluation.paper01_fid_result_validation import verify_fid_result

# Reuse the established reference/feature fixture without duplicating its recipe.
HELPERS = runpy.run_path(str(
    Path(__file__).with_name("test_paper01_kid_result_validation.py")
))
canonical = HELPERS["canonical"]
sha = HELPERS["sha"]


def fixture():
    data = HELPERS["fixture"]()
    arguments = data["arguments"]
    kid_raw = canonical(data["report"]) + b"\n"
    kid_arguments = {
        **arguments,
        "result_raw": kid_raw,
        "expected_result_file_sha256": sha(kid_raw),
    }
    real = arguments["real_features"]
    synthetic = arguments["synthetic_features"]
    labels = np.repeat(np.arange(5, dtype=np.int64), 2)
    fid = {
        "pooled": feature_fid(real, synthetic),
        "class_fid": complete_class_fid(
            real, synthetic, real_labels=labels,
            synthetic_labels=labels, class_count=5,
        ),
    }
    return kid_arguments, fid


def verify(arguments, fid, **overrides):
    raw = canonical(fid) + b"\n"
    kwargs = {
        "result_raw": raw,
        "expected_result_file_sha256": sha(raw),
        "kid_verification_arguments": arguments,
    }
    kwargs.update(overrides)
    return verify_fid_result(**kwargs)


def test_complete_pooled_and_all_class_replay():
    result = verify(*fixture())
    assert result["pooled_fid_replay_verified"] is True
    assert result["all_class_fid_replay_verified"] is True
    assert result["feature_sets_shared_with_kid"] is True
    assert result["class_count"] == 5
    assert result["samples_per_class"] == 2
    assert result["provenance_validation"] == "NOT_PERFORMED"
    assert result["complete_evaluation"] is False


@pytest.mark.parametrize("fault", [
    "pooled", "class_value", "missing_class", "macro",
    "regularization", "count", "coverage", "extra",
])
def test_rehashed_fid_tampering_rejected(fault):
    arguments, fid = fixture()
    if fault == "pooled":
        fid["pooled"]["fid"] += 1
    elif fault == "class_value":
        fid["class_fid"]["per_class"]["0"]["fid"] += 1
    elif fault == "missing_class":
        del fid["class_fid"]["per_class"]["0"]
    elif fault == "macro":
        fid["class_fid"]["macro_cfid"] += 1
    elif fault == "regularization":
        fid["pooled"]["diagonal_regularization"] = 1e-6
    elif fault == "count":
        fid["pooled"]["real_count"] -= 1
    elif fault == "coverage":
        fid["class_fid"]["class_coverage_complete"] = False
    else:
        fid["complete_evaluation"] = True
    with pytest.raises(ValueError):
        verify(arguments, fid)


def test_changed_features_cannot_bypass_kid_binding():
    arguments, fid = fixture()
    arguments["real_features"][0, 0] += 1
    with pytest.raises(ValueError):
        verify(arguments, fid)


def test_rehashed_invalid_kid_report_rejected():
    arguments, fid = fixture()
    import json
    kid = json.loads(arguments["result_raw"])
    kid["kid"]["mean"] += 0.1
    raw = canonical(kid) + b"\n"
    arguments["result_raw"] = raw
    arguments["expected_result_file_sha256"] = sha(raw)
    with pytest.raises(ValueError):
        verify(arguments, fid)


def test_wrong_fid_file_identity_rejected():
    arguments, fid = fixture()
    with pytest.raises(ValueError):
        verify(arguments, fid, expected_result_file_sha256="0" * 64)


def test_verification_receipt_cannot_replace_kid_inputs():
    _, fid = fixture()
    with pytest.raises(TypeError):
        verify({"kid_replay_verified": True}, fid)
