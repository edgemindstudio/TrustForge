import hashlib
import json
from pathlib import Path
import runpy

import numpy as np
import pytest

from trustforge.evaluation.paper01_fidelity_inventory import (
    verify_fidelity_inventory,
)
from trustforge.evaluation.paper01_fid import feature_fid, complete_class_fid
from trustforge.evaluation.paper01_ms_ssim_pairs import select_ms_ssim_pairs
from trustforge.evaluation.paper01_status_inventory import REQUIRED_RESULTS


def helpers(name):
    return runpy.run_path(str(Path(__file__).with_name(name)))


KID = helpers("test_paper01_kid_result_validation.py")
MS = helpers("test_paper01_ms_ssim_result_validation.py")


def raw(value):
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def fixture():
    kid = KID["fixture"]()
    ms = MS["fixture"]()
    ki = kid["arguments"]
    mi = ms["arguments"]
    assert mi.pop("dataset_id") == "cicmaldroid2020"
    manifest = ki["expected_synthetic_manifest_sha256"]
    mi["expected_manifest_sha256"] = manifest
    selection_raw = raw(select_ms_ssim_pairs(
        labels=mi["labels"], class_count=5, manifest_sha256=manifest,
    ))
    mi["selection_raw"] = selection_raw
    mi["expected_selection_file_sha256"] = sha(selection_raw)
    ms["report"]["manifest_sha256"] = manifest
    ms["report"]["selection_file_sha256"] = sha(selection_raw)

    labels = np.repeat(np.arange(5, dtype=np.int64), 2)
    kid_raw = raw(kid["report"])
    config = {
        "selection_file_sha256": ki["expected_selection_file_sha256"],
        "feature_extractor": ki["expected_feature_extractor"],
    }
    fid_config = {**config, "kid_result_file_sha256": sha(kid_raw)}
    groups = {
        "pooled_balanced_kid": (kid["report"], config, 10),
        "pooled_balanced_fid": (
            feature_fid(ki["real_features"], ki["synthetic_features"]),
            fid_config, 1,
        ),
        "class_fid": (
            complete_class_fid(
                ki["real_features"], ki["synthetic_features"],
                real_labels=labels, synthetic_labels=labels, class_count=5,
            ),
            fid_config, 5,
        ),
        "intra_class_ms_ssim": (
            ms["report"], {
                "selection_file_sha256": sha(selection_raw),
                "scorer": mi["expected_scorer_identity"],
            }, 1000,
        ),
    }
    records, artifacts = [], {}
    for name in REQUIRED_RESULTS:
        if name not in groups:
            records.append({
                "result_id": name, "status": "PENDING",
                "reason": "downstream execution pending",
                "result_file_sha256": None,
                "configuration_sha256": None, "coverage": None,
            })
            continue
        value, configuration, count = groups[name]
        result_raw, configuration_raw = raw(value), raw(configuration)
        records.append({
            "result_id": name, "status": "SUCCEEDED", "reason": None,
            "result_file_sha256": sha(result_raw),
            "configuration_sha256": sha(configuration_raw),
            "coverage": {"expected": count, "computed": count},
        })
        artifacts[name] = {
            "result_raw": result_raw, "configuration_raw": configuration_raw,
        }
    return {
        "dataset_id": "cicmaldroid2020", "experimental_seed": 42,
        "records": records, "artifact_bytes": artifacts,
        "expected_synthetic_manifest_sha256": manifest,
        "kid_inputs": ki, "ms_ssim_inputs": mi,
    }


def record(data, name):
    return next(r for r in data["records"] if r["result_id"] == name)


def pending(data, name):
    record(data, name).update(
        status="FAILED", reason="fixture execution interrupted",
        result_file_sha256=None, configuration_sha256=None, coverage=None,
    )
    del data["artifact_bytes"][name]


GROUPS = (
    "pooled_balanced_kid", "pooled_balanced_fid",
    "class_fid", "intra_class_ms_ssim",
)


def test_all_four_groups_verified_without_completion():
    result = verify_fidelity_inventory(**fixture())
    assert set(result["fidelity_diversity_verifications"]) == set(GROUPS)
    assert len(result["required_status_blockers"]) == 3
    assert result["semantic_validation_not_performed"] == []
    assert result["provenance_validation"] == "NOT_PERFORMED"
    assert result["image_score_recomputation"] == "NOT_PERFORMED"
    assert result["complete_evaluation"] is False


@pytest.mark.parametrize("name", GROUPS)
def test_changed_bytes_rejected(name):
    data = fixture()
    data["artifact_bytes"][name]["result_raw"] += b" "
    with pytest.raises(ValueError, match="digest"):
        verify_fidelity_inventory(**data)


@pytest.mark.parametrize("name", GROUPS)
def test_rehashed_semantic_tampering_rejected(name):
    data = fixture()
    value = json.loads(data["artifact_bytes"][name]["result_raw"])
    value["undeclared"] = True
    encoded = raw(value)
    data["artifact_bytes"][name]["result_raw"] = encoded
    record(data, name)["result_file_sha256"] = sha(encoded)
    with pytest.raises(ValueError):
        verify_fidelity_inventory(**data)


@pytest.mark.parametrize("name", GROUPS)
def test_wrong_but_internally_equal_coverage_rejected(name):
    data = fixture()
    record(data, name)["coverage"] = {"expected": 99, "computed": 99}
    with pytest.raises(ValueError, match="coverage"):
        verify_fidelity_inventory(**data)


@pytest.mark.parametrize("name", GROUPS)
def test_rehashed_configuration_tampering_rejected(name):
    data = fixture()
    encoded = raw({"unselected": True})
    data["artifact_bytes"][name]["configuration_raw"] = encoded
    record(data, name)["configuration_sha256"] = sha(encoded)
    with pytest.raises(ValueError, match="configuration"):
        verify_fidelity_inventory(**data)


@pytest.mark.parametrize("which", ["kid_inputs", "ms_ssim_inputs"])
def test_shared_manifest_mismatch_rejected(which):
    data = fixture()
    key = (
        "expected_synthetic_manifest_sha256" if which == "kid_inputs"
        else "expected_manifest_sha256"
    )
    data[which][key] = "f" * 64
    with pytest.raises(ValueError, match="manifest"):
        verify_fidelity_inventory(**data)


def test_dataset_mismatch_rejected():
    data = fixture()
    data["kid_inputs"]["expected_real_validation_identity"]["dataset_id"] = "malex"
    with pytest.raises(ValueError, match="dataset"):
        verify_fidelity_inventory(**data)


def test_fid_requires_successful_kid_evidence():
    data = fixture()
    pending(data, "pooled_balanced_kid")
    data["kid_inputs"] = None
    with pytest.raises(ValueError, match="successful KID"):
        verify_fidelity_inventory(**data)


@pytest.mark.parametrize("name", ["pooled_balanced_fid", "class_fid"])
def test_one_failed_fid_preserves_other_successes(name):
    data = fixture()
    pending(data, name)
    result = verify_fidelity_inventory(**data)
    assert set(result["fidelity_diversity_verifications"]) == set(GROUPS) - {name}
    assert len(result["required_status_blockers"]) == 4
    assert result["complete_evaluation"] is False


def test_failed_ms_ssim_needs_no_scores():
    data = fixture()
    pending(data, "intra_class_ms_ssim")
    data["ms_ssim_inputs"] = None
    result = verify_fidelity_inventory(**data)
    assert len(result["fidelity_diversity_verifications"]) == 3
    assert result["complete_evaluation"] is False


def test_all_pending_requires_no_metric_arrays():
    data = fixture()
    for name in GROUPS:
        pending(data, name)
    data["kid_inputs"] = data["ms_ssim_inputs"] = None
    result = verify_fidelity_inventory(**data)
    assert result["fidelity_diversity_verifications"] == {}
    assert len(result["required_status_blockers"]) == 7
    assert result["complete_evaluation"] is False
