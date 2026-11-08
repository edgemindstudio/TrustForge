from copy import deepcopy
import hashlib
import json
from pathlib import Path
import runpy

import pytest

from trustforge.evaluation.paper01_downstream_inventory import (
    verify_downstream_inventory,
)
from trustforge.evaluation.paper01_paired_results import link_paired_results
from trustforge.evaluation.paper01_status_inventory import REQUIRED_RESULTS


HELPERS = runpy.run_path(str(
    Path(__file__).with_name("test_paper01_paired_results.py")
))


def raw(value):
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def fixture():
    paired_arguments = {
        "arms": HELPERS["arms"](),
        "evaluator_seed": 42,
        "expected_test_identity": deepcopy(HELPERS["TEST"]),
        "expected_initial_weights_sha256": HELPERS["INITIAL"],
        "classifier_recipe_raw": HELPERS["CLASSIFIER"],
        "metric_recipe_raw": HELPERS["METRIC"],
    }
    paired = link_paired_results(**paired_arguments)
    records = []
    artifacts = {}
    for name in REQUIRED_RESULTS:
        if name in ("real_only_downstream", "real_plus_synthetic_downstream"):
            arm = name.removesuffix("_downstream")
            result = paired["arms"][arm]["prediction"]["metrics"]
            configuration_raw = HELPERS["METRIC"]
            count = HELPERS["TEST"]["size"]
        elif name == "paired_downstream":
            result = paired
            configuration_raw = raw({
                "classifier_recipe_sha256": paired["classifier_recipe_sha256"],
                "metric_recipe_sha256": paired["metric_recipe_sha256"],
            })
            count = 2
        else:
            records.append({
                "result_id": name, "status": "PENDING",
                "reason": "fidelity/diversity execution pending",
                "result_file_sha256": None,
                "configuration_sha256": None, "coverage": None,
            })
            continue
        result_raw = raw(result)
        records.append({
            "result_id": name, "status": "SUCCEEDED", "reason": None,
            "result_file_sha256": sha(result_raw),
            "configuration_sha256": sha(configuration_raw),
            "coverage": {"expected": count, "computed": count},
        })
        artifacts[name] = {
            "result_raw": result_raw,
            "configuration_raw": configuration_raw,
        }
    return {
        "dataset_id": "cicmaldroid2020",
        "experimental_seed": 42,
        "records": records,
        "artifact_bytes": artifacts,
        "paired_arguments": paired_arguments,
    }


def record(data, name):
    return next(item for item in data["records"] if item["result_id"] == name)


def replace(data, name, field, value):
    encoded = raw(value)
    data["artifact_bytes"][name][field] = encoded
    hash_field = (
        "result_file_sha256" if field == "result_raw"
        else "configuration_sha256"
    )
    record(data, name)[hash_field] = sha(encoded)


def test_integrated_replay_preserves_pending_groups_and_scope():
    data = fixture()
    before = deepcopy(data["records"])
    result = verify_downstream_inventory(**data)
    assert set(result["downstream_verifications"]) == {
        "real_only_downstream", "real_plus_synthetic_downstream",
        "paired_downstream",
    }
    assert len(result["required_status_blockers"]) == 4
    assert result["semantic_validation_not_performed"] == []
    assert result["provenance_validation"] == "NOT_PERFORMED"
    assert result["publication_validation"] == "NOT_PERFORMED"
    assert result["complete_evaluation"] is False
    assert data["records"] == before


@pytest.mark.parametrize("name", [
    "real_only_downstream", "real_plus_synthetic_downstream",
    "paired_downstream",
])
def test_changed_artifact_bytes_rejected(name):
    data = fixture()
    data["artifact_bytes"][name]["result_raw"] += b" "
    with pytest.raises(ValueError, match="digest"):
        verify_downstream_inventory(**data)


@pytest.mark.parametrize("name", [
    "real_only_downstream", "real_plus_synthetic_downstream",
    "paired_downstream",
])
def test_rehashed_semantic_tampering_rejected(name):
    data = fixture()
    value = json.loads(data["artifact_bytes"][name]["result_raw"])
    key = "macro_f1_difference" if name == "paired_downstream" else "macro_f1"
    value[key] += 0.1
    replace(data, name, "result_raw", value)
    with pytest.raises(ValueError):
        verify_downstream_inventory(**data)


@pytest.mark.parametrize("name", [
    "real_only_downstream", "real_plus_synthetic_downstream",
    "paired_downstream",
])
def test_internally_equal_but_wrong_coverage_rejected(name):
    data = fixture()
    record(data, name)["coverage"] = {"expected": 99, "computed": 99}
    with pytest.raises(ValueError, match="coverage"):
        verify_downstream_inventory(**data)


@pytest.mark.parametrize("name", [
    "real_only_downstream", "paired_downstream",
])
def test_rehashed_configuration_tampering_rejected(name):
    data = fixture()
    replace(data, name, "configuration_raw", {"unselected": True})
    with pytest.raises(ValueError):
        verify_downstream_inventory(**data)


@pytest.mark.parametrize("fault", ["dataset", "seed", "class_count", "size"])
def test_cross_result_identity_mismatch_rejected(fault):
    data = fixture()
    if fault == "seed":
        data["paired_arguments"]["evaluator_seed"] = 43
    else:
        test = data["paired_arguments"]["expected_test_identity"]
        key = "dataset_id" if fault == "dataset" else fault
        test[key] = {
            "dataset": "malex", "class_count": 2, "size": True,
        }[fault]
    with pytest.raises(ValueError):
        verify_downstream_inventory(**data)


def test_successful_pair_requires_both_successful_arm_artifacts():
    data = fixture()
    name = "real_only_downstream"
    record(data, name).update(
        status="FAILED", reason="saved arm artifact unavailable",
        result_file_sha256=None, configuration_sha256=None, coverage=None,
    )
    del data["artifact_bytes"][name]
    with pytest.raises(ValueError, match="both arm artifacts"):
        verify_downstream_inventory(**data)


def test_pending_pair_preserves_verified_arm_results():
    data = fixture()
    name = "paired_downstream"
    record(data, name).update(
        status="PENDING", reason="paired artifact publication pending",
        result_file_sha256=None, configuration_sha256=None, coverage=None,
    )
    del data["artifact_bytes"][name]
    result = verify_downstream_inventory(**data)
    assert len(result["downstream_verifications"]) == 2
    assert len(result["required_status_blockers"]) == 5
    assert result["complete_evaluation"] is False
