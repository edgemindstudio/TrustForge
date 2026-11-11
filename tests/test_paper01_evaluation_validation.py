from copy import deepcopy
import hashlib
import json
from pathlib import Path
import runpy

import numpy as np
import pytest

from trustforge.evaluation.paper01_evaluation_validation import verify_evaluation
from trustforge.evaluation.paper01_fid import feature_fid, complete_class_fid
from trustforge.evaluation.paper01_ms_ssim_pairs import select_ms_ssim_pairs
from trustforge.evaluation.paper01_paired_results import link_paired_results
from trustforge.evaluation.paper01_status_inventory import REQUIRED_RESULTS


def helpers(name):
    return runpy.run_path(str(Path(__file__).with_name(name)))


PAIRED = helpers("test_paper01_paired_results.py")
KID = helpers("test_paper01_kid_result_validation.py")
MS = helpers("test_paper01_ms_ssim_result_validation.py")

DOWNSTREAM_GROUPS = (
    "real_only_downstream", "real_plus_synthetic_downstream", "paired_downstream",
)
FIDELITY_GROUPS = (
    "pooled_balanced_kid", "pooled_balanced_fid",
    "class_fid", "intra_class_ms_ssim",
)
CONSUMPTION_BLOCKER = {
    "evidence": "augmented_pool_consumption",
    "status": "BLOCKED",
    "reason": (
        "augmented-arm consumption of the declared synthetic pool "
        "remains unverified"
    ),
}


def raw(value):
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def fixture():
    """One coherent synthetic-fixture evaluation with all seven groups."""
    downstream_arguments = {
        "arms": PAIRED["arms"](),
        "evaluator_seed": 42,
        "expected_test_identity": deepcopy(PAIRED["TEST"]),
        "expected_initial_weights_sha256": PAIRED["INITIAL"],
        "classifier_recipe_raw": PAIRED["CLASSIFIER"],
        "metric_recipe_raw": PAIRED["METRIC"],
    }
    paired = link_paired_results(**downstream_arguments)

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

    kid_raw = raw(kid["report"])
    config = {
        "selection_file_sha256": ki["expected_selection_file_sha256"],
        "feature_extractor": ki["expected_feature_extractor"],
    }
    labels = np.repeat(np.arange(5, dtype=np.int64), 2)
    groups = {
        "real_only_downstream": (
            paired["arms"]["real_only"]["prediction"]["metrics"],
            PAIRED["METRIC"], PAIRED["TEST"]["size"],
        ),
        "real_plus_synthetic_downstream": (
            paired["arms"]["real_plus_synthetic"]["prediction"]["metrics"],
            PAIRED["METRIC"], PAIRED["TEST"]["size"],
        ),
        "paired_downstream": (paired, raw({
            "classifier_recipe_sha256": paired["classifier_recipe_sha256"],
            "metric_recipe_sha256": paired["metric_recipe_sha256"],
        }), 2),
        "pooled_balanced_kid": (kid["report"], raw(config), 10),
        "pooled_balanced_fid": (
            feature_fid(ki["real_features"], ki["synthetic_features"]),
            raw({**config, "kid_result_file_sha256": sha(kid_raw)}), 1,
        ),
        "class_fid": (
            complete_class_fid(
                ki["real_features"], ki["synthetic_features"],
                real_labels=labels, synthetic_labels=labels, class_count=5,
            ),
            raw({**config, "kid_result_file_sha256": sha(kid_raw)}), 5,
        ),
        "intra_class_ms_ssim": (ms["report"], raw({
            "selection_file_sha256": sha(selection_raw),
            "scorer": mi["expected_scorer_identity"],
        }), 1000),
    }
    assert set(groups) == set(REQUIRED_RESULTS)
    records, artifacts = [], {}
    for name in sorted(groups):
        value, configuration_raw, count = groups[name]
        result_raw = raw(value)
        records.append({
            "result_id": name, "status": "SUCCEEDED", "reason": None,
            "result_file_sha256": sha(result_raw),
            "configuration_sha256": sha(configuration_raw),
            "coverage": {"expected": count, "computed": count},
        })
        artifacts[name] = {
            "result_raw": result_raw, "configuration_raw": configuration_raw,
        }
    augmented = downstream_arguments["arms"]["real_plus_synthetic"]
    return {
        "dataset_id": "cicmaldroid2020", "experimental_seed": 42,
        "records": records, "artifact_bytes": artifacts,
        "downstream_arguments": downstream_arguments,
        "fidelity_arguments": {
            "expected_synthetic_manifest_sha256": manifest,
            "kid_inputs": ki, "ms_ssim_inputs": mi,
        },
        "augmented_pool_linkage": {
            "arm": "real_plus_synthetic",
            "dataset_id": "cicmaldroid2020", "experimental_seed": 42,
            "family": "maf", "manifest_sha256": manifest,
            "checkpoint_sha256": "e" * 64,
            "selected_weights_sha256": augmented["predictions"]["report"][
                "selected_weights_sha256"
            ],
        },
    }


def record(data, name):
    return next(item for item in data["records"] if item["result_id"] == name)


def withdraw(data, name, status="PENDING"):
    record(data, name).update(
        status=status, reason="fixture execution did not produce an artifact",
        result_file_sha256=None, configuration_sha256=None, coverage=None,
    )
    del data["artifact_bytes"][name]


def withdraw_downstream(data):
    for name in DOWNSTREAM_GROUPS:
        withdraw(data, name)
    data["downstream_arguments"]["arms"] = {
        "real_only": None, "real_plus_synthetic": None,
    }


def test_composition_verifies_groups_without_completion():
    data = fixture()
    before = deepcopy(data["records"])
    result = verify_evaluation(**data)
    assert set(result["verified_groups"]) == set(REQUIRED_RESULTS)
    assert result["verified_group_count"] == 7
    assert result["required_status_blockers"] == []
    assert result["lineage_blockers"] == [CONSUMPTION_BLOCKER]
    assert result["declared_pool_linkage"]["family"] == "maf"
    assert result["declared_pool_linkage"]["identity_agreement"] == "REPORTED"
    assert result["synthetic_manifest_sha256"] == (
        data["fidelity_arguments"]["expected_synthetic_manifest_sha256"]
    )
    assert result["semantic_validation_not_performed"] == []
    assert result["provenance_validation"] == "NOT_PERFORMED"
    assert result["feature_extraction_authentication"] == "NOT_PERFORMED"
    assert result["publication_validation"] == "NOT_PERFORMED"
    assert result["complete_evaluation"] is False
    assert data["records"] == before


def test_missing_augmented_linkage_keeps_consumption_blocker():
    data = fixture()
    data["augmented_pool_linkage"] = None
    result = verify_evaluation(**data)
    assert result["verified_group_count"] == 7
    assert result["required_status_blockers"] == []
    assert result["lineage_blockers"] == [CONSUMPTION_BLOCKER]
    assert result["declared_pool_linkage"] is None
    assert result["complete_evaluation"] is False


def test_matching_declaration_does_not_clear_consumption_blocker():
    data = fixture()
    result = verify_evaluation(**data)
    assert result["declared_pool_linkage"] is not None
    assert result["lineage_blockers"] == [CONSUMPTION_BLOCKER]
    assert result["complete_evaluation"] is False


@pytest.mark.parametrize("field,value", [
    ("dataset_id", "malex"),
    ("experimental_seed", 43),
    ("manifest_sha256", "f" * 64),
    ("checkpoint_sha256", "not-a-digest"),
    ("selected_weights_sha256", "0" * 64),
    ("family", " "),
    ("arm", "real_only"),
])
def test_conflicting_pool_linkage_rejected(field, value):
    data = fixture()
    data["augmented_pool_linkage"][field] = value
    with pytest.raises(ValueError, match="pool linkage"):
        verify_evaluation(**data)


def test_incomplete_pool_linkage_fields_rejected():
    data = fixture()
    del data["augmented_pool_linkage"]["checkpoint_sha256"]
    with pytest.raises(ValueError, match="pool linkage fields"):
        verify_evaluation(**data)


def test_dataset_conflict_rejected():
    data = fixture()
    data["dataset_id"] = "malex"
    with pytest.raises(ValueError, match="held-out test identity"):
        verify_evaluation(**data)


def test_held_out_split_conflict_rejected():
    data = fixture()
    data["downstream_arguments"]["expected_test_identity"]["split"] = "train"
    with pytest.raises(ValueError, match="held-out test identity"):
        verify_evaluation(**data)


def test_evaluator_seed_conflict_rejected():
    data = fixture()
    data["experimental_seed"] = 43
    with pytest.raises(ValueError, match="identities"):
        verify_evaluation(**data)


def test_manifest_identity_conflict_rejected():
    data = fixture()
    data["fidelity_arguments"]["expected_synthetic_manifest_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="manifest"):
        verify_evaluation(**data)


def test_single_verified_arm_preserved_with_linkage():
    data = fixture()
    withdraw(data, "real_only_downstream", "FAILED")
    withdraw(data, "paired_downstream")
    data["downstream_arguments"]["arms"]["real_only"] = None
    result = verify_evaluation(**data)
    assert set(result["verified_groups"]) == {
        "real_plus_synthetic_downstream", *FIDELITY_GROUPS,
    }
    blockers = {
        item["result_id"]: item["status"]
        for item in result["required_status_blockers"]
    }
    assert blockers == {
        "real_only_downstream": "FAILED", "paired_downstream": "PENDING",
    }
    assert result["lineage_blockers"] == [CONSUMPTION_BLOCKER]
    assert result["declared_pool_linkage"] is not None
    assert result["complete_evaluation"] is False


def test_single_verified_arm_without_linkage_keeps_both_blockers():
    data = fixture()
    withdraw(data, "real_only_downstream", "FAILED")
    withdraw(data, "paired_downstream")
    data["downstream_arguments"]["arms"]["real_only"] = None
    data["augmented_pool_linkage"] = None
    result = verify_evaluation(**data)
    assert len(result["required_status_blockers"]) == 2
    assert result["lineage_blockers"] == [CONSUMPTION_BLOCKER]
    assert result["declared_pool_linkage"] is None
    assert result["complete_evaluation"] is False


def test_neither_arm_available_still_verifies_fidelity():
    data = fixture()
    withdraw_downstream(data)
    data["augmented_pool_linkage"] = None
    result = verify_evaluation(**data)
    assert set(result["verified_groups"]) == set(FIDELITY_GROUPS)
    blockers = {
        item["result_id"] for item in result["required_status_blockers"]
    }
    assert blockers == set(DOWNSTREAM_GROUPS)
    assert result["lineage_blockers"] == []
    assert result["declared_pool_linkage"] is None
    assert result["complete_evaluation"] is False


def test_successful_downstream_without_predictions_rejected():
    data = fixture()
    data["downstream_arguments"]["arms"] = {
        "real_only": None, "real_plus_synthetic": None,
    }
    with pytest.raises(ValueError, match="prediction arrays"):
        verify_evaluation(**data)


def test_no_downstream_arguments_at_all_rejected_when_successful():
    data = fixture()
    data["downstream_arguments"] = None
    with pytest.raises(ValueError, match="prediction arrays"):
        verify_evaluation(**data)


def test_failed_fidelity_group_preserves_other_verifications():
    data = fixture()
    withdraw(data, "class_fid", "FAILED")
    result = verify_evaluation(**data)
    assert set(result["verified_groups"]) == set(REQUIRED_RESULTS) - {"class_fid"}
    assert result["lineage_blockers"] == [CONSUMPTION_BLOCKER]
    assert result["declared_pool_linkage"] is not None
    assert result["complete_evaluation"] is False


def test_all_groups_pending_yields_only_blockers():
    data = fixture()
    for name in REQUIRED_RESULTS:
        withdraw(data, name)
    data["downstream_arguments"]["arms"] = {
        "real_only": None, "real_plus_synthetic": None,
    }
    data["fidelity_arguments"]["kid_inputs"] = None
    data["fidelity_arguments"]["ms_ssim_inputs"] = None
    data["augmented_pool_linkage"] = None
    result = verify_evaluation(**data)
    assert result["verified_groups"] == {}
    assert len(result["required_status_blockers"]) == 7
    assert result["lineage_blockers"] == []
    assert result["complete_evaluation"] is False


def test_linkage_without_verified_arm_is_reported_as_declared():
    data = fixture()
    withdraw(data, "real_plus_synthetic_downstream", "FAILED")
    withdraw(data, "paired_downstream")
    data["downstream_arguments"]["arms"]["real_plus_synthetic"] = None
    result = verify_evaluation(**data)
    assert set(result["verified_groups"]) == {
        "real_only_downstream", *FIDELITY_GROUPS,
    }
    assert result["lineage_blockers"] == []
    declared = result["declared_pool_linkage"]
    assert declared is not None
    assert declared["identity_agreement"] == "REPORTED"
    assert declared["generator_lineage_authentication"] == "NOT_PERFORMED"
    assert result["complete_evaluation"] is False


@pytest.mark.parametrize("arms", [
    {"real_only": None},
    {"realonly": None, "real_plus_synthetic": None},
    {"real_only": None, "real_plus_synthetic": None, "extra_arm": None},
    None,
    "not-a-mapping",
])
def test_malformed_arm_keys_rejected_when_both_arms_none(arms):
    data = fixture()
    withdraw_downstream(data)
    data["downstream_arguments"]["arms"] = arms
    with pytest.raises(ValueError, match="exactly two named comparison arms"):
        verify_evaluation(**data)


def test_contradictory_seed_rejected_when_both_arms_none():
    data = fixture()
    withdraw_downstream(data)
    data["downstream_arguments"]["evaluator_seed"] = 43
    with pytest.raises(ValueError, match="identities"):
        verify_evaluation(**data)


@pytest.mark.parametrize("field,value", [
    ("dataset_id", "malex"), ("split", "train"), ("class_count", 2), ("size", 0),
])
def test_contradictory_test_identity_rejected_when_both_arms_none(field, value):
    data = fixture()
    withdraw_downstream(data)
    data["downstream_arguments"]["expected_test_identity"][field] = value
    with pytest.raises(ValueError, match="held-out test identity"):
        verify_evaluation(**data)


def test_missing_test_identity_key_rejected_when_both_arms_none():
    data = fixture()
    withdraw_downstream(data)
    del data["downstream_arguments"]["expected_test_identity"]
    with pytest.raises(ValueError, match="held-out test identity"):
        verify_evaluation(**data)


@pytest.mark.parametrize("value", [None, 42, "not-a-dict", ["test"]])
def test_non_dict_test_identity_rejected_when_both_arms_none(value):
    data = fixture()
    withdraw_downstream(data)
    data["downstream_arguments"]["expected_test_identity"] = value
    with pytest.raises(ValueError, match="held-out test identity"):
        verify_evaluation(**data)


def test_failed_downstream_records_allow_fidelity_without_predictions():
    data = fixture()
    for name in DOWNSTREAM_GROUPS:
        withdraw(data, name, "FAILED")
    data["downstream_arguments"]["arms"] = {
        "real_only": None, "real_plus_synthetic": None,
    }
    data["augmented_pool_linkage"] = None
    result = verify_evaluation(**data)
    assert set(result["verified_groups"]) == set(FIDELITY_GROUPS)
    assert {
        item["status"] for item in result["required_status_blockers"]
    } == {"FAILED"}
    assert result["lineage_blockers"] == []
    assert result["complete_evaluation"] is False
