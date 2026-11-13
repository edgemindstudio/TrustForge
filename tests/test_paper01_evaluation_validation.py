from copy import deepcopy
import hashlib
import json
from pathlib import Path
import runpy

import numpy as np
import pytest

from trustforge.evaluation.paper01_classifier_bundle import (
    publish_classifier_bundle,
)
from trustforge.evaluation.paper01_classifier_manifest import (
    prepare_classifier_manifest, canonical_manifest,
)
from trustforge.evaluation.paper01_classifier_state import decode_classifier_weights
import trustforge.evaluation.paper01_classifier_restoration as classifier_restoration
from trustforge.evaluation.paper01_evaluation_validation import verify_evaluation
from trustforge.evaluation.paper01_fid import feature_fid, complete_class_fid
from trustforge.evaluation.paper01_ms_ssim_pairs import select_ms_ssim_pairs
from trustforge.evaluation.paper01_paired_evidence import (
    write_paired_restoration_evidence,
)
from trustforge.evaluation.paper01_paired_results import link_paired_results
from trustforge.evaluation.paper01_status_inventory import REQUIRED_RESULTS


def helpers(name):
    return runpy.run_path(str(Path(__file__).with_name(name)))


PAIRED = helpers("test_paper01_paired_results.py")
KID = helpers("test_paper01_kid_result_validation.py")
MS = helpers("test_paper01_ms_ssim_result_validation.py")
MANIFEST = helpers("test_paper01_classifier_manifest.py")

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
        "augmented_pool_consumption": None,
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


def refresh_paired_artifact(data):
    """Re-link the paired artifact after changing the augmented training report."""
    paired = link_paired_results(**data["downstream_arguments"])
    result_raw = raw(paired)
    record(data, "paired_downstream")["result_file_sha256"] = sha(result_raw)
    data["artifact_bytes"]["paired_downstream"]["result_raw"] = result_raw


def training_identity(data, *, real_size=9):
    """Build the independent augmented training identity fixture."""
    manifest = data["fidelity_arguments"]["expected_synthetic_manifest_sha256"]
    return {
        "condition": "real_plus_synthetic",
        "real": {
            "dataset_id": "cicmaldroid2020", "split": "train",
            "size": real_size, "class_count": 5,
            "declared_provenance_sha256": "b" * 64,
        },
        "synthetic": {
            "schema_version": 1, "dataset_id": "cicmaldroid2020",
            "family": "maf", "experimental_seed": 42,
            "recipe_sha256": "1" * 64,
            "checkpoint_sha256": data["augmented_pool_linkage"][
                "checkpoint_sha256"
            ],
            "manifest_sha256": manifest,
        },
        "position_order": "real_then_synthetic",
    }


def consumption(
    data, *, selected=17, completed=20, real_size=9, synthetic_size=10000,
):
    """Attach a complete training history and matching consumption record."""
    arm = data["downstream_arguments"]["arms"]["real_plus_synthetic"]
    report = arm["training"]
    training_size = real_size + synthetic_size
    updates_per_epoch = (training_size + 127) // 128
    report.update({
        "selected_epoch_index": selected,
        "selected_validation_loss": 1.0,
        "completed_epochs": completed,
        "optimizer_updates": completed * updates_per_epoch,
        "stopping_outcome": "maximum_epochs",
        "history": [
            {
                "epoch_index": index,
                "training_loss": 1.0,
                "validation_loss": float(1 + abs(index - selected)),
                "training_examples": training_size,
                "validation_examples": 5,
                "optimizer_updates": (index + 1) * updates_per_epoch,
            }
            for index in range(completed)
        ],
    })
    real_report = data["downstream_arguments"]["arms"]["real_only"]["training"]
    real_updates_per_epoch = (real_size + 127) // 128
    real_selected = completed - 1
    real_report.update({
        "selected_epoch_index": real_selected,
        "selected_validation_loss": 1.0,
        "completed_epochs": completed,
        "optimizer_updates": completed * real_updates_per_epoch,
        "stopping_outcome": "maximum_epochs",
        "history": [
            {
                "epoch_index": index,
                "training_loss": 1.0,
                "validation_loss": float(1 + abs(index - real_selected)),
                "training_examples": real_size,
                "validation_examples": 5,
                "optimizer_updates": (index + 1) * real_updates_per_epoch,
            }
            for index in range(completed)
        ],
    })
    identity = training_identity(data, real_size=real_size)
    record = {
        "schema": "trustforge.paper01.augmented-pool-consumption.v1",
        "dataset_id": "cicmaldroid2020", "evaluator_seed": 42,
        "arm": "real_plus_synthetic",
        "training_identity": identity,
        "synthetic_examples_per_epoch": synthetic_size,
        "classifier_recipe_sha256": report["recipe_sha256"],
        "selected_weights_sha256": report["selected_weights_sha256"],
        "selected_epoch_index": selected,
        "completed_epochs": completed,
        "selected_prefix": {
            "optimizer_updates": report["history"][selected][
                "optimizer_updates"
            ],
            "synthetic_examples": (selected + 1) * synthetic_size,
            "stream_sha256": "a" * 64,
        },
        "final_prefix": {
            "optimizer_updates": report["optimizer_updates"],
            "synthetic_examples": completed * synthetic_size,
            "stream_sha256": "b" * 64,
        },
    }
    data["augmented_pool_consumption"] = record
    refresh_paired_artifact(data)
    return record


class RestoredClassifierModel:
    def __init__(self, weights):
        self.weights = [np.zeros_like(value) for value in weights]

    def get_weights(self):
        return [value.copy() for value in self.weights]

    def set_weights(self, values):
        self.weights = [value.copy() for value in values]


def checkpoint_evidence(data, tmp_path):
    """Publish a checkpoint bundle and return its verified external evidence."""
    identity = training_identity(data)
    context, report, state = MANIFEST["fixture"]()
    report["initial_weights_sha256"] = "a" * 64
    arm = data["downstream_arguments"]["arms"]["real_plus_synthetic"]
    arm["training"] = report
    arm["predictions"]["report"]["selected_weights_sha256"] = report[
        "selected_weights_sha256"
    ]
    data["augmented_pool_linkage"]["selected_weights_sha256"] = report[
        "selected_weights_sha256"
    ]
    record = consumption(data, synthetic_size=129 - 9)
    context = {
        **context, "dataset_id": "cicmaldroid2020", "evaluator_seed": 42,
        "arm": "real_plus_synthetic",
        "training_sample_count": 129, "validation_sample_count": 5,
        "training_identity_sha256": sha(raw(identity)),
    }
    manifest = prepare_classifier_manifest(
        context=context, training_report=report,
        state_raw=state, recipe_raw=PAIRED["CLASSIFIER"],
    )
    manifest_raw = canonical_manifest(manifest)
    parent = tmp_path / "classifier-publications"
    parent.mkdir(mode=0o700)
    publish_classifier_bundle(
        parent, "real_plus_synthetic", manifest_raw=manifest_raw,
        state_raw=state, recipe_raw=PAIRED["CLASSIFIER"],
        expected_manifest_sha256=sha(manifest_raw), expected_context=context,
        expected_selected_weights_sha256=report["selected_weights_sha256"],
    )
    expected = {
        "context": context,
        "manifest_sha256": sha(manifest_raw),
        "selected_weights_sha256": report["selected_weights_sha256"],
    }
    evidence_dir = tmp_path / "paired-evidence"
    evidence_dir.mkdir(mode=0o700)
    written = write_paired_restoration_evidence(
        run_dir=evidence_dir,
        shared={
            "source_commit": "c" * 40,
            "source_files_sha256": "d" * 64,
            "environment_sha256": "e" * 64,
        },
        job_id="test-job", train_pid=1, evaluator_seed=42,
        initial_weights_sha256="a" * 64,
        classifier_recipe_sha256=sha(PAIRED["CLASSIFIER"]),
        metric_recipe_sha256=sha(PAIRED["METRIC"]),
        test_identity=deepcopy(PAIRED["TEST"]),
        fixture_pool_manifest_sha256=data["fidelity_arguments"][
            "expected_synthetic_manifest_sha256"
        ],
        real_identity=identity["real"],
        augmented_identity=identity,
        validation_identity={
            "dataset_id": "cicmaldroid2020", "split": "validation",
            "size": 5, "class_count": 5,
            "declared_provenance_sha256": "b" * 64,
        },
        sources={"script.py": "a" * 64},
        environment={"python": "test"},
        arms={"real_plus_synthetic": expected},
    )
    evidence = {
        "directory": parent / "real_plus_synthetic",
        "recipe_raw": PAIRED["CLASSIFIER"],
        "expectations_raw": written["receipt_raw"],
        "expected_expectations_sha256": written["receipt_sha256"],
        "context_evidence_raw": written["context_evidence_raw"],
    }
    return {
        "evidence": evidence,
        "record": record,
        "report": report,
        "context": context,
        "manifest_raw": manifest_raw,
        "state_raw": state,
        "expected": expected,
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


def test_self_consistent_consumption_record_cannot_clear_blocker_alone():
    data = fixture()
    record = consumption(data)
    assert record["selected_epoch_index"] < record["completed_epochs"]
    result = verify_evaluation(**data)
    assert set(result["verified_groups"]) == set(REQUIRED_RESULTS)
    assert result["verified_group_count"] == 7
    assert result["required_status_blockers"] == []
    assert len(result["lineage_blockers"]) == 1
    blocker = result["lineage_blockers"][0]
    assert blocker["evidence"] == "augmented_pool_consumption"
    assert blocker["status"] == "BLOCKED"
    assert "checkpoint evidence not supplied" in blocker["reason"]
    assert result["declared_pool_linkage"] is not None
    assert result["verified_pool_consumption"] is None
    assert result["pool_consumption_gap"] is not None
    assert result["provenance_validation"] == "NOT_PERFORMED"
    assert result["complete_evaluation"] is False


def test_verified_checkpoint_round_trip_clears_consumption_blocker(
    tmp_path, monkeypatch,
):
    data = fixture()
    built = checkpoint_evidence(data, tmp_path)
    expected = built["expected"]
    weights = decode_classifier_weights(
        raw=built["state_raw"], dataset_id="cicmaldroid2020",
        recipe_raw=PAIRED["CLASSIFIER"],
        expected_selected_weights_sha256=built["report"][
            "selected_weights_sha256"
        ],
    )
    model = RestoredClassifierModel(weights)
    monkeypatch.setattr(
        classifier_restoration, "require_compute_node", lambda: None,
    )
    monkeypatch.setattr(
        classifier_restoration, "build_classifier_binding",
        lambda **kwargs: {
            "model": model, "controls": {"optimizer_updates": lambda: 0},
        },
    )
    restored = classifier_restoration.restore_selected_classifier(
        built["evidence"]["directory"],
        recipe_raw=built["evidence"]["recipe_raw"],
        expected_manifest_sha256=expected["manifest_sha256"],
        expected_context=expected["context"],
        expected_selected_weights_sha256=expected["selected_weights_sha256"],
    )
    assert restored["report"]["selected_weights_sha256"] == (
        built["report"]["selected_weights_sha256"]
    )
    data["augmented_checkpoint_evidence"] = built["evidence"]
    result = verify_evaluation(**data)
    assert set(result["verified_groups"]) == set(REQUIRED_RESULTS)
    assert result["verified_group_count"] == 7
    assert result["lineage_blockers"] == []
    assert result["verified_pool_consumption"] is not None
    assert result["verified_pool_consumption"]["selected_weights_sha256"] == (
        restored["report"]["selected_weights_sha256"]
    )
    assert result["pool_consumption_gap"] is None
    assert result["complete_evaluation"] is False


def test_conflicting_checkpoint_identity_rejected(tmp_path):
    data = fixture()
    built = checkpoint_evidence(data, tmp_path)
    built["record"]["training_identity"]["real"][
        "declared_provenance_sha256"
    ] = "c" * 64
    data["augmented_checkpoint_evidence"] = built["evidence"]
    with pytest.raises(ValueError, match="binding mismatch"):
        verify_evaluation(**data)


@pytest.mark.parametrize("legacy", ["receipt_schema", "context_identity"])
def test_legacy_checkpoint_evidence_keeps_consumption_blocker(
    tmp_path, legacy,
):
    data = fixture()
    built = checkpoint_evidence(data, tmp_path)
    evidence = dict(built["evidence"])
    receipt = json.loads(evidence["expectations_raw"])
    if legacy == "receipt_schema":
        receipt["schema"] = "trustforge.paper01.legacy-expectations.v1"
        expected_reason = "legacy or unsupported"
    else:
        legacy_context = raw({"legacy": True})
        receipt["context_evidence_sha256"] = sha(legacy_context)
        evidence["context_evidence_raw"] = legacy_context
        expected_reason = "lacks augmented training identity"
    evidence["expectations_raw"] = raw(receipt)
    evidence["expected_expectations_sha256"] = sha(evidence["expectations_raw"])
    data["augmented_checkpoint_evidence"] = evidence
    result = verify_evaluation(**data)
    assert set(result["verified_groups"]) == set(REQUIRED_RESULTS)
    assert len(result["lineage_blockers"]) == 1
    blocker = result["lineage_blockers"][0]
    assert blocker["evidence"] == "augmented_pool_consumption"
    assert blocker["status"] == "BLOCKED"
    assert expected_reason in blocker["reason"]
    assert result["verified_pool_consumption"] is None
    assert result["complete_evaluation"] is False


def test_missing_consumption_record_preserves_verified_results():
    data = fixture()
    result = verify_evaluation(**data)
    assert set(result["verified_groups"]) == set(REQUIRED_RESULTS)
    assert result["lineage_blockers"] == [CONSUMPTION_BLOCKER]
    assert result["pool_consumption_gap"] is not None
    assert result["complete_evaluation"] is False


def test_missing_training_size_retains_explicit_identity_gap():
    data = fixture()
    consumption(data)
    del data["downstream_arguments"]["arms"]["real_plus_synthetic"][
        "training"
    ]["history"]
    refresh_paired_artifact(data)
    result = verify_evaluation(**data)
    assert set(result["verified_groups"]) == set(REQUIRED_RESULTS)
    assert len(result["lineage_blockers"]) == 1
    blocker = result["lineage_blockers"][0]
    assert blocker["evidence"] == "augmented_pool_consumption"
    assert blocker["status"] == "BLOCKED"
    assert "independent augmented training identity" in blocker["reason"]
    assert result["pool_consumption_gap"] is not None
    assert result["complete_evaluation"] is False


@pytest.mark.parametrize("mutation,match", [
    (
        lambda record: record["training_identity"]["synthetic"].__setitem__(
            "manifest_sha256", "f" * 64,
        ),
        "pool identity",
    ),
    (
        lambda record: record.__setitem__("selected_weights_sha256", "0" * 64),
        "selected weights",
    ),
    (
        lambda record: record.__setitem__("selected_epoch_index", 16),
        "selected epoch",
    ),
    (
        lambda record: record["selected_prefix"].__setitem__(
            "synthetic_examples", 1,
        ),
        "selected-epoch counts",
    ),
    (
        lambda record: record["final_prefix"].__setitem__(
            "optimizer_updates", 1,
        ),
        "final optimizer updates",
    ),
])
def test_conflicting_consumption_record_rejected(mutation, match):
    data = fixture()
    record = consumption(data)
    mutation(record)
    with pytest.raises(ValueError, match=match):
        verify_evaluation(**data)


def test_conflicting_real_training_identity_rejected():
    data = fixture()
    record = consumption(data)
    record["training_identity"]["real"]["size"] = 8
    record["synthetic_examples_per_epoch"] = 10001
    record["selected_prefix"]["synthetic_examples"] = 18 * 10001
    record["final_prefix"]["synthetic_examples"] = 20 * 10001
    with pytest.raises(ValueError, match="real training identity"):
        verify_evaluation(**data)


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


@pytest.mark.parametrize("status", ("PENDING", "FAILED"))
def test_unverified_real_report_cannot_drive_consumption_count_conflict(status):
    data = fixture()
    consumption(data)
    withdraw(data, "real_only_downstream", status)
    withdraw(data, "paired_downstream")
    real_report = data["downstream_arguments"]["arms"]["real_only"]["training"]
    for row in real_report["history"]:
        row["training_examples"] = 7
    result = verify_evaluation(**data)
    assert set(result["verified_groups"]) == {
        "real_plus_synthetic_downstream", *FIDELITY_GROUPS,
    }
    blockers = {
        item["result_id"]: item["status"]
        for item in result["required_status_blockers"]
    }
    assert blockers == {
        "real_only_downstream": status, "paired_downstream": "PENDING",
    }
    assert len(result["lineage_blockers"]) == 1
    assert result["lineage_blockers"][0]["evidence"] == (
        "augmented_pool_consumption"
    )
    assert "independent augmented training identity" in (
        result["lineage_blockers"][0]["reason"]
    )
    assert result["pool_consumption_gap"] is not None
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
