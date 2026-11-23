"""Fixture tests for authenticated production aggregation linkage.

No training, inference, feature extraction or Slurm execution occurs.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from trustforge.evaluation.paper01_augmented_checkpoint import (
    verify_augmented_checkpoint_identity,
)
from trustforge.evaluation.paper01_classifier_bundle import (
    publish_classifier_bundle,
)
from trustforge.evaluation.paper01_classifier_control import state_digest
from trustforge.evaluation.paper01_classifier_manifest import (
    canonical_manifest,
    prepare_classifier_manifest,
)
from trustforge.evaluation.paper01_classifier_state import (
    classifier_specification,
    encode_classifier_weights,
)
from trustforge.evaluation.paper01_downstream_metrics import downstream_metrics
from trustforge.evaluation.paper01_paired_results import link_paired_results
from trustforge.evaluation.paper01_production_aggregation import (
    CLASSIFIER_RECIPE_SHA256,
    AggregationError,
    build_aggregate,
    expected_cells,
    publish_aggregate,
    validate_inventory,
)
from trustforge.evaluation.paper01_seed_reporting import (
    summarize_declared_paired_seeds,
)


REPO = Path(__file__).resolve().parents[1]
CLASSIFIER_RAW = (
    REPO / "studies/paper01_benchmark/revision_2026/classifier_recipe.json"
).read_bytes()
METRIC_RAW = (
    REPO / "studies/paper01_benchmark/revision_2026/downstream_metric_recipe.json"
).read_bytes()
DATASET = "cicmaldroid2020"
FAMILY = "vae"
INITIAL = "4e5a33f021164e0d148bba5f058d0dc01a0cd8ca48ad3f0b7c07f9013e7a19ba"
RECEIPT_SCHEMA = "trustforge.paper01.paired-restoration-expectations.v1"
REAL_SIZE = 8
SYNTHETIC_SIZE = 20
TRAIN_SIZE = REAL_SIZE + SYNTHETIC_SIZE
VALIDATION_SIZE = 8
EPOCHS = 20
UPDATES_PER_EPOCH = (TRAIN_SIZE + 127) // 128


def pool_for(seed):
    return f"{seed:02d}" * 32


def _newline_canonical(value):
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _labels():
    return np.repeat(np.arange(5, dtype=np.int64), 4)


def _probabilities(errors):
    labels = _labels()
    values = np.zeros((len(labels), 5), dtype=np.float32)
    for index, label in enumerate(labels):
        if index < errors:
            values[index, (label + 1) % 5] = 0.7
            values[index, label] = 0.3
        else:
            values[index, label] = 0.9
            values[index, (label + 1) % 5] = 0.1
    return values


def _test_identity():
    return {
        "dataset_id": DATASET, "split": "test", "size": 20,
        "class_count": 5, "declared_provenance_sha256": "b" * 64,
    }


def _augmented_identity(seed, pool_sha):
    return {
        "condition": "real_plus_synthetic",
        "position_order": "real_then_synthetic",
        "real": {
            "dataset_id": DATASET, "split": "train", "size": REAL_SIZE,
            "class_count": 5, "declared_provenance_sha256": "b" * 64,
        },
        "synthetic": {
            "dataset_id": DATASET, "family": FAMILY,
            "experimental_seed": seed, "manifest_sha256": pool_sha,
            "class_count": 5, "quota_per_class": 2000,
        },
    }


def _training_report(seed, selected):
    history = [
        {
            "epoch_index": index,
            "training_loss": float(EPOCHS - index),
            "validation_loss": float(EPOCHS - index),
            "training_examples": TRAIN_SIZE,
            "validation_examples": VALIDATION_SIZE,
            "optimizer_updates": (index + 1) * UPDATES_PER_EPOCH,
        }
        for index in range(EPOCHS)
    ]
    return {
        "recipe_sha256": CLASSIFIER_RECIPE_SHA256,
        "evaluator_seed": seed,
        "initial_weights_sha256": INITIAL,
        "selected_weights_sha256": selected,
        "selected_epoch_index": EPOCHS - 1,
        "selected_validation_loss": history[-1]["validation_loss"],
        "completed_epochs": EPOCHS,
        "optimizer_updates": EPOCHS * UPDATES_PER_EPOCH,
        "stopping_outcome": "maximum_epochs",
        "history": history,
    }


def _weights(dataset):
    specification = classifier_specification(
        dataset_id=dataset, recipe_raw=CLASSIFIER_RAW,
    )
    return [
        np.full(entry["shape"], 0.01 * (index + 1), dtype=np.float32)
        for index, entry in enumerate(specification)
    ]


def _publish_bundle(run, seed, pool_sha):
    checkpoints = run / "checkpoints"
    checkpoints.mkdir()
    checkpoints.chmod(0o700)
    weights = _weights(DATASET)
    selected = state_digest(weights)
    report = _training_report(seed, selected)
    state_raw = encode_classifier_weights(
        weights=weights, dataset_id=DATASET, recipe_raw=CLASSIFIER_RAW,
        expected_selected_weights_sha256=selected,
    )
    identity = _augmented_identity(seed, pool_sha)
    context = {
        "dataset_id": DATASET,
        "evaluator_seed": seed,
        "arm": "real_plus_synthetic",
        "training_sample_count": TRAIN_SIZE,
        "validation_sample_count": VALIDATION_SIZE,
        "training_identity_sha256": _sha(_newline_canonical(identity)),
        "validation_identity_sha256": "b" * 64,
        "source_commit": "c" * 40,
        "source_files_sha256": "d" * 64,
        "environment_sha256": "e" * 64,
    }
    manifest = prepare_classifier_manifest(
        context=context, training_report=report, state_raw=state_raw,
        recipe_raw=CLASSIFIER_RAW,
    )
    manifest_raw = canonical_manifest(manifest)
    manifest_sha = _sha(manifest_raw)
    publish_classifier_bundle(
        checkpoints, "real_plus_synthetic",
        manifest_raw=manifest_raw, state_raw=state_raw,
        recipe_raw=CLASSIFIER_RAW,
        expected_manifest_sha256=manifest_sha,
        expected_context=context,
        expected_selected_weights_sha256=selected,
    )
    return {
        "identity": identity, "context": context, "report": report,
        "selected": selected, "manifest_sha": manifest_sha,
    }


def make_run(root, seed, *, augmented_errors=6, family=FAMILY,
             pool_sha=None):
    pool_sha = pool_for(seed) if pool_sha is None else pool_sha
    run = root / f"seed{seed}"
    run.mkdir(parents=True)
    bundle = _publish_bundle(run, seed, pool_sha)
    test = _test_identity()
    labels = _labels()
    selected = {
        "real_only": "1" * 64,
        "real_plus_synthetic": bundle["selected"],
    }
    arms, reports = {}, {}
    for arm, errors in (
        ("real_only", 8),
        ("real_plus_synthetic", augmented_errors),
    ):
        probabilities = _probabilities(errors)
        metrics = downstream_metrics(
            labels=labels, probabilities=probabilities,
            class_count=5, recipe_raw=METRIC_RAW,
        )
        prediction = {
            "schema": "trustforge.paper01.test-predictions.v1",
            "test_identity": test,
            "selected_weights_sha256": selected[arm],
            "inference_state_unchanged": True,
            "position_order":
                "complete_logical_positions_zero_through_size_minus_one",
            "sample_count": 20,
            "maximum_image_batch": 20,
            "ordered_prediction_state_sha256": state_digest(
                [labels, probabilities]
            ),
            "metrics": metrics,
        }
        np.savez(
            run / (arm + "_predictions.npz"),
            labels=labels, probabilities=probabilities,
        )
        if arm == "real_plus_synthetic":
            reports[arm] = bundle["report"]
        else:
            reports[arm] = {
                "evaluator_seed": seed,
                "recipe_sha256": CLASSIFIER_RECIPE_SHA256,
                "initial_weights_sha256": INITIAL,
                "selected_weights_sha256": selected[arm],
            }
        arms[arm] = {"training": reports[arm], "predictions": {
            "report": prediction, "labels": labels,
            "probabilities": probabilities,
        }}
    paired = link_paired_results(
        arms=arms, evaluator_seed=seed, expected_test_identity=test,
        expected_initial_weights_sha256=INITIAL,
        classifier_recipe_raw=CLASSIFIER_RAW,
        metric_recipe_raw=METRIC_RAW,
    )
    (run / "paired_downstream.json").write_text(
        json.dumps(paired, sort_keys=True), encoding="utf-8",
    )

    receipt = {
        "schema": RECEIPT_SCHEMA,
        "evaluator_seed": seed,
        "classifier_recipe_sha256": CLASSIFIER_RECIPE_SHA256,
        "arms": {
            "real_plus_synthetic": {
                "context": bundle["context"],
                "manifest_sha256": bundle["manifest_sha"],
                "selected_weights_sha256": bundle["selected"],
            },
        },
    }
    context_raw = _newline_canonical({
        "augmented_training_identity": bundle["identity"],
    })
    receipt["context_evidence_sha256"] = _sha(context_raw)
    receipt_raw = json.dumps(receipt, sort_keys=True).encode("utf-8")
    (run / "expectations.json").write_bytes(receipt_raw)
    (run / "context_evidence.json").write_bytes(context_raw)

    consumption = {
        "schema": "trustforge.paper01.augmented-pool-consumption.v1",
        "dataset_id": DATASET,
        "evaluator_seed": seed,
        "arm": "real_plus_synthetic",
        "training_identity": bundle["identity"],
        "synthetic_examples_per_epoch": SYNTHETIC_SIZE,
        "classifier_recipe_sha256": CLASSIFIER_RECIPE_SHA256,
        "selected_weights_sha256": bundle["selected"],
        "selected_epoch_index": EPOCHS - 1,
        "completed_epochs": EPOCHS,
        "selected_prefix": {
            "optimizer_updates": EPOCHS * UPDATES_PER_EPOCH,
            "synthetic_examples": EPOCHS * SYNTHETIC_SIZE,
            "stream_sha256": "a" * 64,
        },
        "final_prefix": {
            "optimizer_updates": EPOCHS * UPDATES_PER_EPOCH,
            "synthetic_examples": EPOCHS * SYNTHETIC_SIZE,
            "stream_sha256": "a" * 64,
        },
    }
    (run / "pool_consumption.json").write_bytes(
        _newline_canonical(consumption)
    )

    summary, gap = verify_augmented_checkpoint_identity(
        dataset_id=DATASET, experimental_seed=seed,
        expected_synthetic_manifest_sha256=pool_sha,
        directory=run / "checkpoints" / "real_plus_synthetic",
        recipe_raw=CLASSIFIER_RAW,
        expectations_raw=receipt_raw,
        expected_expectations_sha256=_sha(receipt_raw),
        context_evidence_raw=context_raw,
    )
    assert gap is None

    evaluation = {
        "dataset_id": DATASET, "experimental_seed": seed,
        "synthetic_manifest_sha256": pool_sha,
        "verified_group_count": 7, "required_status_blockers": [],
        "lineage_blockers": [],
        "verified_pool_consumption": summary,
        "complete_evaluation": False,
    }
    (run / "evaluation.json").write_text(
        json.dumps(evaluation), encoding="utf-8",
    )
    restore = {
        "generator_lineage": {
            "dataset_id": DATASET, "family": family,
            "experimental_seed": seed, "pool_manifest_sha256": pool_sha,
        },
        "verified_pool_consumption": summary,
        "test_identity": test, "complete_evaluation": False,
        "full_benchmark_go": False, "job_id": 1, "pid": 2, "train_pid": 3,
    }
    (run / "restore_report.json").write_text(
        json.dumps(restore), encoding="utf-8",
    )
    train = {
        "arms": reports, "initial_weights_sha256": INITIAL,
        "test_identity": test, "test_accessed": False,
        "complete_evaluation": False, "full_benchmark_go": False,
    }
    (run / "train_report.json").write_text(
        json.dumps(train), encoding="utf-8",
    )
    return run


def attempt(run, number=1, status="SUCCEEDED", reason=None):
    return {
        "attempt": number, "run_dir": str(run),
        "declared_status": status, "reason": reason,
    }


def inventory(root, *, seeds=None, family=FAMILY):
    if seeds is None:
        seeds = {
            seed: {
                "pool_manifest_sha256": pool_for(seed),
                "attempts": [attempt(make_run(root, seed))],
            }
            for seed in range(42, 49)
        }
    return {
        "schema":
            "trustforge.paper01.production-aggregation-inventory.v1",
        "cells": [{
            "dataset_id": DATASET, "family": family,
            "seeds": [
                {
                    "experimental_seed": seed,
                    **seeds.get(seed, {
                        "pool_manifest_sha256": None, "attempts": [],
                    }),
                }
                for seed in range(42, 49)
            ],
        }],
    }


def build(root, **kwargs):
    return build_aggregate(
        inventory=inventory(root, **kwargs),
        classifier_recipe_raw=CLASSIFIER_RAW,
        metric_recipe_raw=METRIC_RAW,
    )


def test_expected_matrix_has_196_cells():
    cells = expected_cells()
    assert len(cells) == 4 * 7 * 7
    assert len(set(cells)) == 196


def test_seven_authenticated_pairs_invoke_accepted_reporter(tmp_path):
    document = build(tmp_path)
    cell = document["cells"][0]
    assert cell["authenticated_success_count"] == 7
    assert cell["summary"]["seven_pair_summary"] is not None
    assert cell["summary"]["seven_pair_summary_status"] == (
        "ALL_SEVEN_DECLARED_PAIRS_PRESENT"
    )
    assert all(record["status"] == "SUCCEEDED" for record in cell["records"])
    assert all(
        record["real_plus_synthetic_macro_f1"]
        > record["real_only_macro_f1"]
        for record in cell["records"]
    )
    pools = {
        item["pool_manifest_sha256"] for item in cell["attempt_evidence"]
    }
    assert pools == {pool_for(seed) for seed in range(42, 49)}
    assert cell["benchmark_complete"] is False
    assert document["full_benchmark_go"] is False


def test_cross_seed_pool_substitution_rejected(tmp_path):
    seeds = {
        seed: {
            "pool_manifest_sha256": pool_for(seed),
            "attempts": [attempt(make_run(tmp_path, seed))],
        }
        for seed in range(42, 49)
    }
    seeds[43] = {
        "pool_manifest_sha256": pool_for(43),
        "attempts": [attempt(make_run(
            tmp_path / "wrong", 43, pool_sha=pool_for(42),
        ))],
    }
    document = build(tmp_path, seeds=seeds)
    record = document["cells"][0]["records"][1]
    assert record["experimental_seed"] == 43
    assert record["status"] == "FAILED"
    evidence = document["cells"][0]["attempt_evidence"][1]
    assert evidence["attempts"][0]["authentication"] == "FAILED"


def test_pending_seed_needs_no_pool_identity(tmp_path):
    seeds = {
        42: {
            "pool_manifest_sha256": pool_for(42),
            "attempts": [attempt(make_run(tmp_path, 42))],
        },
    }
    document = build(tmp_path, seeds=seeds)
    cell = document["cells"][0]
    assert cell["authenticated_success_count"] == 1
    for evidence in cell["attempt_evidence"][1:]:
        assert evidence["pool_manifest_sha256"] is None
    assert cell["summary"]["seven_pair_summary"] is None


def test_failed_seed_needs_no_pool_identity(tmp_path):
    seeds = {
        seed: {
            "pool_manifest_sha256": pool_for(seed),
            "attempts": [attempt(make_run(tmp_path, seed))],
        }
        for seed in range(43, 49)
    }
    seeds[42] = {
        "pool_manifest_sha256": None,
        "attempts": [{
            "attempt": 1, "run_dir": str(tmp_path / "failed"),
            "declared_status": "FAILED", "reason": "preflight failed",
        }],
    }
    document = build(tmp_path, seeds=seeds)
    summary = document["cells"][0]["summary"]
    assert summary["declared_successful_pair_count"] == 6
    assert summary["status_counts"]["FAILED"] == 1
    assert summary["declared_failed_fraction"] == 1 / 7
    assert document["cells"][0]["records"][0]["status"] == "FAILED"


def test_success_without_pool_identity_rejected(tmp_path):
    seeds = {
        42: {"pool_manifest_sha256": None,
             "attempts": [attempt(tmp_path / "missing")]},
    }
    with pytest.raises(AggregationError, match="seed pool identity"):
        validate_inventory(inventory(tmp_path, seeds=seeds))


def test_declared_success_without_verification_evidence_not_promoted(
    tmp_path,
):
    seeds = {
        seed: {
            "pool_manifest_sha256": pool_for(seed),
            "attempts": [attempt(make_run(tmp_path, seed))],
        }
        for seed in range(43, 49)
    }
    bad = make_run(tmp_path / "bad", 42)
    for name in (
        "expectations.json", "context_evidence.json",
        "pool_consumption.json",
    ):
        (bad / name).unlink()
    seeds[42] = {
        "pool_manifest_sha256": pool_for(42),
        "attempts": [attempt(bad)],
    }
    document = build(tmp_path, seeds=seeds)
    record = document["cells"][0]["records"][0]
    assert record["status"] == "FAILED"
    assert record["real_only_macro_f1"] is None


def test_altered_consumption_artifact_rejected(tmp_path):
    seeds = {
        seed: {
            "pool_manifest_sha256": pool_for(seed),
            "attempts": [attempt(make_run(tmp_path, seed))],
        }
        for seed in range(43, 49)
    }
    bad = make_run(tmp_path / "bad", 42)
    record = json.loads((bad / "pool_consumption.json").read_text())
    record["synthetic_examples_per_epoch"] += 1
    (bad / "pool_consumption.json").write_text(json.dumps(record))
    seeds[42] = {
        "pool_manifest_sha256": pool_for(42),
        "attempts": [attempt(bad)],
    }
    document = build(tmp_path, seeds=seeds)
    assert document["cells"][0]["records"][0]["status"] == "FAILED"


def test_duplicate_seed_rejected(tmp_path):
    raw = inventory(tmp_path)
    raw["cells"][0]["seeds"].append(
        copy.deepcopy(raw["cells"][0]["seeds"][0])
    )
    with pytest.raises(AggregationError, match="repeats a seed"):
        validate_inventory(raw)


def test_duplicate_attempt_rejected(tmp_path):
    seeds = {
        seed: {
            "pool_manifest_sha256": pool_for(seed),
            "attempts": [attempt(make_run(tmp_path, seed))],
        }
        for seed in range(42, 49)
    }
    seeds[42]["attempts"].append(copy.deepcopy(seeds[42]["attempts"][0]))
    with pytest.raises(AggregationError, match="repeats an attempt number"):
        build(tmp_path, seeds=seeds)


def test_missing_seed_entry_rejected(tmp_path):
    raw = inventory(tmp_path)
    raw["cells"][0]["seeds"].pop()
    with pytest.raises(AggregationError, match="all seven expected seeds"):
        validate_inventory(raw)


@pytest.mark.parametrize("field, value", [
    ("dataset_id", "malex"),
    ("family", "gan"),
    ("pool_manifest_sha256", "c" * 64),
])
def test_lineage_conflict_marks_declared_success_failed(
    tmp_path, field, value,
):
    seeds = {
        seed: {
            "pool_manifest_sha256": pool_for(seed),
            "attempts": [attempt(make_run(tmp_path, seed))],
        }
        for seed in range(43, 49)
    }
    bad = make_run(tmp_path / "bad", 42)
    restore = json.loads((bad / "restore_report.json").read_text())
    if field == "pool_manifest_sha256":
        restore["generator_lineage"]["pool_manifest_sha256"] = value
    else:
        restore["generator_lineage"][field] = value
    (bad / "restore_report.json").write_text(json.dumps(restore))
    seeds[42] = {
        "pool_manifest_sha256": pool_for(42),
        "attempts": [attempt(bad)],
    }
    document = build(tmp_path, seeds=seeds)
    record = document["cells"][0]["records"][0]
    assert record["status"] == "FAILED"
    assert record["real_only_macro_f1"] is None
    attempts = document["cells"][0]["attempt_evidence"][0]["attempts"]
    assert attempts[0]["authentication"] == "FAILED"


def test_earliest_valid_attempt_selected_not_best_score(tmp_path):
    first = make_run(tmp_path / "a1", 42, augmented_errors=8)
    second = make_run(tmp_path / "a2", 42, augmented_errors=1)
    seeds = {
        seed: {
            "pool_manifest_sha256": pool_for(seed),
            "attempts": [attempt(make_run(tmp_path, seed))],
        }
        for seed in range(43, 49)
    }
    seeds[42] = {
        "pool_manifest_sha256": pool_for(42),
        "attempts": [attempt(first, 1), attempt(second, 2)],
    }
    document = build(tmp_path, seeds=seeds)
    cell = document["cells"][0]
    assert cell["attempt_evidence"][0]["selected_attempt"] == 1
    selected = cell["records"][0]["real_plus_synthetic_macro_f1"]
    best = cell["attempt_evidence"][0]["attempts"][1][
        "real_plus_synthetic_macro_f1"
    ]
    assert selected <= best


def test_later_attempt_selected_when_first_fails(tmp_path):
    bad = make_run(tmp_path / "bad", 42)
    (bad / "evaluation.json").unlink()
    good = make_run(tmp_path / "good", 42)
    seeds = {
        seed: {
            "pool_manifest_sha256": pool_for(seed),
            "attempts": [attempt(make_run(tmp_path, seed))],
        }
        for seed in range(43, 49)
    }
    seeds[42] = {
        "pool_manifest_sha256": pool_for(42),
        "attempts": [attempt(bad, 1), attempt(good, 2)],
    }
    document = build(tmp_path, seeds=seeds)
    evidence = document["cells"][0]["attempt_evidence"][0]
    assert evidence["selected_attempt"] == 2
    assert evidence["attempts"][0]["authentication"] == "FAILED"
    assert evidence["attempts"][1]["authentication"] == "PASS"


def test_recipe_identity_mismatch_rejected(tmp_path):
    raw = inventory(tmp_path)
    with pytest.raises(AggregationError, match="classifier recipe"):
        build_aggregate(
            inventory=raw, classifier_recipe_raw=b"{}",
            metric_recipe_raw=METRIC_RAW,
        )


def test_publish_refuses_existing_output(tmp_path):
    path = tmp_path / "aggregate.json"
    publish_aggregate(path, {"a": 1})
    with pytest.raises(FileExistsError):
        publish_aggregate(path, {"a": 2})
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest()


def test_arithmetic_matches_accepted_reporter(tmp_path):
    document = build(tmp_path)
    cell = document["cells"][0]
    assert cell["summary"] == summarize_declared_paired_seeds(
        records=cell["records"]
    )
