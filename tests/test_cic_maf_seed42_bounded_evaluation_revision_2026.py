"""Tests for the bounded CIC MAF seed-42 real-data evaluation driver."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import (
    run_paper01_cic_maf_seed42_bounded_evaluation as driver,
)
from trustforge.evaluation import (
    paper01_classifier_bundle,
    paper01_classifier_manifest,
    paper01_classifier_restoration,
    paper01_classifier_state,
    paper01_classifier_tensorflow,
    paper01_evaluation_validation,
    paper01_fid,
    paper01_fidelity,
    paper01_inception,
    paper01_ms_ssim,
    paper01_ms_ssim_evaluation,
    paper01_paired_results,
    paper01_predictions,
)
from trustforge.evaluation.paper01_status_inventory import REQUIRED_RESULTS
from trustforge.evaluation.paper01_classifier_control import state_digest
from trustforge.evaluation.paper01_inputs import RealSplitView
from trustforge.models.paper01_revision_2026.ar_maf_execution_inputs import (
    synthesis_batch_seed,
)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _pool_manifest():
    records = []
    for class_id in range(driver.CLASS_COUNT):
        for batch_index in range(driver.SAMPLES_PER_CLASS // 8):
            records.append({
                "class_id": class_id,
                "batch_index": batch_index,
                "filename": f"c{class_id}-b{batch_index:03d}.npy",
                "sha256": "1" * 64,
                "sampling_seed": synthesis_batch_seed(
                    dataset_id=driver.DATASET_ID, family=driver.FAMILY,
                    experimental_seed=driver.EXPERIMENTAL_SEED,
                    class_id=class_id, batch_index=batch_index,
                ),
            })
    return {
        "schema_version": 1,
        "dataset_id": driver.DATASET_ID,
        "family": driver.FAMILY,
        "experimental_seed": driver.EXPERIMENTAL_SEED,
        "recipe_sha256": driver.GENERATOR_RECIPE_SHA256,
        "checkpoint_sha256": driver.POOL_CHECKPOINT_SHA256,
        "batches": records,
    }


def _class_labels(counts):
    return np.concatenate([
        np.full(count, class_id, dtype=np.int32)
        for class_id, count in enumerate(counts)
    ])


class _NoTestAccessAdapter:
    def split_size(self, split):
        if split == "test":
            raise AssertionError("test split accessed")
        return driver.SPLIT_COUNTS[split]

    def labels(self, split):
        if split == "test":
            raise AssertionError("test labels accessed")
        return _class_labels(driver.SPLIT_CLASS_COUNTS[split])


def test_source_identity_and_clean_tree_enforced(monkeypatch):
    root = Path.cwd()
    head = "a" * 40
    sources = {"src/example.py": "b" * 64}
    monkeypatch.setattr(
        driver, "_capture_source", lambda value: (head, sources),
    )
    monkeypatch.setattr(driver.subprocess, "check_output", lambda *a, **k: "")
    assert driver._verify_source(root, head) == (head, sources)

    with pytest.raises(RuntimeError, match="unexpected source revision"):
        driver._verify_source(root, "c" * 40)

    monkeypatch.setattr(
        driver.subprocess, "check_output", lambda *a, **k: " M src/example.py\n",
    )
    with pytest.raises(RuntimeError, match="must be clean"):
        driver._verify_source(root, head)


def test_untracked_executable_source_is_rejected(monkeypatch):
    root = Path.cwd()
    head = "a" * 40
    sources = {"src/example.py": "b" * 64}
    monkeypatch.setattr(
        driver, "_capture_source", lambda value: (head, sources),
    )

    def check_output(command, text=True):
        if "status" in command:
            return ""
        if "ls-files" in command:
            return "scripts/rogue_execution.py\n"
        raise AssertionError("unexpected git command")

    monkeypatch.setattr(driver.subprocess, "check_output", check_output)
    with pytest.raises(RuntimeError, match="untracked executable source"):
        driver._verify_source(root, head)

    monkeypatch.setattr(
        driver.subprocess, "check_output", lambda *a, **k: "",
    )
    assert driver._verify_source(root, head) == (head, sources)


def test_dataset_config_identity_enforced(tmp_path, monkeypatch):
    config = {
        "dataset_id": driver.DATASET_ID,
        "paths": {
            "source_csv": "/data/cic.csv",
            "feature_columns_path": "/data/features.json",
        },
    }
    raw = driver._canonical(config)
    path = tmp_path / "dataset_config.json"
    path.write_bytes(raw)
    digest = _sha(raw)
    monkeypatch.setattr(driver, "DATASET_CONFIG_SHA256", digest)
    observed_raw, observed = driver._load_dataset_config(path, digest)
    assert observed_raw == raw
    assert observed == config

    with pytest.raises(ValueError, match="digest mismatch"):
        driver._load_dataset_config(path, "0" * 64)

    monkeypatch.setattr(driver, "DATASET_CONFIG_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="accepted evidence"):
        driver._load_dataset_config(path, digest)


def test_dataset_config_rejects_unexpected_paths(tmp_path, monkeypatch):
    config = {
        "dataset_id": driver.DATASET_ID,
        "paths": {"source_csv": "/data/cic.csv"},
    }
    raw = driver._canonical(config)
    path = tmp_path / "dataset_config.json"
    path.write_bytes(raw)
    digest = _sha(raw)
    monkeypatch.setattr(driver, "DATASET_CONFIG_SHA256", digest)
    with pytest.raises(ValueError, match="invalid CIC dataset configuration"):
        driver._load_dataset_config(path, digest)


def test_accepted_pool_manifest_and_lineage_identities(monkeypatch):
    manifest = _pool_manifest()
    raw = driver._canonical(manifest)
    digest = _sha(raw)
    monkeypatch.setattr(driver, "POOL_MANIFEST_SHA256", digest)
    observed = driver._authenticate_pool_manifest_raw(
        raw, expected_sha=digest,
        expected_checkpoint=driver.POOL_CHECKPOINT_SHA256,
        expected_recipe=driver.GENERATOR_RECIPE_SHA256,
    )
    assert len(observed["batches"]) == driver.POOL_BATCHES

    with pytest.raises(ValueError, match="digest mismatch"):
        driver._authenticate_pool_manifest_raw(
            raw, expected_sha="0" * 64,
            expected_checkpoint=driver.POOL_CHECKPOINT_SHA256,
            expected_recipe=driver.GENERATOR_RECIPE_SHA256,
        )
    with pytest.raises(ValueError, match="identity mismatch"):
        driver._authenticate_pool_manifest_raw(
            raw, expected_sha=digest,
            expected_checkpoint="f" * 64,
            expected_recipe=driver.GENERATOR_RECIPE_SHA256,
        )


def test_pool_manifest_rejects_corrupted_or_incomplete_coverage(monkeypatch):
    manifest = _pool_manifest()

    incomplete = json.loads(json.dumps(manifest))
    incomplete["batches"].pop()
    raw = driver._canonical(incomplete)
    monkeypatch.setattr(driver, "POOL_MANIFEST_SHA256", _sha(raw))
    with pytest.raises(ValueError, match="1250 ordered batches"):
        driver._authenticate_pool_manifest_raw(
            raw, expected_sha=_sha(raw),
            expected_checkpoint=driver.POOL_CHECKPOINT_SHA256,
            expected_recipe=driver.GENERATOR_RECIPE_SHA256,
        )

    duplicate = json.loads(json.dumps(manifest))
    duplicate["batches"][1]["class_id"] = duplicate["batches"][0]["class_id"]
    duplicate["batches"][1]["batch_index"] = duplicate["batches"][0]["batch_index"]
    raw = driver._canonical(duplicate)
    monkeypatch.setattr(driver, "POOL_MANIFEST_SHA256", _sha(raw))
    with pytest.raises(ValueError, match="duplicate"):
        driver._authenticate_pool_manifest_raw(
            raw, expected_sha=_sha(raw),
            expected_checkpoint=driver.POOL_CHECKPOINT_SHA256,
            expected_recipe=driver.GENERATOR_RECIPE_SHA256,
        )

    wrong_seed = json.loads(json.dumps(manifest))
    wrong_seed["batches"][0]["sampling_seed"] += 1
    raw = driver._canonical(wrong_seed)
    monkeypatch.setattr(driver, "POOL_MANIFEST_SHA256", _sha(raw))
    with pytest.raises(ValueError, match="sampling seed"):
        driver._authenticate_pool_manifest_raw(
            raw, expected_sha=_sha(raw),
            expected_checkpoint=driver.POOL_CHECKPOINT_SHA256,
            expected_recipe=driver.GENERATOR_RECIPE_SHA256,
        )


def test_checkpoint_lineage_matching_and_rejection(monkeypatch):
    best_raw = b"accepted-best-state"
    best_sha = _sha(best_raw)
    execution = {"schema": "fixture-execution", "value": 1}
    envelope_sha = _sha(driver._canonical(execution) + b"\n")
    pool_manifest_sha = "2" * 64
    receipt_sha = "3" * 64
    fitted_sha = "4" * 64
    source_commit = "5" * 40
    recipe_sha = "6" * 64

    monkeypatch.setattr(driver, "POOL_CHECKPOINT_SHA256", best_sha)
    monkeypatch.setattr(driver, "POOL_MANIFEST_SHA256", pool_manifest_sha)
    monkeypatch.setattr(driver, "GENERATOR_RECIPE_SHA256", recipe_sha)
    monkeypatch.setattr(driver, "CHECKPOINT_SOURCE_COMMIT", source_commit)
    monkeypatch.setattr(driver, "CHECKPOINT_FITTED_STATE_SHA256", fitted_sha)
    monkeypatch.setattr(driver, "CHECKPOINT_EXPECTATIONS_SHA256", receipt_sha)
    monkeypatch.setattr(
        driver, "CHECKPOINT_EXECUTION_MANIFEST_SHA256", envelope_sha,
    )

    pool_identity = {
        "manifest_sha256": pool_manifest_sha,
        "checkpoint_sha256": best_sha,
        "recipe_sha256": recipe_sha,
    }
    checkpoint_manifest = {
        "source_commit": source_commit,
        "completed_training_steps": driver.CHECKPOINT_COMPLETED_STEPS,
        "artifacts": {"best": {"sha256": best_sha}},
    }
    arguments = {
        "pool_identity": pool_identity,
        "checkpoint_manifest": checkpoint_manifest,
        "best_raw": best_raw,
        "execution_manifest": execution,
        "receipt_sha256": receipt_sha,
        "fitted_state_sha256": fitted_sha,
        "source_commit": source_commit,
    }
    summary = driver._lineage_summary(**arguments)
    assert summary["lineage_authentication"] == "DRIVER_VERIFIED"
    assert summary["checkpoint_best_state_sha256"] == best_sha
    assert summary["complete_evaluation"] is False
    assert len(summary["sha256"]) == 64

    with pytest.raises(ValueError, match="lineage"):
        driver._lineage_summary(**{**arguments, "best_raw": b"other-state"})
    with pytest.raises(ValueError, match="lineage"):
        driver._lineage_summary(**{
            **arguments,
            "execution_manifest": {"schema": "different"},
        })
    with pytest.raises(ValueError, match="lineage"):
        driver._lineage_summary(**{
            **arguments,
            "pool_identity": {**pool_identity, "checkpoint_sha256": "7" * 64},
        })


def test_checkpoint_expectations_must_match_accepted(monkeypatch):
    receipt = {
        "schema": driver.CHECKPOINT_EXPECTATIONS_SCHEMA,
        "source_commit": driver.CHECKPOINT_SOURCE_COMMIT,
        "generation_name": driver.CHECKPOINT_GENERATION,
        "job_id": "311675",
        "fitted_state_sha256": driver.CHECKPOINT_FITTED_STATE_SHA256,
        "execution_manifest_file_sha256":
            driver.CHECKPOINT_EXECUTION_MANIFEST_SHA256,
        "configuration_sha256": "8" * 64,
        "training_pid": 123,
    }
    raw = driver._canonical(receipt)
    digest = _sha(raw)
    monkeypatch.setattr(driver, "CHECKPOINT_EXPECTATIONS_SHA256", digest)
    observed = driver._authenticate_expectations(raw, digest, "8" * 64)
    assert observed["job_id"] == "311675"

    with pytest.raises(ValueError, match="digest mismatch"):
        driver._authenticate_expectations(raw, "0" * 64, "8" * 64)
    with pytest.raises(ValueError, match="receipt mismatch"):
        driver._authenticate_expectations(raw, digest, "9" * 64)

    corrupted = {**receipt, "fitted_state_sha256": "0" * 64}
    corrupted_raw = driver._canonical(corrupted)
    monkeypatch.setattr(
        driver, "CHECKPOINT_EXPECTATIONS_SHA256", _sha(corrupted_raw),
    )
    with pytest.raises(ValueError, match="receipt mismatch"):
        driver._authenticate_expectations(
            corrupted_raw, _sha(corrupted_raw), "8" * 64,
        )


def test_training_identity_never_touches_test_split():
    adapter = _NoTestAccessAdapter()
    evidence = driver._verify_split_identities(adapter, include_test=False)
    assert evidence["split_sizes"] == {
        "train": driver.TRAIN_SIZE, "validation": driver.VALIDATION_SIZE,
    }
    assert evidence["split_class_counts"]["validation"] == (
        driver.SPLIT_CLASS_COUNTS["validation"]
    )
    identity = driver._test_identity("a" * 64)
    assert identity["size"] == driver.TEST_SIZE
    assert identity["split"] == "test"

    with pytest.raises(AssertionError, match="test split accessed"):
        driver._verify_split_identities(adapter, include_test=True)


def test_training_phase_builds_only_train_and_validation_views():
    requested = []

    def real_view(**arguments):
        requested.append(arguments["split"])
        return SimpleNamespace(
            identity={"split": arguments["split"]},
            size=driver.SPLIT_COUNTS[arguments["split"]],
        )

    def combined(*, real, synthetic):
        return SimpleNamespace(identity={"real": real.identity})

    components = {"real_view": real_view, "combined": combined}
    session = {
        "adapter": object(),
        "provenance_sha256": "a" * 64,
        "pool": object(),
    }
    real, validation, augmented = driver._build_training_views(
        session, components,
    )
    assert requested == ["train", "validation"]
    assert "test" not in requested
    assert augmented.identity["real"] == real.identity
    assert validation.identity["split"] == "validation"


def _paired_evidence():
    report = {
        "initial_weights_sha256": "a" * 64,
        "selected_weights_sha256": "b" * 64,
    }
    return {
        "test_evaluated": False,
        "complete_evaluation": False,
        "arms": {
            "real_only": {"report": report},
            "real_plus_synthetic": {
                "report": report,
                "pool_consumption": {"schema": "consumption"},
            },
        },
    }


def test_paired_evidence_requires_consumption_and_shared_initialization():
    augmented_identity = {
        "synthetic": {
            "manifest_sha256": driver.POOL_MANIFEST_SHA256,
            "checkpoint_sha256": driver.POOL_CHECKPOINT_SHA256,
            "family": driver.FAMILY,
            "experimental_seed": driver.EXPERIMENTAL_SEED,
        },
    }
    paired = _paired_evidence()
    consumption = driver._require_training_evidence(
        paired, augmented_identity=augmented_identity,
    )
    assert consumption == {"schema": "consumption"}

    missing = _paired_evidence()
    del missing["arms"]["real_plus_synthetic"]["pool_consumption"]
    with pytest.raises(ValueError, match="consumption"):
        driver._require_training_evidence(
            missing, augmented_identity=augmented_identity,
        )

    real_only_consumption = _paired_evidence()
    real_only_consumption["arms"]["real_only"]["pool_consumption"] = {}
    with pytest.raises(ValueError, match="real-only"):
        driver._require_training_evidence(
            real_only_consumption, augmented_identity=augmented_identity,
        )

    mismatched = _paired_evidence()
    mismatched["arms"]["real_plus_synthetic"]["report"] = {
        "initial_weights_sha256": "c" * 64,
        "selected_weights_sha256": "b" * 64,
    }
    with pytest.raises(ValueError, match="initial classifier weights"):
        driver._require_training_evidence(
            mismatched, augmented_identity=augmented_identity,
        )

    test_access = _paired_evidence()
    test_access["test_evaluated"] = True
    with pytest.raises(ValueError, match="pre-test"):
        driver._require_training_evidence(
            test_access, augmented_identity=augmented_identity,
        )


def _receipt():
    shared = {
        "source_commit": "a" * 40,
        "source_files_sha256": "b" * 64,
        "environment_sha256": "c" * 64,
    }
    arms = {
        name: {
            "context": {"arm": name},
            "manifest_sha256": "d" * 64,
            "selected_weights_sha256": "e" * 64,
        }
        for name in ("real_only", "real_plus_synthetic")
    }
    receipt = {
        "schema": driver.RECEIPT_SCHEMA,
        **shared,
        "job_id": "1234",
        "train_pid": 111,
        "evaluator_seed": driver.EXPERIMENTAL_SEED,
        "initial_weights_sha256": "f" * 64,
        "classifier_recipe_sha256": driver.CLASSIFIER_RECIPE_SHA256,
        "metric_recipe_sha256": driver.METRIC_RECIPE_SHA256,
        "test_identity": driver._test_identity("1" * 64),
        "synthetic_pool_manifest_sha256": driver.POOL_MANIFEST_SHA256,
        "generator_lineage_sha256": "2" * 64,
        "context_evidence_sha256": "3" * 64,
        "arms": arms,
    }
    return shared, receipt


def test_restore_receipt_authentication_rejects_mismatches():
    shared, receipt = _receipt()
    raw = driver._canonical(receipt)
    arguments = {
        "shared": shared,
        "classifier_recipe_sha256": driver.CLASSIFIER_RECIPE_SHA256,
        "metric_recipe_sha256": driver.METRIC_RECIPE_SHA256,
        "test_identity": driver._test_identity("1" * 64),
        "pool_manifest_sha256": driver.POOL_MANIFEST_SHA256,
        "lineage_sha256": "2" * 64,
        "job_id": "1234",
        "pid": 222,
    }
    observed = driver._authenticate_receipt(raw, _sha(raw), **arguments)
    assert observed["train_pid"] == 111

    with pytest.raises(ValueError, match="receipt digest"):
        driver._authenticate_receipt(raw, "0" * 64, **arguments)
    with pytest.raises(ValueError, match="source/environment"):
        driver._authenticate_receipt(
            raw, _sha(raw), **{**arguments, "shared": {**shared, "source_commit": "9" * 40}},
        )
    with pytest.raises(ValueError, match="receipt identity"):
        driver._authenticate_receipt(
            raw, _sha(raw), **{**arguments, "pool_manifest_sha256": "9" * 64},
        )
    with pytest.raises(ValueError, match="receipt identity"):
        driver._authenticate_receipt(
            raw, _sha(raw), **{**arguments, "lineage_sha256": "9" * 64},
        )
    with pytest.raises(ValueError, match="receipt identity"):
        driver._authenticate_receipt(raw, _sha(raw), **{**arguments, "pid": 111})
    with pytest.raises(ValueError, match="receipt identity"):
        driver._authenticate_receipt(
            raw, _sha(raw),
            **{**arguments, "test_identity": driver._test_identity("4" * 64)},
        )


def test_restore_receipt_requires_both_arms():
    shared, receipt = _receipt()
    broken = {**receipt, "arms": {"real_only": receipt["arms"]["real_only"]}}
    raw = driver._canonical(broken)
    with pytest.raises(ValueError, match="requires both arms"):
        driver._authenticate_receipt(
            raw, _sha(raw), shared=shared,
            classifier_recipe_sha256=driver.CLASSIFIER_RECIPE_SHA256,
            metric_recipe_sha256=driver.METRIC_RECIPE_SHA256,
            test_identity=driver._test_identity("1" * 64),
            pool_manifest_sha256=driver.POOL_MANIFEST_SHA256,
            lineage_sha256="2" * 64, job_id="1234", pid=222,
        )


def test_recipes_are_exactly_the_accepted_bytes():
    recipes = driver._load_recipes(Path.cwd())
    assert _sha(recipes["generator"]) == driver.GENERATOR_RECIPE_SHA256
    assert _sha(recipes["classifier"]) == driver.CLASSIFIER_RECIPE_SHA256
    assert _sha(recipes["metric"]) == driver.METRIC_RECIPE_SHA256


def test_unified_evaluation_must_verify_all_groups_and_stay_incomplete():
    good = {
        "complete_evaluation": False,
        "verified_group_count": 7,
        "required_status_blockers": [],
        "lineage_blockers": [],
        "verified_pool_consumption": {"schema": "verified"},
    }
    driver._require_verified_evaluation(good)

    for mutation, match in (
        ({"complete_evaluation": True}, "complete_evaluation=false"),
        ({"verified_group_count": 6}, "complete_evaluation=false"),
        ({"required_status_blockers": [{}]}, "complete_evaluation=false"),
        ({"lineage_blockers": [{}]}, "complete_evaluation=false"),
        ({"verified_pool_consumption": None}, "complete_evaluation=false"),
    ):
        with pytest.raises(ValueError, match=match):
            driver._require_verified_evaluation({**good, **mutation})


def test_reports_cannot_claim_completion_or_benchmark_go():
    driver._assert_incomplete_report({
        "complete_evaluation": False, "full_benchmark_go": False,
    })
    with pytest.raises(ValueError, match="complete_evaluation=false"):
        driver._assert_incomplete_report({
            "complete_evaluation": True, "full_benchmark_go": False,
        })
    with pytest.raises(ValueError, match="complete_evaluation=false"):
        driver._assert_incomplete_report({
            "complete_evaluation": False, "full_benchmark_go": True,
        })


def test_publication_is_exclusive_and_never_overwrites(tmp_path):
    path = tmp_path / "evidence.json"
    driver._write_exclusive_json(path, {"status": "PASS"})
    with pytest.raises(FileExistsError):
        driver._write_exclusive_json(path, {"status": "OVERWRITE"})
    assert json.loads(path.read_bytes()) == {"status": "PASS"}

    run = tmp_path / "run"
    created = driver._require_empty_run_dir(run)
    assert created.is_dir()
    with pytest.raises(FileExistsError):
        driver._require_empty_run_dir(run)
    assert driver._require_existing_run_dir(run) == run.resolve()


def test_missing_regular_evidence_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="regular file"):
        driver._read_regular(tmp_path / "missing.json", name="evidence")

    target = tmp_path / "target.json"
    target.write_bytes(b"{}")
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="regular file"):
        driver._read_regular(link, name="evidence")


# ---------------------------------------------------------------------------
# Bounded mocked composition of the actual train/restore entry points
# ---------------------------------------------------------------------------


class _FakeAdapter:
    def split_size(self, split):
        return driver.SPLIT_COUNTS[split]

    def labels(self, split):
        return _class_labels(driver.SPLIT_CLASS_COUNTS[split])

    def canonical_batch(self, split, positions):
        labels = self.labels(split)[positions]
        images = np.zeros(
            (len(positions), 40, 40, 3), dtype=np.float32,
        )
        images[:, :, :, 0] = (
            labels.astype(np.float32) / np.float32(4)
        )[:, None, None]
        return images, np.eye(5, dtype=np.float32)[labels]


def _fake_session(ordering):
    provenance = "a" * 64
    pool_identity = {
        "schema_version": 1,
        "dataset_id": driver.DATASET_ID,
        "family": driver.FAMILY,
        "experimental_seed": driver.EXPERIMENTAL_SEED,
        "recipe_sha256": driver.GENERATOR_RECIPE_SHA256,
        "checkpoint_sha256": driver.POOL_CHECKPOINT_SHA256,
        "manifest_sha256": driver.POOL_MANIFEST_SHA256,
    }
    lineage = {
        "schema": "trustforge.paper01.cic-maf.seed42.lineage-evidence.v1",
        "sha256": "b" * 64,
        "lineage_authentication": "DRIVER_VERIFIED",
        "complete_evaluation": False,
        "full_benchmark_go": False,
    }
    splits = []

    def real_view(**arguments):
        split = arguments["split"]
        splits.append(split)
        if split == "test":
            if not ordering.get("authenticated"):
                raise AssertionError(
                    "test view built before receipt authentication"
                )
            ordering["test_after_auth"] = True
        return RealSplitView(
            adapter=arguments["adapter"],
            dataset_id=arguments["dataset_id"],
            split=split,
            provenance_sha256=arguments["provenance_sha256"],
        )

    def combined(*, real, synthetic):
        return SimpleNamespace(
            identity={
                "condition": "real_plus_synthetic",
                "real": real.identity,
                "synthetic": pool_identity,
                "position_order": "real_then_synthetic",
            },
            size=driver.TRAIN_SIZE + driver.POOL_SAMPLES,
        )

    return {
        "root": Path.cwd(),
        "head": "c" * 40,
        "sources": {"src/example.py": "d" * 64},
        "config_raw": b"{}",
        "recipes": driver._load_recipes(Path.cwd()),
        "components": {"real_view": real_view, "combined": combined},
        "adapter": _FakeAdapter(),
        "identity": {"dataset_provenance_sha256": provenance},
        "provenance_sha256": provenance,
        "split_evidence": {},
        "pool": SimpleNamespace(identity=pool_identity),
        "pool_raw": b"",
        "pool_sha256": driver.POOL_MANIFEST_SHA256,
        "expectations_raw": b"",
        "lineage": lineage,
        "splits": splits,
    }


def _fake_weights():
    return {
        "real_only": [np.array([0.0], dtype=np.float32)],
        "real_plus_synthetic": [np.array([1.0], dtype=np.float32)],
    }


def _fake_selected_digests():
    return {
        name: state_digest(values)
        for name, values in _fake_weights().items()
    }


def _patch_train(monkeypatch, session, written, *, fail_on=None):
    monkeypatch.setattr(
        driver, "_prepare_inputs",
        lambda args, *, include_test_identity: session,
    )
    monkeypatch.setattr(
        driver, "_environment",
        lambda: {"python": "test", "tensorflow": "test"},
    )
    monkeypatch.setattr(driver, "_verify_imports", lambda root, sources: 1)
    monkeypatch.setattr(
        driver, "_verify_source",
        lambda root, head: (session["head"], session["sources"]),
    )
    monkeypatch.setenv("SLURM_JOB_ID", "1234")
    monkeypatch.setattr(driver.os, "getpid", lambda: 111)

    real_writer = driver._write_exclusive_json

    def writer(path, value):
        name = Path(path).name
        if fail_on is not None and name == fail_on:
            raise RuntimeError("simulated persistence interruption")
        written.append(name)
        return real_writer(path, value)

    monkeypatch.setattr(driver, "_write_exclusive_json", writer)

    monkeypatch.setattr(
        paper01_classifier_tensorflow, "require_compute_node", lambda: None,
    )

    selected = _fake_selected_digests()
    base_report = {
        "initial_weights_sha256": "c" * 64,
        "selected_weights_sha256": selected["real_only"],
    }
    augmented_report = {
        "initial_weights_sha256": "c" * 64,
        "selected_weights_sha256": selected["real_plus_synthetic"],
    }
    paired = {
        "status": "MOCK_PAIRED",
        "initial_weights_sha256": "c" * 64,
        "test_evaluated": False,
        "complete_evaluation": False,
        "arms": {
            "real_only": {
                "report": base_report,
                "selected_weights": [1.0],
            },
            "real_plus_synthetic": {
                "report": augmented_report,
                "selected_weights": [2.0],
                "pool_consumption": {
                    "schema": "trustforge.paper01.augmented-pool-consumption.v1",
                },
            },
        },
    }
    monkeypatch.setattr(
        paper01_classifier_tensorflow, "prepare_paired_classifiers",
        lambda **kwargs: paired,
    )
    monkeypatch.setattr(
        paper01_classifier_state, "encode_classifier_weights",
        lambda **kwargs: b"state-bytes",
    )
    monkeypatch.setattr(
        paper01_classifier_manifest, "prepare_classifier_manifest",
        lambda *, context, **kwargs: {"arm": context["arm"]},
    )
    monkeypatch.setattr(
        paper01_classifier_manifest, "canonical_manifest",
        lambda value: driver._canonical(value),
    )

    published = []

    def publish(
        parent, name, *, manifest_raw, state_raw, recipe_raw,
        expected_manifest_sha256, expected_context,
        expected_selected_weights_sha256,
    ):
        directory = Path(parent) / name
        directory.mkdir()
        (directory / "manifest.json").write_bytes(manifest_raw)
        (directory / "state.npz").write_bytes(state_raw)
        published.append(name)

    monkeypatch.setattr(
        paper01_classifier_bundle, "publish_classifier_bundle", publish,
    )
    return paired, published


def _patch_restore(
    monkeypatch, session, ordering, captured, *, summary=None,
    component_calls=None,
):
    calls = {} if component_calls is None else component_calls
    monkeypatch.setattr(
        driver, "_prepare_inputs",
        lambda args, *, include_test_identity: session,
    )
    monkeypatch.setattr(
        driver, "_environment",
        lambda: {"python": "test", "tensorflow": "test"},
    )
    monkeypatch.setattr(driver, "_verify_imports", lambda root, sources: 1)
    monkeypatch.setattr(
        driver, "_verify_source",
        lambda root, head: (session["head"], session["sources"]),
    )
    monkeypatch.setattr(driver.os, "getpid", lambda: 222)

    real_authenticate = driver._authenticate_receipt

    def authenticate(*arguments, **keywords):
        result = real_authenticate(*arguments, **keywords)
        ordering["authenticated"] = True
        return result

    monkeypatch.setattr(driver, "_authenticate_receipt", authenticate)
    monkeypatch.setattr(
        paper01_classifier_tensorflow, "require_compute_node", lambda: None,
    )

    weights = _fake_weights()

    class _Tensor:
        def __init__(self, value):
            self._value = value

        def numpy(self):
            return self._value

    class _Model:
        def __call__(self, images, training=False):
            return _Tensor(np.full(
                (len(images), driver.CLASS_COUNT), 0.2, dtype=np.float32,
            ))

    def restore(
        directory, *, recipe_raw, expected_manifest_sha256,
        expected_context, expected_selected_weights_sha256,
    ):
        name = Path(directory).name
        return {
            "binding": {
                "model": _Model(),
                "controls": {
                    "get_weights": lambda: weights[name],
                    "get_execution_state":
                        lambda: [np.array([0.0], dtype=np.float32)],
                },
            },
            "manifest": {"training_report": {"arm": name}},
            "report": {
                "selected_weights_sha256": expected_selected_weights_sha256,
            },
        }

    monkeypatch.setattr(
        paper01_classifier_restoration, "restore_selected_classifier", restore,
    )

    selected = _fake_selected_digests()

    def link(**kwargs):
        return {
            "classifier_recipe_sha256": _sha(
                kwargs["classifier_recipe_raw"]
            ),
            "metric_recipe_sha256": _sha(kwargs["metric_recipe_raw"]),
            "arms": {
                "real_only": {
                    "prediction": {
                        "metrics": {},
                        "selected_weights_sha256": selected["real_only"],
                    },
                },
                "real_plus_synthetic": {
                    "prediction": {
                        "metrics": {},
                        "selected_weights_sha256":
                            selected["real_plus_synthetic"],
                    },
                },
            },
        }

    monkeypatch.setattr(
        paper01_paired_results, "link_paired_results", link,
    )

    class FakeExtractor:
        def __init__(self, *, weights_path):
            self.identity = {"architecture": "fake-inception"}

    monkeypatch.setattr(paper01_inception, "InceptionFeatures", FakeExtractor)

    def prepare(**kwargs):
        calls["fidelity"] = kwargs
        return {
            "real_features": np.zeros((625, 2048), dtype=np.float32),
            "synthetic_features": np.zeros((625, 2048), dtype=np.float32),
            "report": {
                "feature_extractor": {"architecture": "fake-inception"},
                "kid": {"draws": [{}, {}]},
                "samples_per_class": 125,
            },
        }

    monkeypatch.setattr(
        paper01_fidelity, "prepare_features_and_kid", prepare,
    )

    class FakeScorer:
        identity = {"schema": "fake-ms-ssim"}

    monkeypatch.setattr(paper01_ms_ssim, "ThreeScaleMSSSIM", FakeScorer)

    def ms_ssim(**kwargs):
        calls["ms_ssim"] = kwargs
        return {
            "per_class": {
                str(class_id): {"scores": [0.5] * 200}
                for class_id in range(driver.CLASS_COUNT)
            },
        }

    monkeypatch.setattr(
        paper01_ms_ssim_evaluation, "evaluate_ms_ssim_pairs", ms_ssim,
    )
    monkeypatch.setattr(
        paper01_fid, "feature_fid", lambda real, synthetic: {"fid": 1.0},
    )
    monkeypatch.setattr(
        paper01_fid, "complete_class_fid",
        lambda real, synthetic, **kwargs: {"classes": {}},
    )

    default = {
        "complete_evaluation": False,
        "verified_group_count": 7,
        "required_status_blockers": [],
        "lineage_blockers": [],
        "verified_pool_consumption": {"selected_weights_sha256": "e" * 64},
        "verified_groups": ["all"],
    }

    def verify(**kwargs):
        captured.update(kwargs)
        return default if summary is None else summary

    monkeypatch.setattr(
        paper01_evaluation_validation, "verify_evaluation", verify,
    )


def _run_mocked_train(tmp_path, monkeypatch):
    ordering = {}
    session = _fake_session(ordering)
    run = tmp_path / "run"
    written = []
    _patch_train(monkeypatch, session, written)
    driver._run_train(SimpleNamespace(
        run_dir=str(run), expected_head=session["head"], weights="/fake",
    ))
    return ordering, session, run, written


def test_actual_train_entry_point_publishes_bundles_and_receipt(
    tmp_path, monkeypatch, capsys,
):
    ordering, session, run, written = _run_mocked_train(tmp_path, monkeypatch)
    assert session["splits"] == ["train", "validation"]
    assert "test" not in session["splits"]

    for name in ("real_only", "real_plus_synthetic"):
        directory = run / "checkpoints" / name
        assert (directory / "manifest.json").is_file()
        assert (directory / "state.npz").is_file()

    receipt_raw = (run / "expectations.json").read_bytes()
    receipt = json.loads(receipt_raw)
    assert receipt["schema"] == driver.RECEIPT_SCHEMA
    assert receipt["synthetic_pool_manifest_sha256"] == (
        driver.POOL_MANIFEST_SHA256
    )
    assert receipt["job_id"] == "1234"
    assert receipt["train_pid"] == 111
    assert set(receipt["arms"]) == {"real_only", "real_plus_synthetic"}
    assert receipt["context_evidence_sha256"] == _sha(
        (run / "context_evidence.json").read_bytes()
    )
    for name, arm in receipt["arms"].items():
        manifest_raw = (
            run / "checkpoints" / name / "manifest.json"
        ).read_bytes()
        assert arm["manifest_sha256"] == _sha(manifest_raw)
    assert (run / "train_report.json").is_file()
    assert (run / "pool_consumption.json").is_file()
    assert (run / "lineage_evidence.json").is_file()

    assert written.index("train_report.json") < written.index(
        "expectations.json"
    )
    assert "CIC_MAF_SEED42_EVALUATION_TRAIN=PASS" in capsys.readouterr().out


def test_training_receipt_is_withheld_when_train_report_fails(
    tmp_path, monkeypatch,
):
    session = _fake_session({})
    run = tmp_path / "run"
    written = []
    _patch_train(monkeypatch, session, written, fail_on="train_report.json")
    with pytest.raises(RuntimeError, match="simulated persistence interruption"):
        driver._run_train(SimpleNamespace(
            run_dir=str(run), expected_head=session["head"], weights="/fake",
        ))
    assert (run / "context_evidence.json").is_file()
    assert not (run / "train_report.json").exists()
    assert not (run / "expectations.json").exists()


def test_actual_restore_entry_point_authenticates_before_test_and_composes(
    tmp_path, monkeypatch, capsys,
):
    ordering, session, run, _ = _run_mocked_train(tmp_path, monkeypatch)
    capsys.readouterr()
    receipt_raw = (run / "expectations.json").read_bytes()
    captured = {}
    calls = {}
    _patch_restore(
        monkeypatch, session, ordering, captured, component_calls=calls,
    )
    driver._run_restore(SimpleNamespace(
        run_dir=str(run), expected_head=session["head"], weights="/fake",
        receipt_sha256=_sha(receipt_raw),
    ))

    assert ordering.get("test_after_auth") is True
    assert "test" in session["splits"]
    assert [record["result_id"] for record in captured["records"]] == list(
        REQUIRED_RESULTS
    )
    assert set(captured["artifact_bytes"]) == set(REQUIRED_RESULTS)
    assert set(captured["downstream_arguments"]["arms"]) == {
        "real_only", "real_plus_synthetic",
    }
    assert set(captured["fidelity_arguments"]) == {
        "expected_synthetic_manifest_sha256", "kid_inputs", "ms_ssim_inputs",
    }
    assert captured["augmented_pool_linkage"]["manifest_sha256"] == (
        driver.POOL_MANIFEST_SHA256
    )
    assert captured["augmented_checkpoint_evidence"][
        "expected_expectations_sha256"
    ] == _sha(receipt_raw)

    fidelity = calls["fidelity"]
    assert isinstance(fidelity["real_validation_view"], RealSplitView)
    assert fidelity["real_validation_labels"].shape == (
        driver.VALIDATION_SIZE,
    )
    assert fidelity["synthetic_labels"].shape == (driver.POOL_SAMPLES,)
    assert isinstance(fidelity["selection_raw"], bytes)
    assert calls["ms_ssim"]["labels"].shape == (driver.POOL_SAMPLES,)
    assert isinstance(calls["ms_ssim"]["selection_raw"], bytes)

    evaluation = json.loads((run / "evaluation.json").read_bytes())
    assert evaluation["complete_evaluation"] is False
    restore_report = json.loads((run / "restore_report.json").read_bytes())
    assert restore_report["complete_evaluation"] is False
    assert restore_report["full_benchmark_go"] is False
    out = capsys.readouterr().out
    assert "CIC_MAF_SEED42_EVALUATION_RESTORE=PASS" in out
    assert "CIC_MAF_SEED42_BOUNDED_EVALUATION=PASS" in out
    assert "COMPLETE_EVALUATION=False" in out
    assert "FULL_BENCHMARK_GO=NO" in out


def test_restore_fails_closed_without_success_marker_on_incomplete_result(
    tmp_path, monkeypatch, capsys,
):
    ordering, session, run, _ = _run_mocked_train(tmp_path, monkeypatch)
    capsys.readouterr()
    receipt_raw = (run / "expectations.json").read_bytes()
    captured = {}
    _patch_restore(
        monkeypatch, session, ordering, captured,
        summary={
            "complete_evaluation": False,
            "verified_group_count": 6,
            "required_status_blockers": [],
            "lineage_blockers": [],
            "verified_pool_consumption": {"selected_weights_sha256": "e" * 64},
            "verified_groups": [],
        },
    )
    with pytest.raises(ValueError, match="complete_evaluation=false"):
        driver._run_restore(SimpleNamespace(
            run_dir=str(run), expected_head=session["head"], weights="/fake",
            receipt_sha256=_sha(receipt_raw),
        ))
    assert not (run / "evaluation.json").exists()
    assert not (run / "restore_report.json").exists()
    assert "CIC_MAF_SEED42_BOUNDED_EVALUATION=PASS" not in capsys.readouterr().out
