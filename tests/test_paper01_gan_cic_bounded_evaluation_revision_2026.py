"""Focused tests for the bounded CIC GAN seed-42 evaluation driver.

Fixture-only. No production pool, checkpoint, classifier training or
real-data evaluation is accessed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import run_paper01_cic_gan_seed42_bounded_evaluation as driver
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
)
from trustforge.evaluation.paper01_classifier_control import (
    state_digest,
)
from trustforge.evaluation.paper01_gan_pool import GANPoolView
from trustforge.evaluation.paper01_inputs import RealSplitView
from trustforge.evaluation.paper01_status_inventory import REQUIRED_RESULTS
from trustforge.models.paper01_revision_2026.gan_cic_pool_publication import (
    GAN_POOL_BATCH_SIZE,
    GAN_POOL_SCHEMA,
)
from trustforge.models.paper01_revision_2026.synthesis_linkage import (
    LATENT_STREAM_CONTRACT,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


# ---------------------------------------------------------------------------
# Fixtures and stand-ins
# ---------------------------------------------------------------------------


def _gan_manifest():
    """Structurally complete accepted-schema GAN pool manifest (625 batches)."""
    records = []
    for class_id in range(driver.CLASS_COUNT):
        for batch_index in range(125):
            records.append({
                "class_id": class_id,
                "batch_index": batch_index,
                "sample_indices": list(range(
                    batch_index * GAN_POOL_BATCH_SIZE,
                    (batch_index + 1) * GAN_POOL_BATCH_SIZE,
                )),
                "filename": f"c{class_id}-b{batch_index:03d}.npy",
                "sha256": format(class_id * 125 + batch_index, "064x"),
                "shape": [GAN_POOL_BATCH_SIZE, 40, 40, 3],
                "dtype": "float32",
            })
    return {
        "schema": GAN_POOL_SCHEMA,
        "dataset_id": driver.DATASET_ID,
        "family": driver.FAMILY,
        "experimental_seed": driver.EXPERIMENTAL_SEED,
        "class_mapping": ["0", "1", "2", "3", "4"],
        "class_count": driver.CLASS_COUNT,
        "checkpoint_role": "terminal",
        "checkpoint_sha256": driver.POOL_CHECKPOINT_SHA256,
        "checkpoint_run_identity": driver.CHECKPOINT_RUN_IDENTITY,
        "executable_source_commit": driver.GENERATOR_SOURCE_COMMIT,
        "recipe_sha256": driver.GENERATOR_RECIPE_SHA256,
        "latent_stream_contract": LATENT_STREAM_CONTRACT,
        "rng": {
            "contract": LATENT_STREAM_CONTRACT,
            "bit_generator": "PCG64",
            "latent_dimension": 100,
        },
        "model_call_batch_size": GAN_POOL_BATCH_SIZE,
        "quota_per_class": driver.SAMPLES_PER_CLASS,
        "batches_per_class": 125,
        "complete_pool": True,
        "expected_total": driver.POOL_SAMPLES,
        "observed_total": driver.POOL_SAMPLES,
        "status": "COMPLETE",
        "authoritative_artifact": True,
        "canonical_output": {
            "shape": [40, 40, 3],
            "layout": "NHWC",
            "dtype": "float32",
            "range": [0.0, 1.0],
        },
        "batches": records,
    }


def _write_manifest(directory, manifest):
    raw = driver._canonical(manifest)
    path = directory / "manifest.json"
    path.write_bytes(raw)
    return path, raw, _sha(raw)


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


class _FakeAdapter:
    def split_size(self, split):
        return driver.SPLIT_COUNTS[split]

    def labels(self, split):
        return _class_labels(driver.SPLIT_CLASS_COUNTS[split])

    def canonical_batch(self, split, positions):
        labels = self.labels(split)[positions]
        images = np.zeros((len(positions), 40, 40, 3), dtype=np.float32)
        images[:, :, :, 0] = (
            labels.astype(np.float32) / np.float32(4)
        )[:, None, None]
        return images, np.eye(5, dtype=np.float32)[labels]


def _pool_identity():
    return {
        "schema": GAN_POOL_SCHEMA,
        "dataset_id": driver.DATASET_ID,
        "family": driver.FAMILY,
        "experimental_seed": driver.EXPERIMENTAL_SEED,
        "class_mapping": ["0", "1", "2", "3", "4"],
        "class_count": driver.CLASS_COUNT,
        "checkpoint_role": "terminal",
        "checkpoint_sha256": driver.POOL_CHECKPOINT_SHA256,
        "checkpoint_run_identity": driver.CHECKPOINT_RUN_IDENTITY,
        "executable_source_commit": driver.GENERATOR_SOURCE_COMMIT,
        "recipe_sha256": driver.GENERATOR_RECIPE_SHA256,
        "manifest_sha256": driver.POOL_MANIFEST_SHA256,
        "model_call_batch_size": GAN_POOL_BATCH_SIZE,
        "quota_per_class": driver.SAMPLES_PER_CLASS,
    }


def _lineage():
    return driver._lineage_summary(
        pool_identity=_pool_identity(),
        manifest=_gan_manifest(),
        source_commit=driver.GENERATOR_SOURCE_COMMIT,
    )


# ---------------------------------------------------------------------------
# GAN manifest authentication
# ---------------------------------------------------------------------------


def test_accepted_gan_manifest_identity_and_batch_count(monkeypatch):
    manifest = _gan_manifest()
    raw = driver._canonical(manifest)
    digest = _sha(raw)
    monkeypatch.setattr(driver, "POOL_MANIFEST_SHA256", digest)

    observed = driver._authenticate_gan_manifest(raw, expected_sha=digest)
    assert len(observed["batches"]) == driver.POOL_BATCHES
    assert observed["class_mapping"] == ["0", "1", "2", "3", "4"]
    assert observed["quota_per_class"] == driver.SAMPLES_PER_CLASS
    assert observed["checkpoint_run_identity"] == (
        driver.CHECKPOINT_RUN_IDENTITY
    )
    assert observed["checkpoint_sha256"] == driver.POOL_CHECKPOINT_SHA256
    assert observed["executable_source_commit"] == (
        driver.GENERATOR_SOURCE_COMMIT
    )
    assert observed["recipe_sha256"] == driver.GENERATOR_RECIPE_SHA256

    with pytest.raises(ValueError, match="digest mismatch"):
        driver._authenticate_gan_manifest(raw, expected_sha="0" * 64)
    monkeypatch.setattr(driver, "POOL_MANIFEST_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="accepted evidence"):
        driver._authenticate_gan_manifest(raw, expected_sha=digest)


@pytest.mark.parametrize(
    "field,value",
    (
        ("dataset_id", "ustc_tfc2016"),
        ("family", "maf"),
        ("experimental_seed", 43),
        ("class_mapping", ["0", "1", "2", "3"]),
        ("checkpoint_sha256", "f" * 64),
        ("checkpoint_run_identity", "commitdeadbee_attempt01"),
        ("executable_source_commit", "f" * 40),
        ("recipe_sha256", "e" * 64),
        ("complete_pool", False),
        ("status", "INCOMPLETE"),
        ("authoritative_artifact", False),
        ("batches_per_class", 124),
        ("model_call_batch_size", 8),
    ),
)
def test_gan_manifest_rejects_altered_identity(field, value, monkeypatch):
    manifest = _gan_manifest()
    manifest[field] = value
    raw = driver._canonical(manifest)
    monkeypatch.setattr(driver, "POOL_MANIFEST_SHA256", _sha(raw))
    with pytest.raises(ValueError, match="identity mismatch"):
        driver._authenticate_gan_manifest(raw, expected_sha=_sha(raw))


def test_gan_manifest_rejects_incomplete_batch_inventory(monkeypatch):
    manifest = _gan_manifest()
    manifest["batches"].pop()
    raw = driver._canonical(manifest)
    monkeypatch.setattr(driver, "POOL_MANIFEST_SHA256", _sha(raw))
    with pytest.raises(ValueError, match="625 ordered batches"):
        driver._authenticate_gan_manifest(raw, expected_sha=_sha(raw))


def test_maf_manifest_cannot_be_presented_as_gan(monkeypatch):
    maf_manifest = {
        "schema_version": 1,
        "dataset_id": driver.DATASET_ID,
        "family": "maf",
        "experimental_seed": driver.EXPERIMENTAL_SEED,
        "recipe_sha256": "a" * 64,
        "checkpoint_sha256": "b" * 64,
        "batches": [{}] * 1_250,
    }
    raw = driver._canonical(maf_manifest)
    monkeypatch.setattr(driver, "POOL_MANIFEST_SHA256", _sha(raw))
    with pytest.raises(ValueError, match="identity mismatch"):
        driver._authenticate_gan_manifest(raw, expected_sha=_sha(raw))


def test_gan_pool_view_accepts_complete_625_batch_interface(tmp_path):
    manifest = _gan_manifest()
    path, _, digest = _write_manifest(tmp_path, manifest)
    pool = GANPoolView(
        path,
        manifest_sha256=digest,
        family=driver.FAMILY,
        experimental_seed=driver.EXPERIMENTAL_SEED,
        recipe_sha256=driver.GENERATOR_RECIPE_SHA256,
        checkpoint_sha256=driver.POOL_CHECKPOINT_SHA256,
    )
    assert pool.size == driver.POOL_SAMPLES
    identity = pool.identity
    assert identity["manifest_sha256"] == digest
    assert identity["class_mapping"] == ["0", "1", "2", "3", "4"]
    assert identity["quota_per_class"] == driver.SAMPLES_PER_CLASS
    assert identity["checkpoint_run_identity"] == (
        driver.CHECKPOINT_RUN_IDENTITY
    )


def test_gan_pool_view_rejects_duplicate_or_incomplete_coverage(tmp_path):
    manifest = _gan_manifest()
    duplicate = json.loads(json.dumps(manifest))
    duplicate["batches"][1] = dict(duplicate["batches"][0])
    path, _, digest = _write_manifest(tmp_path, duplicate)
    with pytest.raises(ValueError, match="duplicate GAN batch coverage"):
        GANPoolView(
            path, manifest_sha256=digest, family=driver.FAMILY,
            experimental_seed=driver.EXPERIMENTAL_SEED,
            recipe_sha256=driver.GENERATOR_RECIPE_SHA256,
            checkpoint_sha256=driver.POOL_CHECKPOINT_SHA256,
        )

    shorter = json.loads(json.dumps(manifest))
    shorter["batches"].pop()
    path, _, digest = _write_manifest(tmp_path, shorter)
    with pytest.raises(ValueError, match="625 batch records"):
        GANPoolView(
            path, manifest_sha256=digest, family=driver.FAMILY,
            experimental_seed=driver.EXPERIMENTAL_SEED,
            recipe_sha256=driver.GENERATOR_RECIPE_SHA256,
            checkpoint_sha256=driver.POOL_CHECKPOINT_SHA256,
        )


def test_pool_preflight_rejects_incomplete_publication_markers(tmp_path):
    manifest = _gan_manifest()
    path, _, digest = _write_manifest(tmp_path, manifest)
    pool = GANPoolView(
        path, manifest_sha256=digest, family=driver.FAMILY,
        experimental_seed=driver.EXPERIMENTAL_SEED,
        recipe_sha256=driver.GENERATOR_RECIPE_SHA256,
        checkpoint_sha256=driver.POOL_CHECKPOINT_SHA256,
    )
    (tmp_path / "FAILURE.json").write_bytes(b"{}")
    with pytest.raises(RuntimeError, match="marked incomplete"):
        pool.preflight()
    (tmp_path / "FAILURE.json").unlink()
    (tmp_path / ".manifest.pending").write_bytes(b"{}")
    with pytest.raises(RuntimeError, match="publication is incomplete"):
        pool.preflight()


# ---------------------------------------------------------------------------
# GAN lineage
# ---------------------------------------------------------------------------


def test_lineage_summary_binds_pool_checkpoint_and_source():
    summary = driver._lineage_summary(
        pool_identity=_pool_identity(),
        manifest=_gan_manifest(),
        source_commit=driver.GENERATOR_SOURCE_COMMIT,
    )
    assert summary["schema"] == driver.LINEAGE_SCHEMA
    assert summary["dataset_id"] == driver.DATASET_ID
    assert summary["family"] == driver.FAMILY
    assert summary["experimental_seed"] == driver.EXPERIMENTAL_SEED
    assert summary["pool_manifest_sha256"] == driver.POOL_MANIFEST_SHA256
    assert summary["pool_checkpoint_sha256"] == driver.POOL_CHECKPOINT_SHA256
    assert summary["checkpoint_run_identity"] == (
        driver.CHECKPOINT_RUN_IDENTITY
    )
    assert summary["generator_source_commit"] == (
        driver.GENERATOR_SOURCE_COMMIT
    )
    assert summary["lineage_authentication"] == "DRIVER_VERIFIED"
    assert summary["synthesis_record_authentication"] == (
        "NOT_ARTIFACT_VERIFIED"
    )
    assert summary["complete_evaluation"] is False
    assert len(summary["sha256"]) == 64


def test_lineage_summary_rejects_conflicting_identities():
    good = {
        "pool_identity": _pool_identity(),
        "manifest": _gan_manifest(),
        "source_commit": driver.GENERATOR_SOURCE_COMMIT,
    }
    with pytest.raises(ValueError, match="lineage"):
        driver._lineage_summary(**{**good, "source_commit": "f" * 40})
    with pytest.raises(ValueError, match="lineage"):
        driver._lineage_summary(**{
            **good,
            "pool_identity": {
                **_pool_identity(), "checkpoint_sha256": "f" * 64,
            },
        })
    with pytest.raises(ValueError, match="lineage"):
        driver._lineage_summary(**{
            **good,
            "manifest": {**_gan_manifest(), "status": "INCOMPLETE"},
        })
    with pytest.raises(ValueError, match="lineage"):
        driver._lineage_summary(**{
            **good,
            "manifest": {
                **_gan_manifest(),
                "checkpoint_run_identity": "commitdeadbee_attempt01",
            },
        })


# ---------------------------------------------------------------------------
# Source, recipes and split identity
# ---------------------------------------------------------------------------


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
        driver.subprocess, "check_output",
        lambda *a, **k: " M scripts/rogue.py\n",
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


def test_recipes_are_exactly_the_accepted_bytes():
    recipes = driver._load_recipes(REPOSITORY_ROOT)
    assert _sha(recipes["generator"]) == driver.GENERATOR_RECIPE_SHA256
    assert _sha(recipes["classifier"]) == driver.CLASSIFIER_RECIPE_SHA256
    assert _sha(recipes["metric"]) == driver.METRIC_RECIPE_SHA256
    # The GAN synthesis recipe is not the MAF generator recipe.
    assert driver.GENERATOR_RECIPE_SHA256 == (
        "98f9ef427e1b2a793636d448fe3f272893a24848e4b7257764964015e19987a9"
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
    assert identity["class_count"] == driver.CLASS_COUNT

    with pytest.raises(AssertionError, match="test split accessed"):
        driver._verify_split_identities(adapter, include_test=True)


# ---------------------------------------------------------------------------
# Views and paired evidence
# ---------------------------------------------------------------------------


def test_training_phase_builds_only_train_and_validation_views():
    requested = []
    combined_arguments = {}

    def real_view(**arguments):
        requested.append(arguments["split"])
        return SimpleNamespace(
            identity={"split": arguments["split"]},
            size=driver.SPLIT_COUNTS[arguments["split"]],
        )

    def combined(*, real, synthetic):
        combined_arguments["synthetic"] = synthetic
        return SimpleNamespace(identity={"real": real.identity})

    components = {"real_view": real_view, "combined": combined}
    pool = SimpleNamespace(identity=_pool_identity())
    session = {
        "adapter": object(),
        "provenance_sha256": "a" * 64,
        "pool": pool,
    }
    real, validation, augmented = driver._build_training_views(
        session, components,
    )
    assert requested == ["train", "validation"]
    assert "test" not in requested
    assert combined_arguments["synthetic"] is pool
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


def test_paired_evidence_requires_consumption_and_gan_identity():
    augmented_identity = {"synthetic": _pool_identity()}
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

    wrong_pool = _paired_evidence()
    with pytest.raises(ValueError, match="accepted GAN pool"):
        driver._require_training_evidence(
            wrong_pool,
            augmented_identity={
                "synthetic": {
                    **_pool_identity(), "manifest_sha256": "9" * 64,
                },
            },
        )

    test_access = _paired_evidence()
    test_access["test_evaluated"] = True
    with pytest.raises(ValueError, match="pre-test"):
        driver._require_training_evidence(
            test_access, augmented_identity=augmented_identity,
        )


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


def test_unified_evaluation_must_verify_all_groups_and_stay_incomplete():
    good = {
        "complete_evaluation": False,
        "verified_group_count": 7,
        "required_status_blockers": [],
        "lineage_blockers": [],
        "verified_pool_consumption": {"schema": "verified"},
    }
    driver._require_verified_evaluation(good)

    for mutation in (
        {"complete_evaluation": True},
        {"verified_group_count": 6},
        {"required_status_blockers": [{}]},
        {"lineage_blockers": [{}]},
        {"verified_pool_consumption": None},
    ):
        with pytest.raises(ValueError, match="complete_evaluation=false"):
            driver._require_verified_evaluation({**good, **mutation})


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
# Preflight phase
# ---------------------------------------------------------------------------


def _preflight_session(*, verified_batches=625, verified_images=10_000):
    pool = SimpleNamespace(identity=_pool_identity())
    return {
        "head": "c" * 40,
        "config_raw": b"{}",
        "split_evidence": {},
        "identity": _pool_identity(),
        "pool": pool,
        "pool_evidence": {
            "verified_batches": verified_batches,
            "verified_images": verified_images,
        },
        "pool_sha256": driver.POOL_MANIFEST_SHA256,
        "lineage": _lineage(),
        "recipes": driver._load_recipes(REPOSITORY_ROOT),
    }


def test_preflight_covers_complete_gan_pool_and_stays_incomplete(
    monkeypatch, capsys,
):
    session = _preflight_session()
    monkeypatch.setattr(
        driver, "_prepare_inputs",
        lambda args, *, include_test_identity: session,
    )
    summary = driver._run_preflight(SimpleNamespace())
    assert summary["pool"]["verified_batches"] == driver.POOL_BATCHES
    assert summary["pool"]["verified_images"] == driver.POOL_SAMPLES
    assert summary["generator_identity"]["family"] == driver.FAMILY
    assert summary["generator_lineage"]["schema"] == driver.LINEAGE_SCHEMA
    assert summary["complete_evaluation"] is False
    assert summary["full_benchmark_go"] is False
    assert "CIC_GAN_SEED42_EVALUATION_PREFLIGHT=PASS" in (
        capsys.readouterr().out
    )


def test_preflight_rejects_partial_pool_coverage(monkeypatch):
    session = _preflight_session(verified_batches=624)
    monkeypatch.setattr(
        driver, "_prepare_inputs",
        lambda args, *, include_test_identity: session,
    )
    with pytest.raises(RuntimeError, match="did not cover the pool"):
        driver._run_preflight(SimpleNamespace())


# ---------------------------------------------------------------------------
# Bounded mocked composition of the actual train/restore entry points
# ---------------------------------------------------------------------------


def _fake_session(ordering):
    provenance = "a" * 64
    pool_identity = _pool_identity()
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
        "root": REPOSITORY_ROOT,
        "head": "c" * 40,
        "sources": {"src/example.py": "d" * 64},
        "config_raw": b"{}",
        "recipes": driver._load_recipes(REPOSITORY_ROOT),
        "components": {"real_view": real_view, "combined": combined},
        "adapter": _FakeAdapter(),
        "identity": _pool_identity(),
        "provenance_sha256": provenance,
        "split_evidence": {},
        "pool": SimpleNamespace(identity=pool_identity),
        "pool_raw": b"",
        "pool_sha256": driver.POOL_MANIFEST_SHA256,
        "lineage": _lineage(),
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


def _patch_train(monkeypatch, session, written, captured, *, fail_on=None):
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
                    "schema":
                        "trustforge.paper01.augmented-pool-consumption.v1",
                },
            },
        },
    }

    def prepare(**kwargs):
        captured["paired_arguments"] = kwargs
        return paired

    monkeypatch.setattr(
        paper01_classifier_tensorflow, "prepare_paired_classifiers", prepare,
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
    component_calls=None, tamper_arm=None,
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
    monkeypatch.setenv("SLURM_JOB_ID", "1234")

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
        selected = expected_selected_weights_sha256
        if tamper_arm is not None and name == tamper_arm:
            selected = "0" * 64
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
            "report": {"selected_weights_sha256": selected},
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
    captured = {}
    _patch_train(monkeypatch, session, written, captured)
    driver._run_train(SimpleNamespace(
        run_dir=str(run), expected_head=session["head"], weights="/fake",
    ))
    return ordering, session, run, written, captured


def test_actual_train_entry_point_publishes_gan_receipt(
    tmp_path, monkeypatch, capsys,
):
    ordering, session, run, written, captured = _run_mocked_train(
        tmp_path, monkeypatch,
    )
    assert session["splits"] == ["train", "validation"]
    assert "test" not in session["splits"]

    # Accepted classifier configuration and evaluator seed are reused as-is.
    arguments = captured["paired_arguments"]
    assert arguments["evaluator_seed"] == driver.EXPERIMENTAL_SEED == 42
    assert arguments["recipe_raw"] == session["recipes"]["classifier"]
    assert arguments["real_training"].identity["split"] == "train"
    assert arguments["augmented_training"].identity["synthetic"][
        "manifest_sha256"
    ] == driver.POOL_MANIFEST_SHA256

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
    assert receipt["generator_lineage_sha256"] == session["lineage"]["sha256"]
    assert receipt["evaluator_seed"] == 42
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

    lineage = json.loads((run / "lineage_evidence.json").read_bytes())
    assert lineage["schema"] == driver.LINEAGE_SCHEMA
    assert lineage["pool_checkpoint_sha256"] == driver.POOL_CHECKPOINT_SHA256
    assert lineage["pool_manifest_sha256"] == driver.POOL_MANIFEST_SHA256
    assert lineage["checkpoint_run_identity"] == (
        driver.CHECKPOINT_RUN_IDENTITY
    )

    assert written.index("train_report.json") < written.index(
        "expectations.json"
    )
    assert "CIC_GAN_SEED42_EVALUATION_TRAIN=PASS" in capsys.readouterr().out


def test_training_receipt_is_withheld_when_train_report_fails(
    tmp_path, monkeypatch,
):
    session = _fake_session({})
    run = tmp_path / "run"
    written = []
    captured = {}
    _patch_train(
        monkeypatch, session, written, captured, fail_on="train_report.json",
    )
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
    ordering, session, run, _, _ = _run_mocked_train(tmp_path, monkeypatch)
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
    linkage = captured["augmented_pool_linkage"]
    assert linkage["family"] == driver.FAMILY
    assert linkage["manifest_sha256"] == driver.POOL_MANIFEST_SHA256
    assert linkage["checkpoint_sha256"] == driver.POOL_CHECKPOINT_SHA256
    assert captured["augmented_checkpoint_evidence"][
        "expected_expectations_sha256"
    ] == _sha(receipt_raw)

    fidelity = calls["fidelity"]
    assert isinstance(fidelity["real_validation_view"], RealSplitView)
    assert fidelity["real_validation_labels"].shape == (
        driver.VALIDATION_SIZE,
    )
    assert fidelity["synthetic_labels"].shape == (driver.POOL_SAMPLES,)
    assert np.bincount(fidelity["synthetic_labels"]).tolist() == (
        [driver.SAMPLES_PER_CLASS] * driver.CLASS_COUNT
    )
    assert isinstance(fidelity["selection_raw"], bytes)
    assert calls["ms_ssim"]["labels"].shape == (driver.POOL_SAMPLES,)
    assert isinstance(calls["ms_ssim"]["selection_raw"], bytes)

    evaluation = json.loads((run / "evaluation.json").read_bytes())
    assert evaluation["complete_evaluation"] is False
    restore_report = json.loads((run / "restore_report.json").read_bytes())
    assert restore_report["complete_evaluation"] is False
    assert restore_report["full_benchmark_go"] is False
    assert restore_report["generator_lineage"]["schema"] == (
        driver.LINEAGE_SCHEMA
    )
    out = capsys.readouterr().out
    assert "CIC_GAN_SEED42_EVALUATION_RESTORE=PASS" in out
    assert "CIC_GAN_SEED42_BOUNDED_EVALUATION=PASS" in out
    assert "COMPLETE_EVALUATION=False" in out
    assert "FULL_BENCHMARK_GO=NO" in out


def test_restore_rejects_tampered_receipt(tmp_path, monkeypatch):
    ordering, session, run, _, _ = _run_mocked_train(tmp_path, monkeypatch)
    receipt_raw = (run / "expectations.json").read_bytes()
    receipt = json.loads(receipt_raw)
    receipt["arms"]["real_plus_synthetic"]["selected_weights_sha256"] = (
        "9" * 64
    )
    (run / "expectations.json").write_bytes(driver._canonical(receipt))
    captured = {}
    _patch_restore(monkeypatch, session, ordering, captured)
    with pytest.raises(ValueError, match="receipt digest mismatch"):
        driver._run_restore(SimpleNamespace(
            run_dir=str(run), expected_head=session["head"], weights="/fake",
            receipt_sha256=_sha(receipt_raw),
        ))


def test_restore_rejects_noncanonical_pool_consumption(
    tmp_path, monkeypatch,
):
    ordering, session, run, _, _ = _run_mocked_train(tmp_path, monkeypatch)
    receipt_raw = (run / "expectations.json").read_bytes()
    consumption = json.loads((run / "pool_consumption.json").read_bytes())
    (run / "pool_consumption.json").write_text(
        json.dumps(consumption, indent=2), encoding="utf-8",
    )
    captured = {}
    _patch_restore(monkeypatch, session, ordering, captured)
    with pytest.raises(ValueError, match="not canonical"):
        driver._run_restore(SimpleNamespace(
            run_dir=str(run), expected_head=session["head"], weights="/fake",
            receipt_sha256=_sha(receipt_raw),
        ))


def test_restore_rejects_mismatched_restored_weights(tmp_path, monkeypatch):
    ordering, session, run, _, _ = _run_mocked_train(tmp_path, monkeypatch)
    receipt_raw = (run / "expectations.json").read_bytes()
    captured = {}
    _patch_restore(
        monkeypatch, session, ordering, captured,
        tamper_arm="real_plus_synthetic",
    )
    with pytest.raises(ValueError, match="restored or linked selected weights"):
        driver._run_restore(SimpleNamespace(
            run_dir=str(run), expected_head=session["head"], weights="/fake",
            receipt_sha256=_sha(receipt_raw),
        ))


def test_restore_fails_closed_without_success_marker_on_incomplete_result(
    tmp_path, monkeypatch, capsys,
):
    ordering, session, run, _, _ = _run_mocked_train(tmp_path, monkeypatch)
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
    assert "CIC_GAN_SEED42_BOUNDED_EVALUATION=PASS" not in (
        capsys.readouterr().out
    )


# ---------------------------------------------------------------------------
# CLI and launcher boundaries
# ---------------------------------------------------------------------------


def _base_arguments():
    return [
        "--repository-root", str(REPOSITORY_ROOT),
        "--expected-head", "a" * 40,
        "--dataset-config", "/data/cic_config.json",
        "--config-sha256", "b" * 64,
        "--pool-manifest", "/data/pool/manifest.json",
        "--pool-manifest-sha256", "c" * 64,
        "--weights", "/data/inception.h5",
    ]


def test_cli_rejects_unapproved_seed_and_incomplete_configuration():
    with pytest.raises(SystemExit):
        driver._parser().parse_args(
            ["--preflight-only", "--experimental-seed", "43", *_base_arguments()]
        )

    with pytest.raises(SystemExit, match="preflight does not accept"):
        driver.main(
            ["--preflight-only", *_base_arguments(), "--run-dir", "/tmp/run"]
        )
    with pytest.raises(SystemExit, match="--run-dir is required"):
        driver.main(["--phase", "train", *_base_arguments()])
    with pytest.raises(SystemExit, match="train phase does not accept"):
        driver.main([
            "--phase", "train", *_base_arguments(),
            "--run-dir", "/tmp/run", "--receipt-sha256", "d" * 64,
        ])
    with pytest.raises(SystemExit, match="restore phase requires"):
        driver.main([
            "--phase", "restore", *_base_arguments(), "--run-dir", "/tmp/run",
        ])


def test_gan_launcher_is_fail_closed_pending_execution_authorization():
    path = (
        REPOSITORY_ROOT
        / "slurm/paper01_revision_2026/gan_cic_seed42_bounded_evaluation.slurm"
    )
    text = path.read_text(encoding="utf-8")
    assert "GAN_EVAL_LAUNCHER_BLOCKED" in text
    assert "RESEARCHER_DECISION_PENDING" in text
    assert "authorization_status" in text
    assert "APPROVED" in text
    assert "sbatch" not in text
    assert "run_paper01_cic_gan_seed42_bounded_evaluation.py" in text
    assert "--preflight-only" in text
    assert "--phase train" in text
    assert "--phase restore" in text
