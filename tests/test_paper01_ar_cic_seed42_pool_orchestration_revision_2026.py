"""Focused tests for the bounded CIC AR seed-42 pool coordinator."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import shutil
from pathlib import Path
import subprocess

import numpy as np
import pytest

from scripts import run_paper01_ar_cic_seed42_pool as coordinator
from trustforge.models.paper01_revision_2026.ar_maf_execution_inputs import (
    synthesis_batch_seed,
)
from trustforge.models.paper01_revision_2026.ar_cic_pool_authorization import (
    RESOURCE_FIELDS,
)


ACCEPTED_CHECKPOINT = (
    "d58a64ad8cadc93e745945191e374a9c27b5668cf569f5bac4761a172b29a0f0"
)
ACCEPTED_TRAINING_COMMIT = "e4473abcdbc53f0de3a01ebe48858f2f3c7b841a"
ACCEPTED_RECIPE = (
    "202db292bb18d3e6f04f6f8cc36658884994c3bd9f4f48e2e93e181d348ea478"
)
ACCEPTED_CONFIG = (
    "ea1672eba0183b692ed29ffe7e8f4b9a707d487049c27623a38ddddff1a99f8b"
)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _sha_file(path):
    return _sha(Path(path).read_bytes())


@dataclass
class Fixture:
    repository: Path
    run_root: Path
    checkpoint_parent: Path
    dataset_config: Path
    head: str
    recipe_sha: str
    config_sha: str
    plan: dict


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    repository = (tmp_path / "repository").resolve()
    repository.mkdir()
    recipe = repository / "recipe.json"
    recipe.write_text('{"fixture": "ar-recipe"}\n', encoding="utf-8")
    (repository / "README").write_text("fixture\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(
        ["git", "-C", str(repository), "add", "."], check=True,
    )
    subprocess.run(
        [
            "git", "-C", str(repository),
            "-c", "user.name=Fixture",
            "-c", "user.email=fixture@example.invalid",
            "commit", "-qm", "fixture",
        ],
        check=True,
    )
    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True,
    ).strip()

    checkpoint_parent = (tmp_path / "checkpoints").resolve()
    (checkpoint_parent / coordinator.GENERATION).mkdir(parents=True)

    dataset_config = (tmp_path / "dataset_config.json").resolve()
    dataset_config.write_bytes(b'{"fixture": "config"}\n')
    config_sha = _sha_file(dataset_config)

    monkeypatch.setattr(coordinator, "RECIPE_PATH", "recipe.json")
    monkeypatch.setattr(coordinator, "RECIPE_SHA256", _sha_file(recipe))
    monkeypatch.setattr(coordinator, "CONFIG_SHA256", config_sha)

    run_root = (tmp_path / "run").resolve()
    plan = coordinator.build_plan(
        repository_root=repository,
        expected_head=head,
        run_root=run_root,
        checkpoint_parent=checkpoint_parent,
        dataset_config=dataset_config,
        config_sha256=config_sha,
    )
    return Fixture(
        repository=repository,
        run_root=run_root,
        checkpoint_parent=checkpoint_parent,
        dataset_config=dataset_config,
        head=head,
        recipe_sha=_sha_file(recipe),
        config_sha=config_sha,
        plan=plan,
    )


def _report(fixture, task, batch_index, archive_sha, **overrides):
    seed = synthesis_batch_seed(
        dataset_id="cicmaldroid2020",
        family="ar",
        experimental_seed=42,
        class_id=task["class_id"],
        batch_index=batch_index,
    )
    report = {
        "dataset_id": "cicmaldroid2020",
        "family": "ar",
        "experimental_seed": 42,
        "class_id": task["class_id"],
        "batch_index": batch_index,
        "sampling_seed": seed,
        "recipe_sha256": fixture.recipe_sha,
        "checkpoint_sha256": ACCEPTED_CHECKPOINT,
        "synthesis_commit": fixture.head,
        "config_sha256": fixture.config_sha,
        "samples_sha256": archive_sha,
        "shape": [8, 40, 40, 3],
        "dtype": "float32",
    }
    report.update(overrides)
    return report


def _write_report_workers(fixture, *, skip=None, extra=None, mutate=None):
    """Create report-valid workers with tiny placeholder archives."""
    run_root = fixture.run_root
    for task in fixture.plan["tasks"]:
        worker = coordinator.worker_directory(run_root, task)
        worker.mkdir(parents=True)
        for batch_index in range(task["batch_start"], task["batch_stop"]):
            if skip == (task["class_id"], batch_index):
                continue
            batch_dir = worker / f"batch-{batch_index:03d}"
            batch_dir.mkdir()
            archive = batch_dir / "samples.npz"
            raw = f"{task['class_id']}-{batch_index}".encode("ascii")
            archive.write_bytes(raw)
            report = _report(fixture, task, batch_index, _sha(raw))
            if mutate is not None:
                report = mutate(task, batch_index, report)
            (batch_dir / "report.json").write_text(
                json.dumps(report) + "\n", encoding="utf-8",
            )
        if extra == task["class_id"] and task["batch_start"] == 0:
            batch_dir = worker / "batch-249"
            batch_dir.mkdir()
            (batch_dir / "samples.npz").write_bytes(b"extra")
            (batch_dir / "report.json").write_text(
                json.dumps(_report(fixture, task, 249, _sha(b"extra"))) + "\n",
                encoding="utf-8",
            )


def _full_workers(fixture, *, invalid_first=False, invalid_image=False):
    """Create complete real-shape workers with per-class hard-linked archives."""
    archives = {}
    for class_id in range(5):
        images = np.zeros((8, 40, 40, 3), dtype=np.float32)
        class_ids = np.full(8, class_id, dtype=np.int32)
        if invalid_first and class_id == 0:
            class_ids = np.full(8, 1, dtype=np.int32)
        if invalid_image and class_id == 0:
            images = np.zeros((8, 40, 40, 2), dtype=np.float32)
        buffer = io.BytesIO()
        np.savez(buffer, images=images, class_ids=class_ids)
        raw = buffer.getvalue()
        archives[class_id] = (raw, _sha(raw))

    for task in fixture.plan["tasks"]:
        worker = coordinator.worker_directory(fixture.run_root, task)
        worker.mkdir(parents=True)
        raw, digest = archives[task["class_id"]]
        for batch_index in range(task["batch_start"], task["batch_stop"]):
            batch_dir = worker / f"batch-{batch_index:03d}"
            batch_dir.mkdir()
            (batch_dir / "samples.npz").write_bytes(raw)
            (batch_dir / "report.json").write_text(
                json.dumps(_report(fixture, task, batch_index, digest)) + "\n",
                encoding="utf-8",
            )


def _authorization_record(fixture):
    return {
        "schema": "trustforge.paper01.ar-cic-seed42-pool-authorization.v1",
        "paper": "paper01",
        "revision": "revision_2026",
        "family": "ar",
        "dataset_id": "cicmaldroid2020",
        "experimental_seed": 42,
        "attempt": 1,
        "checkpoint_sha256": fixture.plan["checkpoint_sha256"],
        "dataset_config_sha256": fixture.plan["config_sha256"],
        "recipe_sha256": fixture.plan["recipe_sha256"],
        "synthesis_commit": fixture.plan["synthesis_commit"],
        "run_root": fixture.plan["run_root"],
        "destination": fixture.plan["destination"],
        "expected_task_count": len(fixture.plan["tasks"]),
        "expected_batch_count": fixture.plan["total_batches"],
        "resources": {
            key: fixture.plan["resources"][key]
            for key in RESOURCE_FIELDS
        },
        "authorization_status": "APPROVED",
        "authorization_id": "fixture-ar-pool-authorization",
    }


def _write_authorization(fixture):
    path = fixture.run_root.parent / "authorization.json"
    path.write_text(
        json.dumps(
            _authorization_record(fixture),
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    path.chmod(0o600)
    return path


def test_production_bindings_remain_exact():
    assert coordinator.FAMILY == "ar"
    assert coordinator.DATASET_ID == "cicmaldroid2020"
    assert coordinator.EXPERIMENTAL_SEED == 42
    assert coordinator.CLASS_COUNT == 5
    assert coordinator.BATCHES_PER_CLASS == 250
    assert coordinator.BATCH_SIZE == 8
    assert coordinator.TOTAL_BATCHES == 1250
    assert coordinator.POOL_SAMPLES == 10000
    assert coordinator.TRAINING_COMMIT == ACCEPTED_TRAINING_COMMIT
    assert coordinator.CHECKPOINT_SHA256 == ACCEPTED_CHECKPOINT
    assert coordinator.GENERATION == "ar_cic_seed42"
    assert coordinator.RECIPE_SHA256 == ACCEPTED_RECIPE
    assert coordinator.CONFIG_SHA256 == ACCEPTED_CONFIG


def test_plan_has_exactly_1250_identities_and_250_per_class(fixture):
    tasks = fixture.plan["tasks"]
    assert len(tasks) == 50
    assert sum(task["batch_count"] for task in tasks) == 1250
    identities = {
        (task["class_id"], batch_index)
        for task in tasks
        for batch_index in range(task["batch_start"], task["batch_stop"])
    }
    assert len(identities) == 1250
    for class_id in range(5):
        assert sum(
            1 for task in tasks if task["class_id"] == class_id
        ) == 10
        assert sum(
            task["batch_count"] for task in tasks
            if task["class_id"] == class_id
        ) == 250


def test_partition_boundaries_do_not_change_sampling_identities():
    canonical = {
        synthesis_batch_seed(
            dataset_id="cicmaldroid2020", family="ar",
            experimental_seed=42, class_id=class_id,
            batch_index=batch_index,
        )
        for class_id in range(5)
        for batch_index in range(250)
    }
    for ranges_per_class in (5, 10, 25):
        tasks = coordinator.partition_ranges(ranges_per_class)
        seeds = {
            synthesis_batch_seed(
                dataset_id="cicmaldroid2020", family="ar",
                experimental_seed=42, class_id=task["class_id"],
                batch_index=batch_index,
            )
            for task in tasks
            for batch_index in range(task["batch_start"], task["batch_stop"])
        }
        assert seeds == canonical
        assert len(seeds) == 1250
        for task in tasks:
            assert task["sampling_seed_start"] == synthesis_batch_seed(
                dataset_id="cicmaldroid2020", family="ar",
                experimental_seed=42, class_id=task["class_id"],
                batch_index=task["batch_start"],
            )
            assert task["sampling_seed_end"] == synthesis_batch_seed(
                dataset_id="cicmaldroid2020", family="ar",
                experimental_seed=42, class_id=task["class_id"],
                batch_index=task["batch_stop"] - 1,
            )


def test_plan_rejects_source_mismatch(fixture):
    with pytest.raises(RuntimeError, match="HEAD differs"):
        coordinator.build_plan(
            repository_root=fixture.repository,
            expected_head="f" * 40,
            run_root=fixture.run_root,
            checkpoint_parent=fixture.checkpoint_parent,
            dataset_config=fixture.dataset_config,
            config_sha256=fixture.config_sha,
        )


def test_plan_rejects_config_digest_mismatch(fixture):
    with pytest.raises(ValueError, match="configuration digest"):
        coordinator.build_plan(
            repository_root=fixture.repository,
            expected_head=fixture.head,
            run_root=fixture.run_root,
            checkpoint_parent=fixture.checkpoint_parent,
            dataset_config=fixture.dataset_config,
            config_sha256="0" * 64,
        )


def test_plan_rejects_recipe_digest_mismatch(fixture, monkeypatch):
    monkeypatch.setattr(coordinator, "RECIPE_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="recipe digest"):
        coordinator.build_plan(
            repository_root=fixture.repository,
            expected_head=fixture.head,
            run_root=fixture.run_root,
            checkpoint_parent=fixture.checkpoint_parent,
            dataset_config=fixture.dataset_config,
            config_sha256=fixture.config_sha,
        )


def test_plan_rejects_missing_checkpoint_generation(fixture):
    for child in (fixture.checkpoint_parent / coordinator.GENERATION).iterdir():
        child.unlink()
    (fixture.checkpoint_parent / coordinator.GENERATION).rmdir()
    with pytest.raises(ValueError, match="checkpoint generation"):
        coordinator.build_plan(
            repository_root=fixture.repository,
            expected_head=fixture.head,
            run_root=fixture.run_root,
            checkpoint_parent=fixture.checkpoint_parent,
            dataset_config=fixture.dataset_config,
            config_sha256=fixture.config_sha,
        )


def test_emit_plan_refuses_existing_run_root(fixture):
    fixture.run_root.mkdir()
    with pytest.raises(FileExistsError, match="fresh destination"):
        coordinator.main([
            "--emit-plan",
            "--repository-root", str(fixture.repository),
            "--expected-head", fixture.head,
            "--run-root", str(fixture.run_root),
            "--checkpoint-parent", str(fixture.checkpoint_parent),
            "--dataset-config", str(fixture.dataset_config),
            "--config-sha256", fixture.config_sha,
        ])


def test_emit_plan_creates_no_directories(fixture, capsys):
    coordinator.main([
        "--emit-plan",
        "--repository-root", str(fixture.repository),
        "--expected-head", fixture.head,
        "--run-root", str(fixture.run_root),
        "--checkpoint-parent", str(fixture.checkpoint_parent),
        "--dataset-config", str(fixture.dataset_config),
        "--config-sha256", fixture.config_sha,
    ])
    output = json.loads(capsys.readouterr().out)
    assert output["execution_authorized"] is False
    assert output["total_batches"] == 1250
    assert not fixture.run_root.exists()


def test_worker_partition_rejects_missing_batch(fixture):
    _write_report_workers(fixture, skip=(2, 100))
    with pytest.raises(ValueError, match="range coverage differs"):
        coordinator.validate_worker_partition(fixture.plan, fixture.run_root)


def test_worker_partition_rejects_extra_batch(fixture):
    _write_report_workers(fixture, extra=3)
    with pytest.raises(ValueError, match="range coverage differs"):
        coordinator.validate_worker_partition(fixture.plan, fixture.run_root)


def test_worker_partition_rejects_wrong_class(fixture):
    def mutate(task, batch_index, report):
        if (task["class_id"], batch_index) == (0, 0):
            report["class_id"] = 1
        return report

    _write_report_workers(fixture, mutate=mutate)
    with pytest.raises(ValueError, match="class/index mismatch"):
        coordinator.validate_worker_partition(fixture.plan, fixture.run_root)


def test_worker_partition_rejects_wrong_seed(fixture):
    def mutate(task, batch_index, report):
        if (task["class_id"], batch_index) == (0, 0):
            report["sampling_seed"] += 1
        return report

    _write_report_workers(fixture, mutate=mutate)
    with pytest.raises(ValueError, match="sampling seed mismatch"):
        coordinator.validate_worker_partition(fixture.plan, fixture.run_root)


def test_worker_partition_rejects_wrong_source(fixture):
    def mutate(task, batch_index, report):
        if (task["class_id"], batch_index) == (0, 0):
            report["synthesis_commit"] = "a" * 40
        return report

    _write_report_workers(fixture, mutate=mutate)
    with pytest.raises(ValueError, match="identity mismatch"):
        coordinator.validate_worker_partition(fixture.plan, fixture.run_root)


@pytest.mark.parametrize(
    "field",
    ("checkpoint_sha256", "recipe_sha256", "config_sha256"),
)
def test_worker_partition_rejects_wrong_provenance(fixture, field):
    def mutate(task, batch_index, report):
        if (task["class_id"], batch_index) == (0, 0):
            report[field] = "0" * 64
        return report

    _write_report_workers(fixture, mutate=mutate)
    with pytest.raises(ValueError, match="identity mismatch"):
        coordinator.validate_worker_partition(fixture.plan, fixture.run_root)


def test_worker_partition_rejects_missing_worker(fixture):
    _write_report_workers(fixture)
    missing = coordinator.worker_directory(
        fixture.run_root, fixture.plan["tasks"][0],
    )
    shutil.rmtree(missing)
    with pytest.raises(ValueError, match="worker partition directory set"):
        coordinator.validate_worker_partition(fixture.plan, fixture.run_root)


def test_finalize_requires_authorization(fixture):
    _write_report_workers(fixture)
    missing = fixture.run_root.parent / "missing-authorization.json"
    with pytest.raises(PermissionError, match="authorization"):
        coordinator.finalize_pool(
            plan=fixture.plan,
            run_root=fixture.run_root,
            destination=fixture.plan["destination"],
            authorization_path=missing,
        )
    assert not Path(fixture.plan["destination"]).exists()


def test_finalize_publishes_only_complete_pool_and_verifies(fixture):
    authorization = _write_authorization(fixture)
    _full_workers(fixture)

    summary = coordinator.finalize_pool(
        plan=fixture.plan,
        run_root=fixture.run_root,
        destination=fixture.plan["destination"],
        authorization_path=authorization,
    )
    assert summary["batch_count"] == 1250
    assert summary["pool_samples"] == 10000
    assert summary["class_counts"] == [2000] * 5
    assert summary["publication_authorized"] is True
    assert summary["publication_authorization_id"] == (
        "fixture-ar-pool-authorization"
    )
    manifest = Path(fixture.plan["destination"]) / "manifest.json"
    assert manifest.is_file()
    assert _sha_file(manifest) == summary["manifest_sha256"]

    verified = coordinator.verify_pool(manifest)
    assert verified["batch_count"] == 1250
    assert verified["class_counts"] == [2000] * 5

    with pytest.raises(FileExistsError, match="already exists"):
        coordinator.finalize_pool(
            plan=fixture.plan,
            run_root=fixture.run_root,
            destination=fixture.plan["destination"],
            authorization_path=authorization,
        )


def test_finalize_rejects_invalid_labels_before_manifest(fixture):
    authorization = _write_authorization(fixture)
    _full_workers(fixture, invalid_first=True)

    with pytest.raises(ValueError, match="class IDs mismatch"):
        coordinator.finalize_pool(
            plan=fixture.plan,
            run_root=fixture.run_root,
            destination=fixture.plan["destination"],
            authorization_path=authorization,
        )
    assert not (Path(fixture.plan["destination"]) / "manifest.json").exists()


def test_finalize_rejects_invalid_image_shape_before_manifest(fixture):
    authorization = _write_authorization(fixture)
    _full_workers(fixture, invalid_image=True)

    with pytest.raises(ValueError, match="invalid canonical batch"):
        coordinator.finalize_pool(
            plan=fixture.plan,
            run_root=fixture.run_root,
            destination=fixture.plan["destination"],
            authorization_path=authorization,
        )
    assert not (Path(fixture.plan["destination"]) / "manifest.json").exists()


def test_missing_batch_prevents_manifest(fixture):
    authorization = _write_authorization(fixture)
    _write_report_workers(fixture, skip=(4, 249))
    with pytest.raises(ValueError, match="range coverage differs"):
        coordinator.finalize_pool(
            plan=fixture.plan,
            run_root=fixture.run_root,
            destination=fixture.plan["destination"],
            authorization_path=authorization,
        )
    assert not (Path(fixture.plan["destination"]) / "manifest.json").exists()


def test_verify_rejects_changed_published_archive(fixture):
    authorization = _write_authorization(fixture)
    _full_workers(fixture)
    coordinator.finalize_pool(
        plan=fixture.plan,
        run_root=fixture.run_root,
        destination=fixture.plan["destination"],
        authorization_path=authorization,
    )
    manifest = Path(fixture.plan["destination"]) / "manifest.json"
    target = Path(fixture.plan["destination"]) / "c0-b000.npy"
    target.unlink()
    np.save(target, np.ones((8, 40, 40, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="digest mismatch"):
        coordinator.verify_pool(manifest)


def test_coordinator_has_no_slurm_submission_path():
    source = Path(coordinator.__file__).read_text(encoding="utf-8")
    assert "sbatch" not in source
    assert "srun" not in source
    assert "subprocess.run(" not in source
    assert "AR_CIC_POOL_PUBLICATION_ENABLED" not in source
    assert "--authorization" in source
