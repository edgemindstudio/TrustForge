"""Focused tests for AR CIC seed-42 production submission and authorization."""

from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from scripts import launch_paper01_ar_cic_seed42_pool as launcher
from scripts import run_paper01_ar_cic_seed42_pool as coordinator
from trustforge.models.paper01_revision_2026.ar_cic_pool_authorization import (
    RESOURCE_FIELDS,
    validate_ar_cic_pool_authorization,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ARRAY_SCRIPT = (
    REPOSITORY_ROOT
    / "slurm/paper01_revision_2026/ar_cic_seed42_pool.slurm"
)
FINALIZER_SCRIPT = (
    REPOSITORY_ROOT
    / "slurm/paper01_revision_2026/ar_cic_seed42_pool_finalize.slurm"
)
RANGE_SCRIPT = (
    REPOSITORY_ROOT
    / "slurm/paper01_revision_2026/ar_cic_range_worker.slurm"
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass
class Fixture:
    repository: Path
    run_root: Path
    checkpoint_parent: Path
    dataset_config: Path
    head: str
    plan: dict
    authorization: Path


def _record(plan):
    return {
        "schema": "trustforge.paper01.ar-cic-seed42-pool-authorization.v1",
        "paper": "paper01",
        "revision": "revision_2026",
        "family": "ar",
        "dataset_id": "cicmaldroid2020",
        "experimental_seed": 42,
        "attempt": 1,
        "checkpoint_sha256": plan["checkpoint_sha256"],
        "dataset_config_sha256": plan["config_sha256"],
        "recipe_sha256": plan["recipe_sha256"],
        "synthesis_commit": plan["synthesis_commit"],
        "run_root": plan["run_root"],
        "destination": plan["destination"],
        "expected_task_count": len(plan["tasks"]),
        "expected_batch_count": plan["total_batches"],
        "resources": {
            key: plan["resources"][key] for key in RESOURCE_FIELDS
        },
        "authorization_status": "APPROVED",
        "authorization_id": "fixture-ar-pool-authorization",
    }


def _write_record(path, record):
    path.write_text(
        json.dumps(record, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    path.chmod(0o600)
    return path


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
    config_sha = _sha(dataset_config)

    monkeypatch.setattr(coordinator, "RECIPE_PATH", "recipe.json")
    monkeypatch.setattr(coordinator, "RECIPE_SHA256", _sha(recipe))
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
    authorization = _write_record(
        tmp_path / "authorization.json", _record(plan),
    )
    return Fixture(
        repository=repository,
        run_root=run_root,
        checkpoint_parent=checkpoint_parent,
        dataset_config=dataset_config,
        head=head,
        plan=plan,
        authorization=authorization,
    )


def _main_arguments(fixture, *, submit=False, authorization=None):
    arguments = [
        "--repository-root", str(fixture.repository),
        "--expected-head", fixture.head,
        "--run-root", str(fixture.run_root),
        "--checkpoint-parent", str(fixture.checkpoint_parent),
        "--dataset-config", str(fixture.dataset_config),
        "--config-sha256", fixture.plan["config_sha256"],
        "--authorization", str(authorization or fixture.authorization),
    ]
    if submit:
        arguments.append("--submit")
    return arguments


def test_valid_authorization_is_accepted(fixture):
    record = validate_ar_cic_pool_authorization(
        fixture.authorization,
        plan=fixture.plan,
        run_root=fixture.run_root,
        destination=fixture.plan["destination"],
    )
    assert record["authorization_id"] == "fixture-ar-pool-authorization"
    assert record["expected_task_count"] == 50
    assert record["expected_batch_count"] == 1250


def test_missing_or_symlinked_authorization_is_rejected(fixture, tmp_path):
    missing = tmp_path / "missing.json"
    with pytest.raises(PermissionError, match="authorization"):
        validate_ar_cic_pool_authorization(
            missing,
            plan=fixture.plan,
            run_root=fixture.run_root,
            destination=fixture.plan["destination"],
        )
    link = tmp_path / "authorization-link.json"
    link.symlink_to(fixture.authorization)
    with pytest.raises(PermissionError, match="authorization"):
        validate_ar_cic_pool_authorization(
            link,
            plan=fixture.plan,
            run_root=fixture.run_root,
            destination=fixture.plan["destination"],
        )


def test_writable_or_malformed_authorization_is_rejected(fixture, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not-json", encoding="utf-8")
    bad.chmod(0o600)
    with pytest.raises(PermissionError, match="unreadable"):
        validate_ar_cic_pool_authorization(
            bad,
            plan=fixture.plan,
            run_root=fixture.run_root,
            destination=fixture.plan["destination"],
        )

    writable = _write_record(
        tmp_path / "writable.json", _record(fixture.plan),
    )
    writable.chmod(0o666)
    with pytest.raises(PermissionError, match="writable"):
        validate_ar_cic_pool_authorization(
            writable,
            plan=fixture.plan,
            run_root=fixture.run_root,
            destination=fixture.plan["destination"],
        )


def test_authorization_rejects_mismatched_identity(fixture, tmp_path):
    mutations = {
        "synthesis_commit": "a" * 40,
        "checkpoint_sha256": "0" * 64,
        "dataset_config_sha256": "0" * 64,
        "recipe_sha256": "0" * 64,
        "run_root": "/tmp/other-run",
        "destination": "/tmp/other-run/pool",
        "expected_task_count": 49,
        "expected_batch_count": 1249,
        "authorization_status": "PENDING",
        "authorization_id": "RESEARCHER_DECISION_PENDING",
    }
    for index, (field, value) in enumerate(mutations.items()):
        record = copy.deepcopy(_record(fixture.plan))
        record[field] = value
        path = _write_record(tmp_path / f"mutated-{index}.json", record)
        with pytest.raises(PermissionError):
            validate_ar_cic_pool_authorization(
                path,
                plan=fixture.plan,
                run_root=fixture.run_root,
                destination=fixture.plan["destination"],
            )

    resource_record = copy.deepcopy(_record(fixture.plan))
    resource_record["resources"] = {
        **resource_record["resources"], "tasks": 49,
    }
    path = _write_record(tmp_path / "mutated-resources.json", resource_record)
    with pytest.raises(PermissionError, match="resource differs"):
        validate_ar_cic_pool_authorization(
            path,
            plan=fixture.plan,
            run_root=fixture.run_root,
            destination=fixture.plan["destination"],
        )


def test_submitter_dry_run_submits_nothing(fixture, monkeypatch, capsys):
    called = []
    monkeypatch.setattr(
        launcher, "_sbatch",
        lambda arguments: called.append(arguments) or "11111",
    )
    launcher.main(_main_arguments(fixture))
    output = json.loads(capsys.readouterr().out)
    assert output["submission"] is False
    assert called == []
    assert not fixture.run_root.exists()
    worker = output["worker_command"]
    finalizer = output["finalizer_command"]
    assert "--array=0-49%2" in worker
    assert "--no-requeue" in worker
    assert "--dependency=afterok:<worker_array_job_id>" in finalizer


def test_submitter_requires_valid_authorization(fixture, monkeypatch, tmp_path):
    called = []
    monkeypatch.setattr(
        launcher, "_sbatch",
        lambda arguments: called.append(arguments) or "11111",
    )
    missing = tmp_path / "missing-authorization.json"
    with pytest.raises(PermissionError, match="authorization"):
        launcher.main(_main_arguments(
            fixture, submit=True, authorization=missing,
        ))
    assert called == []
    assert not fixture.run_root.exists()


def test_submitter_rejects_existing_run_root(fixture, monkeypatch):
    fixture.run_root.mkdir()
    called = []
    monkeypatch.setattr(
        launcher, "_sbatch",
        lambda arguments: called.append(arguments) or "11111",
    )
    with pytest.raises(FileExistsError, match="fresh destination"):
        launcher.main(_main_arguments(fixture, submit=True))
    assert called == []


def test_submitter_wires_array_and_afterok_finalizer(
    fixture, monkeypatch, capsys,
):
    calls = []

    def submit(arguments):
        calls.append(list(arguments))
        return "11111\n" if len(calls) == 1 else "22222\n"

    monkeypatch.setattr(launcher, "_sbatch", submit)
    monkeypatch.setattr(launcher.os, "environ", dict(launcher.os.environ))
    launcher.main(_main_arguments(fixture, submit=True))
    output = json.loads(capsys.readouterr().out)

    assert output["submission"] is True
    assert output["worker_job_id"] == "11111"
    assert output["finalizer_job_id"] == "22222"
    assert len(calls) == 2
    assert "--array=0-49%2" in calls[0]
    assert "--no-requeue" in calls[0]
    assert "--dependency=afterok:11111" in calls[1]
    assert not any("--force" in argument for argument in calls[0] + calls[1])
    assert not any("--requeue" in argument for argument in calls[0] + calls[1])
    assert (fixture.run_root / "workers").is_dir()
    assert (fixture.run_root / "logs").is_dir()


def test_partial_submission_reports_worker_id_and_failure(
    fixture, monkeypatch, capsys,
):
    calls = []

    def submit(arguments):
        calls.append(list(arguments))
        if len(calls) == 1:
            return "11111\n"
        raise subprocess.CalledProcessError(
            1, ["sbatch", *arguments], stderr="finalizer rejected",
        )

    monkeypatch.setattr(launcher, "_sbatch", submit)
    monkeypatch.setattr(launcher.os, "environ", dict(launcher.os.environ))
    with pytest.raises(SystemExit) as excinfo:
        launcher.main(_main_arguments(fixture, submit=True))

    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert captured.out.strip() == ""
    payload = json.loads(captured.err)
    assert payload["status"] == "PARTIAL_SUBMISSION"
    assert payload["worker_job_id"] == "11111"
    assert payload["finalizer_job_id"] is None
    assert payload["disposition"] == (
        "WORKER_ARRAY_SUBMITTED_FINALIZER_NOT_SUBMITTED"
    )
    assert payload["recovery"] == "RESEARCHER_DIRECTED_ONLY"
    assert payload["automatic_retry"] is False
    assert payload["automatic_cancel"] is False
    assert payload["error_type"] == "CalledProcessError"
    assert len(calls) == 2
    assert (fixture.run_root / "workers").is_dir()
    assert not (fixture.run_root / "pool").exists()


def test_finalize_cli_requires_authorization(fixture, monkeypatch):
    fake_plan = {
        "tasks": [],
        "total_batches": 1250,
        "run_root": str(fixture.run_root),
        "destination": str(fixture.run_root / "pool"),
    }
    monkeypatch.setattr(coordinator, "build_plan", lambda **kwargs: fake_plan)
    with pytest.raises(SystemExit, match="requires --authorization"):
        coordinator.main([
            "--finalize",
            "--repository-root", str(fixture.repository),
            "--expected-head", fixture.head,
            "--run-root", str(fixture.run_root),
            "--checkpoint-parent", str(fixture.checkpoint_parent),
            "--dataset-config", str(fixture.dataset_config),
            "--config-sha256", fixture.plan["config_sha256"],
        ])


def test_slurm_scripts_declare_production_resources():
    array_text = ARRAY_SCRIPT.read_text(encoding="utf-8")
    finalizer_text = FINALIZER_SCRIPT.read_text(encoding="utf-8")
    range_text = RANGE_SCRIPT.read_text(encoding="utf-8")

    assert "#SBATCH --array=0-49%2" in array_text
    assert "#SBATCH --time=01:30:00" in array_text
    assert "#SBATCH --no-requeue" in array_text
    assert "#SBATCH --cpus-per-task=8" in array_text
    assert "#SBATCH --mem=32G" in array_text
    assert "sbatch" not in array_text
    assert "srun" not in array_text

    assert "#SBATCH --time=00:30:00" in finalizer_text
    assert "#SBATCH --no-requeue" in finalizer_text
    assert "#SBATCH --cpus-per-task=8" in finalizer_text
    assert "#SBATCH --mem=32G" in finalizer_text
    assert "sbatch" not in finalizer_text
    assert "srun" not in finalizer_text

    # The accepted diagnostic range launcher remains unchanged.
    assert "#SBATCH --time=00:30:00" in range_text
