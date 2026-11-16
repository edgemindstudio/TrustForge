"""Focused tests for scoped GAN production authorization.

Static and mocked validation only. No TensorFlow training, checkpoint
publication, Slurm submission or production directory creation occurs.
"""

from __future__ import annotations

import importlib.util
import json
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from trustforge.models.paper01_revision_2026 import (
    gan_governed_runner,
    gan_production_authorization as authorization,
    gan_production_binding as binding,
)
from trustforge.models.paper01_revision_2026.gan_class_sampler import (
    GAN_CLASS_SAMPLING_ALPHA,
)
from trustforge.models.paper01_revision_2026.gan_governed_runner import (
    GANRunPreparation,
)
from trustforge.models.paper01_revision_2026.gan_model_factory import (
    GAN_LATENT_DIM,
)
from trustforge.models.paper01_revision_2026.gan_run_governance import (
    BATCH_SIZE,
    GANRunPlan,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WRAPPER_PATH = (
    REPOSITORY_ROOT
    / "scripts"
    / "train_paper01_gan_revision_production.py"
)
LAUNCHER_PATH = (
    REPOSITORY_ROOT
    / "slurm"
    / "paper01_revision_2026"
    / "gan_revision_production.slurm"
)

SPEC = importlib.util.spec_from_file_location(
    "paper01_gan_scope_entry",
    WRAPPER_PATH,
)
WRAPPER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WRAPPER)

EXECUTABLE = "abc1234" + "0" * 33
OTHER_EXECUTABLE = "b" * 40
RUN_IDENTITY = "commitabc1234_attempt01"


def _identity(executable, attempt):
    return f"commit{executable[:7]}_attempt{attempt:02d}"


def _grant(
    *,
    dataset_id="cicmaldroid2020",
    experimental_seed=42,
    attempt=1,
    executable=EXECUTABLE,
    scientific_baseline=authorization.SCIENTIFIC_BASELINE_COMMIT,
    mode="production",
    resources=None,
    checkpoint_directory=None,
):
    identity = _identity(executable, attempt)

    if checkpoint_directory is None:
        checkpoint_directory = str(
            binding.resolve_gan_production_checkpoint_directory(
                dataset_id=dataset_id,
                experimental_seed=experimental_seed,
                run_identity=identity,
                expected_source_commit=executable,
            )
        )

    return {
        "schema": authorization.SCHEMA,
        "paper": "paper01",
        "family": "gan",
        "mode": mode,
        "dataset_id": dataset_id,
        "experimental_seed": experimental_seed,
        "attempt": attempt,
        "run_identity": identity,
        "scientific_baseline_commit": scientific_baseline,
        "executable_source_commit": executable,
        "checkpoint_directory": checkpoint_directory,
        "authorization_id": "p1-gan-cic-s42-a01",
        "resources": dict(
            authorization.EXPECTED_RESOURCES
            if resources is None
            else resources
        ),
    }


def _allocation(**overrides):
    values = {
        "JobId": "12345",
        "JobState": "RUNNING",
        "Partition": "talon",
        "NumNodes": "1",
        "NumTasks": "1",
        "NumCPUs": "8",
        "TRES": "cpu=8,mem=32G,node=1,billing=8",
        "TimeLimit": "12:00:00",
        "Requeue": "0",
        "Restarts": "0",
        "Gres": "(null)",
    }
    values.update(overrides)
    return values


def _write(path, record):
    path.write_text(json.dumps(record), encoding="utf-8")
    path.chmod(0o600)


@pytest.fixture
def context(tmp_path, monkeypatch):
    root = tmp_path / "paper01"
    path = tmp_path / "production_authorization.json"

    monkeypatch.setattr(binding, "ARTIFACT_ROOT", root)

    checkpoint = binding.resolve_gan_production_checkpoint_directory(
        dataset_id="cicmaldroid2020",
        experimental_seed=42,
        run_identity=RUN_IDENTITY,
        expected_source_commit=EXECUTABLE,
    )

    monkeypatch.setattr(authorization, "AUTHORIZATION_PATH", path)
    monkeypatch.setattr(
        authorization, "ACCOUNT_HOME", Path.home().resolve()
    )
    monkeypatch.setattr(
        authorization,
        "capture_git_state",
        lambda repository_root: SimpleNamespace(
            commit=EXECUTABLE,
            dirty=False,
        ),
    )
    monkeypatch.setattr(
        authorization,
        "capture_gan_slurm_allocation",
        lambda: _allocation(),
    )

    return SimpleNamespace(
        root=root,
        path=path,
        checkpoint=checkpoint,
    )


def _require(context, **overrides):
    values = {
        "dataset_id": "cicmaldroid2020",
        "experimental_seed": 42,
        "checkpoint_directory": context.checkpoint,
        "repository_root": Path("."),
    }
    values.update(overrides)
    return authorization.require_gan_production_scope(**values)


def _preparation(context):
    plan = GANRunPlan(
        dataset_id="cicmaldroid2020",
        experimental_seed=42,
        class_mapping=tuple(str(index) for index in range(5)),
        train_size=8117,
        batch_size=BATCH_SIZE,
        steps_per_pass=32,
        planned_generator_iterations=64,
        planned_discriminator_updates=128,
    )
    return GANRunPreparation(
        plan=plan,
        checkpoint_directory=context.checkpoint,
    )


def test_exact_proposed_identity_is_eligible(context):
    _write(context.path, _grant())

    scope = _require(context)

    assert scope.authorization_id == "p1-gan-cic-s42-a01"
    assert scope.dataset_id == "cicmaldroid2020"
    assert scope.experimental_seed == 42
    assert scope.attempt == 1
    assert scope.run_identity == RUN_IDENTITY
    assert scope.executable_source_commit == EXECUTABLE
    assert scope.scientific_baseline_commit == (
        authorization.SCIENTIFIC_BASELINE_COMMIT
    )
    assert scope.checkpoint_directory == context.checkpoint
    assert not context.checkpoint.exists()


def test_missing_authorization_denies(context):
    with pytest.raises(
        authorization.GANProductionAuthorizationError,
        match="absent",
    ):
        _require(context)

    assert not context.checkpoint.exists()


def test_run_identity_mismatch_denies(context):
    _write(context.path, _grant())

    with pytest.raises(
        authorization.GANProductionAuthorizationError,
        match="dataset",
    ):
        _require(context, dataset_id="malex")

    with pytest.raises(
        authorization.GANProductionAuthorizationError,
        match="seed",
    ):
        _require(context, experimental_seed=43)


@pytest.mark.parametrize(
    "overrides",
    [
        {"paper": "paper02"},
        {"family": "vae"},
        {"mode": "feasibility"},
        {"schema": "unsupported"},
        {"dataset_id": "malex"},
        {"experimental_seed": 43},
        {"attempt": 2},
        {"scientific_baseline_commit": "0" * 40},
        {"executable_source_commit": "c" * 40},
        {"authorization_id": "bad id"},
        {
            "checkpoint_directory": (
                "/tmp/paper01/gan/cicmaldroid2020/seed42/"
                "checkpoints/production/commitabc1234_attempt01"
            )
        },
    ],
)
def test_invalid_grant_fields_deny(context, overrides):
    record = _grant()
    record.update(overrides)
    _write(context.path, record)

    with pytest.raises(authorization.GANProductionAuthorizationError):
        _require(context)


def test_missing_and_extra_grant_fields_deny(context):
    record = _grant()
    del record["resources"]
    _write(context.path, record)

    with pytest.raises(
        authorization.GANProductionAuthorizationError,
        match="fields",
    ):
        _require(context)

    record = _grant()
    record["unexpected"] = True
    _write(context.path, record)

    with pytest.raises(
        authorization.GANProductionAuthorizationError,
        match="fields",
    ):
        _require(context)


def test_symlinked_authorization_denies(context):
    target = context.path.with_name("real_authorization.json")
    _write(target, _grant())
    context.path.symlink_to(target)

    with pytest.raises(
        authorization.GANProductionAuthorizationError,
        match="absent",
    ):
        _require(context)


def test_group_writable_authorization_denies(context):
    _write(context.path, _grant())
    context.path.chmod(0o666)

    with pytest.raises(
        authorization.GANProductionAuthorizationError,
        match="writable",
    ):
        _require(context)


def test_dirty_executable_source_denies(context, monkeypatch):
    _write(context.path, _grant())
    monkeypatch.setattr(
        authorization,
        "capture_git_state",
        lambda repository_root: SimpleNamespace(
            commit=EXECUTABLE,
            dirty=True,
        ),
    )

    with pytest.raises(
        authorization.GANProductionAuthorizationError,
        match="dirty",
    ):
        _require(context)


def test_executable_source_mismatch_denies(context, monkeypatch):
    other_checkpoint = (
        binding.resolve_gan_production_checkpoint_directory(
            dataset_id="cicmaldroid2020",
            experimental_seed=42,
            run_identity=_identity(OTHER_EXECUTABLE, 1),
            expected_source_commit=OTHER_EXECUTABLE,
        )
    )
    _write(context.path, _grant(executable=OTHER_EXECUTABLE))
    monkeypatch.setattr(
        authorization,
        "capture_git_state",
        lambda repository_root: SimpleNamespace(
            commit=EXECUTABLE,
            dirty=False,
        ),
    )

    with pytest.raises(
        authorization.GANProductionAuthorizationError,
        match="differs",
    ):
        _require(context, checkpoint_directory=other_checkpoint)


@pytest.mark.parametrize(
    "overrides",
    [
        {"JobState": "COMPLETED"},
        {"Partition": "talon-large"},
        {"NumNodes": "2"},
        {"NumTasks": "2"},
        {"NumCPUs": "4"},
        {"TRES": "cpu=4,mem=32G,node=1,billing=4"},
        {"TRES": "cpu=8,mem=16G,node=1,billing=8"},
        {"TimeLimit": "06:00:00"},
        {"Requeue": "1"},
        {"Restarts": "1"},
        {"Gres": "gpu:1"},
        {
            "TRES": (
                "cpu=8,mem=32G,node=1,billing=8,gres/gpu=1"
            )
        },
    ],
)
def test_unauthorized_allocation_denies(
    context, monkeypatch, overrides,
):
    _write(context.path, _grant())

    monkeypatch.setattr(
        authorization,
        "capture_gan_slurm_allocation",
        lambda: _allocation(**overrides),
    )

    with pytest.raises(authorization.GANProductionAuthorizationError):
        _require(context)


def test_missing_allocation_field_denies(context, monkeypatch):
    _write(context.path, _grant())
    allocation = _allocation()
    del allocation["NumTasks"]

    monkeypatch.setattr(
        authorization,
        "capture_gan_slurm_allocation",
        lambda: allocation,
    )

    with pytest.raises(
        authorization.GANProductionAuthorizationError,
        match="NumTasks",
    ):
        _require(context)


def test_slurm_allocation_parser_round_trip():
    raw = (
        "JobId=12345 JobName=p1r1-gan-prod Partition=talon "
        "NumNodes=1 NumTasks=1 NumCPUs=8 "
        "TRES=cpu=8,mem=32G,node=1,billing=8 "
        "TimeLimit=12:00:00 Requeue=0 Restarts=0 Gres=(null)"
    )

    parsed = authorization.parse_gan_slurm_allocation(raw)

    assert parsed["Partition"] == "talon"
    assert parsed["TRES"] == "cpu=8,mem=32G,node=1,billing=8"


def test_slurm_allocation_parser_rejects_multiline():
    with pytest.raises(authorization.GANProductionAuthorizationError):
        authorization.parse_gan_slurm_allocation(
            "JobId=1 Partition=talon\nJobId=2 Partition=talon"
        )


def test_runner_denies_without_authorization(context, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setattr(
        gan_governed_runner,
        "GAN_PRODUCTION_TRAINING_ENABLED",
        True,
    )

    def unexpected_provenance(**kwargs):
        raise AssertionError("provenance was reached")

    monkeypatch.setattr(
        gan_governed_runner,
        "require_gan_dataset_provenance",
        unexpected_provenance,
    )

    with pytest.raises(
        authorization.GANProductionAuthorizationError,
        match="absent",
    ):
        gan_governed_runner._execute_gan_authorized_run(
            _preparation(context),
            dataset=object(),
            mode="production",
        )

    assert not context.checkpoint.exists()


def test_feasibility_invocation_cannot_use_production_grant(
    context, monkeypatch,
):
    _write(context.path, _grant())
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setattr(
        gan_governed_runner,
        "GAN_FEASIBILITY_TRAINING_ENABLED",
        False,
    )

    with pytest.raises(PermissionError, match="feasibility"):
        gan_governed_runner.execute_gan_feasibility_run(
            _preparation(context),
            dataset=object(),
        )


def test_wrapper_denies_before_adapter_and_directory(
    context, monkeypatch,
):
    calls = []

    monkeypatch.setattr(
        WRAPPER,
        "require_environment",
        lambda args: SimpleNamespace(
            commit=EXECUTABLE,
            dirty=False,
        ),
    )
    monkeypatch.setattr(
        gan_governed_runner,
        "GAN_PRODUCTION_TRAINING_ENABLED",
        True,
    )
    monkeypatch.setattr(
        WRAPPER,
        "build_adapter",
        lambda args: calls.append("adapter"),
    )

    args = SimpleNamespace(
        dataset_id="cicmaldroid2020",
        experimental_seed=42,
        attempt=1,
        expected_source_commit=EXECUTABLE,
        repository_root=Path("."),
    )

    with pytest.raises(
        authorization.GANProductionAuthorizationError,
        match="absent",
    ):
        WRAPPER.run(args)

    assert calls == []
    assert not context.root.exists()


def test_second_execution_at_same_attempt_is_rejected(context):
    _write(context.path, _grant())

    first = binding.create_gan_production_checkpoint_directory(
        dataset_id="cicmaldroid2020",
        experimental_seed=42,
        run_identity=RUN_IDENTITY,
        expected_source_commit=EXECUTABLE,
    )
    assert first.is_dir()

    with pytest.raises(FileExistsError):
        binding.create_gan_production_checkpoint_directory(
            dataset_id="cicmaldroid2020",
            experimental_seed=42,
            run_identity=RUN_IDENTITY,
            expected_source_commit=EXECUTABLE,
        )


def test_retry_with_new_attempt_is_unauthorized(context):
    _write(context.path, _grant(attempt=1))
    retry_directory = (
        binding.resolve_gan_production_checkpoint_directory(
            dataset_id="cicmaldroid2020",
            experimental_seed=42,
            run_identity="commitabc1234_attempt02",
            expected_source_commit=EXECUTABLE,
        )
    )

    with pytest.raises(authorization.GANProductionAuthorizationError):
        _require(context, checkpoint_directory=retry_directory)


def test_scientific_controls_and_flags_remain_unchanged():
    assert GAN_LATENT_DIM == 100
    assert BATCH_SIZE == 256
    assert GAN_CLASS_SAMPLING_ALPHA == 0.85
    assert gan_governed_runner.GAN_FEASIBILITY_TRAINING_ENABLED is False
    assert gan_governed_runner.GAN_PRODUCTION_TRAINING_ENABLED is False


def test_launcher_declares_bounded_resources_without_override():
    text = LAUNCHER_PATH.read_text(encoding="utf-8")

    assert "#SBATCH --no-requeue" in text
    assert "#SBATCH --time=12:00:00" in text
    assert "GAN_CHECKPOINT_DIR" not in text
    assert "app.main" not in text
    assert "gan.train" not in text
