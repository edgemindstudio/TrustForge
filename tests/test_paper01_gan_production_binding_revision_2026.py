"""Focused tests for the GAN production launcher and checkpoint binding."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from trustforge.models.paper01_revision_2026 import (
    gan_governed_runner,
)
from trustforge.models.paper01_revision_2026 import (
    gan_production_binding as binding,
)
from trustforge.models.paper01_revision_2026.experimental_seed_policy import (
    require_governed_experimental_seed,
)
from trustforge.models.paper01_revision_2026.gan_class_sampler import (
    GAN_CLASS_SAMPLING_ALPHA,
)
from trustforge.models.paper01_revision_2026.gan_model_factory import (
    GAN_LATENT_DIM,
)
from trustforge.models.paper01_revision_2026.gan_run_governance import (
    BATCH_SIZE,
    MAX_ITERATIONS,
    MIN_ITERATIONS,
    NOMINAL_PASSES,
    plan_gan_run,
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

COMMIT = "abc1234" + "0" * 33
IDENTITY = "commitabc1234_attempt01"

SPEC = importlib.util.spec_from_file_location(
    "paper01_gan_production_entry",
    WRAPPER_PATH,
)
WRAPPER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WRAPPER)


@pytest.fixture
def artifact_root(tmp_path, monkeypatch):
    root = tmp_path / "paper01"
    monkeypatch.setattr(binding, "ARTIFACT_ROOT", root)
    return root


def resolve(**overrides):
    values = {
        "dataset_id": "cicmaldroid2020",
        "experimental_seed": 42,
        "run_identity": IDENTITY,
        "expected_source_commit": COMMIT,
    }
    values.update(overrides)
    return binding.resolve_gan_production_checkpoint_directory(**values)


def test_run_identity_is_deterministic():
    assert binding.build_gan_run_identity(
        expected_source_commit=COMMIT, attempt=1,
    ) == IDENTITY
    assert binding.build_gan_run_identity(
        expected_source_commit=COMMIT, attempt=12,
    ) == "commitabc1234_attempt12"
    assert binding.build_gan_run_identity(
        expected_source_commit=COMMIT, attempt=1,
    ) == binding.build_gan_run_identity(
        expected_source_commit=COMMIT, attempt=1,
    )


def test_resolution_examples(artifact_root):
    cic = resolve()
    malex = resolve(
        dataset_id="malex",
        experimental_seed=43,
        run_identity="commitabc1234_attempt02",
    )

    assert cic == (
        artifact_root
        / "gan"
        / "cicmaldroid2020"
        / "seed42"
        / "checkpoints"
        / "production"
        / "commitabc1234_attempt01"
    )
    assert malex == (
        artifact_root
        / "gan"
        / "malex"
        / "seed43"
        / "checkpoints"
        / "production"
        / "commitabc1234_attempt02"
    )


def test_resolution_does_not_create_directories(artifact_root):
    directory = resolve()
    assert not directory.exists()
    assert not artifact_root.exists()


def test_creation_is_fresh_and_non_overwriting(artifact_root):
    directory = binding.create_gan_production_checkpoint_directory(
        dataset_id="cicmaldroid2020",
        experimental_seed=42,
        run_identity=IDENTITY,
        expected_source_commit=COMMIT,
    )
    assert directory.is_dir()

    with pytest.raises(FileExistsError):
        binding.create_gan_production_checkpoint_directory(
            dataset_id="cicmaldroid2020",
            experimental_seed=42,
            run_identity=IDENTITY,
            expected_source_commit=COMMIT,
        )


def test_creation_rejects_symlinked_component(artifact_root):
    artifact_root.mkdir(parents=True)
    target = artifact_root / "elsewhere"
    target.mkdir()
    (artifact_root / "gan").symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        binding.create_gan_production_checkpoint_directory(
            dataset_id="cicmaldroid2020",
            experimental_seed=42,
            run_identity=IDENTITY,
            expected_source_commit=COMMIT,
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"dataset_id": "unknown"},
        {"experimental_seed": 41},
        {"experimental_seed": 49},
        {"experimental_seed": True},
        {"run_identity": "../escape"},
        {"run_identity": "commitabc1234_attempt1"},
        {"run_identity": "commitdeadbee_attempt01"},
        {"run_identity": "COMMITABC1234_attempt01"},
        {"run_identity": "commitabc1234"},
        {"expected_source_commit": "abc"},
    ],
)
def test_invalid_identities_fail_closed(artifact_root, overrides):
    with pytest.raises((TypeError, ValueError)):
        resolve(**overrides)


@pytest.mark.parametrize("attempt", [0, -1, True, "1", 1.0])
def test_invalid_attempts_rejected(attempt):
    with pytest.raises((TypeError, ValueError)):
        binding.build_gan_run_identity(
            expected_source_commit=COMMIT,
            attempt=attempt,
        )


def test_unauthorized_run_creates_nothing(artifact_root, monkeypatch):
    calls = []

    monkeypatch.setattr(
        WRAPPER,
        "require_environment",
        lambda args: SimpleNamespace(commit=COMMIT, dirty=False),
    )
    monkeypatch.setattr(
        WRAPPER,
        "build_adapter",
        lambda args: calls.append("adapter"),
    )
    monkeypatch.setattr(
        gan_governed_runner,
        "GAN_PRODUCTION_TRAINING_ENABLED",
        False,
    )

    args = SimpleNamespace(
        dataset_id="cicmaldroid2020",
        experimental_seed=42,
        attempt=1,
        expected_source_commit=COMMIT,
        repository_root=Path("."),
    )

    with pytest.raises(PermissionError, match="not authorized"):
        WRAPPER.run(args)

    assert calls == []
    assert not artifact_root.exists()


def test_authorized_run_creates_destination_before_prepare(
    artifact_root, monkeypatch,
):
    observed = {}

    monkeypatch.setattr(
        WRAPPER,
        "require_environment",
        lambda args: SimpleNamespace(commit=COMMIT, dirty=False),
    )
    monkeypatch.setattr(
        gan_governed_runner,
        "GAN_PRODUCTION_TRAINING_ENABLED",
        True,
    )
    monkeypatch.setattr(
        WRAPPER, "build_adapter", lambda args: object(),
    )
    monkeypatch.setattr(
        WRAPPER,
        "resolve_gan_class_mapping",
        lambda **kwargs: ("0",),
    )

    def stop_at_prepare(**kwargs):
        observed["directory"] = kwargs["checkpoint_directory"]
        raise RuntimeError("stop before model construction")

    monkeypatch.setattr(WRAPPER, "prepare_gan_run", stop_at_prepare)

    args = SimpleNamespace(
        dataset_id="cicmaldroid2020",
        experimental_seed=42,
        attempt=1,
        expected_source_commit=COMMIT,
        repository_root=Path("."),
    )

    with pytest.raises(RuntimeError, match="stop before"):
        WRAPPER.run(args)

    assert observed["directory"].is_dir()
    assert observed["directory"].name == IDENTITY


def test_unsupported_seed_creates_nothing(artifact_root, monkeypatch):
    calls = []

    monkeypatch.setattr(
        WRAPPER,
        "require_environment",
        lambda args: SimpleNamespace(commit=COMMIT, dirty=False),
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
        experimental_seed=41,
        attempt=1,
        expected_source_commit=COMMIT,
        repository_root=Path("."),
    )

    with pytest.raises(ValueError, match="42-48"):
        WRAPPER.run(args)

    assert calls == []
    assert not artifact_root.exists()


def test_execution_flags_remain_disabled():
    assert gan_governed_runner.GAN_FEASIBILITY_TRAINING_ENABLED is False
    assert gan_governed_runner.GAN_PRODUCTION_TRAINING_ENABLED is False

    with pytest.raises(PermissionError, match="not authorized"):
        gan_governed_runner.require_gan_production_authorization()


def test_compute_node_guard_precedes_execution(monkeypatch):
    monkeypatch.setattr(
        gan_governed_runner,
        "GAN_PRODUCTION_TRAINING_ENABLED",
        True,
    )
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)

    with pytest.raises(RuntimeError, match="Slurm"):
        gan_governed_runner._execute_gan_authorized_run(
            object(),
            dataset=object(),
            mode="production",
        )


def test_launcher_targets_governed_entry_point():
    text = LAUNCHER_PATH.read_text(encoding="utf-8")

    assert "scripts/train_paper01_gan_revision_production.py" in text
    assert "GAN_EXPECTED_SOURCE_COMMIT" in text
    assert "GAN_ATTEMPT" in text
    assert "GAN_CHECKPOINT_DIR" not in text
    assert "app.main" not in text
    assert "gan.train" not in text


def test_wrapper_uses_existing_runner_only():
    text = WRAPPER_PATH.read_text(encoding="utf-8")

    assert "prepare_gan_run" in text
    assert "execute_gan_production_run" in text
    assert "require_gan_production_authorization" in text
    assert "create_gan_production_checkpoint_directory" in text
    assert "--checkpoint-dir" not in text
    assert "app.main" not in text
    assert "gan.train" not in text


def test_scientific_controls_unchanged():
    assert BATCH_SIZE == 256
    assert GAN_LATENT_DIM == 100
    assert GAN_CLASS_SAMPLING_ALPHA == 0.85

    train_size = 8117
    plan = plan_gan_run(
        dataset_id="cicmaldroid2020",
        experimental_seed=42,
        class_mapping=tuple(str(index) for index in range(5)),
        train_size=train_size,
    )
    expected = min(
        MAX_ITERATIONS,
        max(
            MIN_ITERATIONS,
            NOMINAL_PASSES * math.ceil(train_size / BATCH_SIZE),
        ),
    )
    assert plan.planned_generator_iterations == expected
    assert plan.planned_discriminator_updates == 2 * expected

    for seed in range(42, 49):
        assert require_governed_experimental_seed(seed) == seed

    for seed in (41, 49):
        with pytest.raises(ValueError):
            require_governed_experimental_seed(seed)
