"""Focused Stage 3 VAE production entry-point tests.

No real datasets, training, Slurm jobs, checkpoints or synthesis are used.
Tests that require TensorFlow skip when it is unavailable in the local
environment; the remaining contract checks use fixtures and mocks only.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = (
    REPO_ROOT / "scripts" / "train_paper01_vae_revision_production.py"
)
GOVERNED_PATH = (
    REPO_ROOT
    / "src"
    / "trustforge"
    / "models"
    / "paper01_revision_2026"
    / "variational_autoencoder_governed_training.py"
)
MODEL_PATH = GOVERNED_PATH.with_name(
    "variational_autoencoder_model.py"
)
OBSERVATION_PATH = GOVERNED_PATH.with_name(
    "variational_autoencoder_observation.py"
)

COMMIT = "a" * 40


def load_cli():
    spec = importlib.util.spec_from_file_location(
        "paper01_vae_production_cli",
        CLI_PATH,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load VAE production CLI")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def cli():
    return load_cli()


def make_args(**overrides):
    values = {
        "dataset_id": "cicmaldroid2020",
        "experimental_seed": 42,
        "attempt": 1,
        "expected_source_commit": COMMIT,
        "repository_root": REPO_ROOT,
        "ustc_source_dir": None,
        "cic_source_csv": Path("/fixture/source.csv"),
        "cic_feature_columns": Path("/fixture/features.json"),
        "malnet_image_root": None,
        "malnet_preparation_dir": None,
        "malnet_mapping_path": None,
        "malex_label_archive": None,
        "malex_benign_root": None,
        "malex_malware_root": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def fake_environment():
    return SimpleNamespace(
        actual_source_commit=COMMIT,
        repository_root=REPO_ROOT,
        module_paths={"governed": "fixture"},
    )


# 1. Missing required identity rejection.
def test_parser_requires_explicit_identity(cli):
    with pytest.raises(SystemExit):
        cli.make_parser().parse_args([])

    with pytest.raises(SystemExit):
        cli.make_parser().parse_args(
            [
                "--dataset-id",
                "cicmaldroid2020",
                "--attempt",
                "1",
                "--expected-source-commit",
                COMMIT,
                "--repository-root",
                str(REPO_ROOT),
            ]
        )


def test_missing_dataset_source_argument_rejected(cli):
    with pytest.raises(
        ValueError, match="missing required dataset source"
    ):
        cli.validate_request(make_args(cic_source_csv=None))


def test_invalid_source_commit_rejected(cli):
    with pytest.raises(ValueError, match="full lowercase SHA-1"):
        cli.validate_request(
            make_args(expected_source_commit="not-a-commit")
        )


# 2. Invalid dataset or seed rejection.
def test_invalid_dataset_rejected(cli):
    with pytest.raises(
        ValueError, match="unsupported governed dataset"
    ):
        cli.validate_request(make_args(dataset_id="unknown"))


def test_invalid_seed_rejected(cli):
    with pytest.raises(ValueError, match="42-48"):
        cli.validate_request(make_args(experimental_seed=41))


# 3. Source-commit mismatch rejection.
def test_source_commit_mismatch_rejected(cli, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "fixture-only")
    monkeypatch.setattr(
        cli,
        "capture_git_state",
        lambda root: SimpleNamespace(
            repo_root=REPO_ROOT,
            commit="b" * 40,
            branch="fixture",
            dirty=False,
        ),
    )
    monkeypatch.setattr(
        cli,
        "require_revision_module_binding",
        lambda **kwargs: {"governed": "fixture"},
    )

    with pytest.raises(
        RuntimeError, match="does not match the expected commit"
    ):
        cli.require_environment(make_args())


# 4. Dirty tracked-tree/index rejection.
def test_dirty_worktree_rejected(cli, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "fixture-only")
    monkeypatch.setattr(
        cli,
        "capture_git_state",
        lambda root: SimpleNamespace(
            repo_root=REPO_ROOT,
            commit=COMMIT,
            branch="fixture",
            dirty=True,
        ),
    )

    with pytest.raises(RuntimeError, match="clean"):
        cli.require_environment(make_args())


def test_slurm_allocation_required(cli, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)

    with pytest.raises(RuntimeError, match="Slurm"):
        cli.require_environment(make_args())


# 5. Incorrect module/import-path rejection.
def test_foreign_module_path_rejected(cli):
    foreign = SimpleNamespace(__file__="/tmp/vae/train.py")

    with pytest.raises(
        RuntimeError, match="Revision-2026 module path"
    ):
        cli.require_revision_module_binding(
            repository_root=REPO_ROOT,
            modules={"foreign": foreign},
        )


def test_actual_revision_modules_bind(cli):
    pytest.importorskip("tensorflow")

    paths = cli.require_revision_module_binding(
        repository_root=REPO_ROOT,
    )

    assert paths
    assert all(
        "src/trustforge/models/paper01_revision_2026" in path
        for path in paths.values()
    )


# 6. Disabled authorization rejection.
def test_gate_remains_disabled_in_source():
    source = GOVERNED_PATH.read_text(encoding="utf-8")

    assert "VAE_REAL_DATASET_TRAINING_ENABLED = False" in source
    assert "def require_vae_production_authorization()" in source
    assert "require_vae_production_authorization()" in source


def test_disabled_gate_helper_raises(cli):
    pytest.importorskip("tensorflow")

    governed = cli._load_governed_module()

    assert governed.VAE_REAL_DATASET_TRAINING_ENABLED is False

    with pytest.raises(PermissionError, match="not authorized"):
        governed.require_vae_production_authorization()


# 7. Authorization rejection before adapter creation.
def test_gate_rejects_before_adapter_creation(cli, monkeypatch):
    calls = []

    class Governed:
        @staticmethod
        def require_vae_production_authorization():
            calls.append("gate")
            raise PermissionError(
                "VAE real-dataset training is not authorized"
            )

        @staticmethod
        def run_governed_vae_training(**kwargs):
            raise AssertionError("training must not run")

    monkeypatch.setattr(
        cli, "_load_governed_module", lambda: Governed
    )
    monkeypatch.setattr(
        cli, "require_environment", lambda args: fake_environment()
    )
    monkeypatch.setattr(
        cli,
        "build_adapter",
        lambda args: calls.append("adapter"),
    )

    with pytest.raises(PermissionError, match="not authorized"):
        cli.run(make_args())

    assert calls == ["gate"]


# 8. Correct Revision-2026 adapter dispatch using mocks.
def test_cic_adapter_dispatch(cli, monkeypatch, tmp_path):
    source = tmp_path / "source.csv"
    columns = tmp_path / "features.json"
    source.write_text("fixture\n", encoding="utf-8")
    columns.write_text("[]\n", encoding="utf-8")

    observed = {}

    class FakeCIC:
        def __init__(
            self,
            source_csv,
            feature_columns_path,
            *,
            validate_hashes,
        ):
            observed["source"] = source_csv
            observed["columns"] = feature_columns_path
            observed["validate_hashes"] = validate_hashes

    monkeypatch.setattr(
        cli,
        "_load_adapter_types",
        lambda: {"cicmaldroid2020": FakeCIC},
    )

    adapter = cli.build_adapter(
        make_args(
            cic_source_csv=source,
            cic_feature_columns=columns,
        )
    )

    assert isinstance(adapter, FakeCIC)
    assert observed["source"] == source.resolve()
    assert observed["columns"] == columns.resolve()
    assert observed["validate_hashes"] is True


def test_unsupported_adapter_dispatch_rejected(cli, monkeypatch):
    monkeypatch.setattr(cli, "_load_adapter_types", lambda: {})

    with pytest.raises(
        ValueError, match="unsupported governed dataset"
    ):
        cli.build_adapter(make_args(dataset_id="unknown"))


def test_symlinked_source_rejected(cli, tmp_path):
    real = tmp_path / "source.csv"
    real.write_text("fixture\n", encoding="utf-8")
    link = tmp_path / "link.csv"
    link.symlink_to(real)

    with pytest.raises(ValueError, match="missing or symlinked"):
        cli.require_sources(
            make_args(cic_source_csv=link),
            ("cic_source_csv",),
        )


# 9. Approved effective-configuration identity.
def test_effective_configuration_identity(cli):
    constants = {
        "epochs": 2,
        "batch_size": 64,
        "learning_rate": 0.001,
        "image_shape": (40, 40, 3),
        "latent_dim": 100,
        "observation_variance": 1.0,
        "kl_weight": 1.0,
    }

    config = cli.effective_configuration(constants=constants)

    assert config["epochs"] == 2
    assert config["batch_size"] == 64
    assert config["learning_rate"] == 0.001
    assert config["canonical_image_shape"] == [40, 40, 3]
    assert config["latent_dimension"] == 100
    assert config["observation_variance"] == 1.0
    assert config["kl_coefficient"] == 1.0
    assert config["kl_annealing"] is False
    assert config["optimizer"] == "adam"
    assert config["checkpoint_roles"] == ["best", "last"]
    assert config["test_split_usage"] == "prohibited"


def test_actual_scientific_constants_unchanged():
    governed = GOVERNED_PATH.read_text(encoding="utf-8")
    model = MODEL_PATH.read_text(encoding="utf-8")
    observation = OBSERVATION_PATH.read_text(encoding="utf-8")

    assert "APPROVED_EPOCHS = 2" in governed
    assert "APPROVED_BATCH_SIZE = 64" in governed
    assert "APPROVED_LEARNING_RATE = 0.001" in governed
    assert "IMAGE_SHAPE = (40, 40, 3)" in model
    assert "LATENT_DIM = 100" in model
    assert "OBSERVATION_VARIANCE = 1.0" in observation
    assert "KL_WEIGHT = 1.0" in observation


# 10. No test-partition access.
def test_test_partition_is_rejected():
    from trustforge.models.paper01_revision_2026.variational_autoencoder_data_integration import (
        canonical_vae_batch,
    )

    class Adapter:
        def split_size(self, split):
            return 10

        def canonical_batch(self, split, positions):
            raise AssertionError(
                "adapter must not be called for a test split"
            )

    with pytest.raises(ValueError, match="prohibits this split"):
        canonical_vae_batch(
            adapter=Adapter(),
            split="test",
            logical_positions=np.array([0], dtype=np.int64),
            num_classes=5,
        )


# 11. No silent retry, resume or legacy fallback.
def test_no_legacy_or_implicit_automation_paths():
    source = CLI_PATH.read_text(encoding="utf-8")
    lowered = source.lower()

    for token in (
        "vae.train",
        "vae.pipeline",
        "app.main",
        "slurm/oneoffs/run_vae.slurm",
        "sbatch",
        "srun",
    ):
        assert token not in source

    assert "retry" not in lowered
    assert "resume" not in lowered
    assert "while true" not in lowered


# 12. No checkpoint preparation or publication.
def test_run_exposes_not_prepared_and_not_published(
    cli, monkeypatch, capsys
):
    class Governed:
        @staticmethod
        def require_vae_production_authorization():
            return None

        @staticmethod
        def run_governed_vae_training(**kwargs):
            return {
                "checkpoint_status": "NOT_PREPARED",
                "publication_status": "NOT_PUBLISHED",
                "dataset_provenance": {"fixture_only": True},
                "training_result": {
                    "epochs_completed": 2,
                    "completed_updates": 6,
                },
            }

    monkeypatch.setattr(
        cli, "_load_governed_module", lambda: Governed
    )
    monkeypatch.setattr(
        cli, "require_environment", lambda args: fake_environment()
    )
    monkeypatch.setattr(cli, "build_adapter", lambda args: object())
    monkeypatch.setattr(
        cli, "effective_configuration", lambda: {"epochs": 2}
    )

    summary = cli.run(make_args())

    assert summary["checkpoint_status"] == "NOT_PREPARED"
    assert summary["publication_status"] == "NOT_PUBLISHED"
    assert summary["expected_source_commit"] == COMMIT
    assert summary["actual_source_commit"] == COMMIT
    assert summary["source_status"] == "CLEAN"
    assert "VAE_REVISION_PRODUCTION_RESULT" in capsys.readouterr().out


def test_no_checkpoint_or_synthesis_automation_in_source():
    source = CLI_PATH.read_text(encoding="utf-8")

    for token in (
        "training_checkpoint_bridge",
        "prepare_vae_training_checkpoints",
        "publish_candidate_vae_commit",
        "synthesize_vae_batch",
    ):
        assert token not in source
