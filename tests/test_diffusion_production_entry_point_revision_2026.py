"""Focused Diffusion Stage 3 production entry-point tests.

No real datasets, training, Slurm jobs, checkpoints or synthesis are used.
Tests that require TensorFlow use the existing environment and skip when it
is unavailable. All dataset and training interactions are mocked.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = (
    REPO_ROOT
    / "scripts"
    / "train_paper01_diffusion_revision_production.py"
)
SLURM_PATH = (
    REPO_ROOT
    / "slurm"
    / "paper01_revision_2026"
    / "diffusion_revision_production.slurm"
)
GOVERNED_PATH = (
    REPO_ROOT
    / "src"
    / "trustforge"
    / "models"
    / "paper01_revision_2026"
    / "diffusion_governed_training.py"
)
MODEL_BUILDER_PATH = REPO_ROOT / "diffusion" / "models.py"

COMMIT = "a" * 40


def load_cli():
    spec = importlib.util.spec_from_file_location(
        "paper01_diffusion_production_cli",
        CLI_PATH,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load Diffusion production CLI")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def cli():
    return load_cli()


def test_optree_preloaded_before_production_imports():
    source = CLI_PATH.read_text(encoding="utf-8")
    optree_index = source.index("import optree")

    assert optree_index < source.index("from trustforge")
    assert optree_index < source.index("def _load_governed_module")
    assert optree_index < source.index("def _load_model_builder_module")


def test_entry_import_preloads_optree_in_fresh_process():
    script = f"""
import importlib.util
from pathlib import Path
import sys

repo = Path({str(REPO_ROOT)!r})
sys.path.insert(0, str(repo / "src"))
sys.path.insert(0, str(repo))

spec = importlib.util.spec_from_file_location(
    "diffusion_entry_optree_probe",
    {str(CLI_PATH)!r},
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

assert "optree" in sys.modules, "optree was not preloaded"
assert "tensorflow" not in sys.modules, "TensorFlow imported too early"
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def make_args(**overrides):
    values = {
        "dataset_id": "cicmaldroid2020",
        "experimental_seed": 42,
        "attempt": 1,
        "expected_source_commit": COMMIT,
        "repository_root": REPO_ROOT,
        "checkpoint_root": None,
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
    )


def fake_binding():
    return SimpleNamespace(
        module_paths={"governed": "fixture"},
        model_builder_path=str(MODEL_BUILDER_PATH),
    )


def fake_result():
    return {
        "dataset_id": "cicmaldroid2020",
        "experimental_seed": 42,
        "class_count": 5,
        "dataset_provenance": {"fixture_only": True},
        "scientific_configuration": {"fixture_config": True},
        "training_result": {
            "completed_epochs": 2,
            "completed_updates": 2,
            "best_epoch": 1,
            "best_validation_mse": 1.0,
        },
        "checkpoint_status": "NOT_PREPARED",
        "publication_status": "NOT_PUBLISHED",
    }


# 1. Missing explicit identity rejection.
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


# 2. Invalid dataset, seed or attempt rejection.
def test_invalid_dataset_rejected(cli):
    with pytest.raises(
        ValueError, match="unsupported governed dataset"
    ):
        cli.validate_request(make_args(dataset_id="unknown"))


def test_invalid_seed_rejected(cli):
    with pytest.raises(ValueError, match="42-48"):
        cli.validate_request(make_args(experimental_seed=41))


@pytest.mark.parametrize("attempt", [0, -1, 1.5])
def test_invalid_attempt_rejected(cli, attempt):
    with pytest.raises(ValueError, match="positive integer"):
        cli.validate_request(make_args(attempt=attempt))


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

    with pytest.raises(
        RuntimeError, match="does not match the expected commit"
    ):
        cli.require_environment(make_args())


# 4. Dirty tracked/index/untracked tree rejection.
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


def test_clean_environment_accepted(cli, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "fixture-only")
    monkeypatch.setattr(
        cli,
        "capture_git_state",
        lambda root: SimpleNamespace(
            repo_root=REPO_ROOT,
            commit=COMMIT,
            branch="fixture",
            dirty=False,
        ),
    )

    environment = cli.require_environment(make_args())

    assert environment.actual_source_commit == COMMIT
    assert environment.repository_root == REPO_ROOT


# 5. Incorrect/foreign module path rejection.
def test_foreign_revision_module_rejected(cli):
    foreign = SimpleNamespace(__file__="/tmp/foreign.py")
    fake_builder = SimpleNamespace(
        __file__=str(MODEL_BUILDER_PATH),
        build_diffusion_model=lambda **kwargs: None,
    )

    with pytest.raises(
        RuntimeError, match="Revision-2026 module path"
    ):
        cli.require_module_binding(
            repository_root=REPO_ROOT,
            modules={"foreign": foreign},
            model_builder=fake_builder,
        )


# 6. Accepted model-builder import binding.
def test_foreign_model_builder_path_rejected(cli):
    valid_module = SimpleNamespace(__file__=str(GOVERNED_PATH))
    foreign_builder = SimpleNamespace(
        __file__=str(REPO_ROOT / "diffusion" / "train.py"),
        build_diffusion_model=lambda **kwargs: None,
    )

    with pytest.raises(
        RuntimeError, match="not the accepted source"
    ):
        cli.require_module_binding(
            repository_root=REPO_ROOT,
            modules={"governed": valid_module},
            model_builder=foreign_builder,
        )


def test_actual_model_builder_binding(cli):
    pytest.importorskip("tensorflow")

    binding = cli.require_module_binding(
        repository_root=REPO_ROOT,
    )

    assert binding.model_builder_path == str(
        MODEL_BUILDER_PATH.resolve()
    )
    assert binding.module_paths
    assert all(
        "src/trustforge/models/paper01_revision_2026" in path
        for path in binding.module_paths.values()
    )
    builder = cli._load_model_builder_module()
    assert callable(builder.build_diffusion_model)


# 6b. Checkpoint root CLI registration.
def test_checkpoint_root_parsed_as_path(cli, tmp_path):
    root = tmp_path / "checkpoints"
    args = cli.make_parser().parse_args([
        "--dataset-id", "cicmaldroid2020",
        "--experimental-seed", "42",
        "--attempt", "1",
        "--expected-source-commit", COMMIT,
        "--repository-root", str(REPO_ROOT),
        "--checkpoint-root", str(root),
    ])
    assert isinstance(args.checkpoint_root, Path)
    assert args.checkpoint_root == root


def test_checkpoint_root_forwarded_to_publication(
    cli, monkeypatch, tmp_path,
):
    captured = {}

    class Governed:
        DIFFUSION_REAL_DATASET_TRAINING_ENABLED = True

        @staticmethod
        def run_governed_diffusion_training(**kwargs):
            return fake_result()

    def publish(**kwargs):
        captured.update(kwargs)
        return {
            "checkpoint_directory": "/fixture/checkpoints",
            "manifest_sha256": "b" * 64,
            "best_state_sha256": "c" * 64,
            "last_state_sha256": "d" * 64,
            "checkpoint_role_primary": "best",
        }

    monkeypatch.setattr(cli, "_load_governed_module", lambda: Governed)
    monkeypatch.setattr(
        cli, "require_environment", lambda args: fake_environment()
    )
    monkeypatch.setattr(
        cli, "require_module_binding", lambda **kwargs: fake_binding()
    )
    monkeypatch.setattr(cli, "build_adapter", lambda args: object())
    monkeypatch.setattr(cli, "publish_governed_checkpoint", publish)

    cli.run(make_args(checkpoint_root=tmp_path))

    assert captured["checkpoint_root"] == tmp_path


# 7. Disabled authorization gate remains False.
def test_gate_remains_disabled_in_source():
    source = GOVERNED_PATH.read_text(encoding="utf-8")

    assert "DIFFUSION_REAL_DATASET_TRAINING_ENABLED = False" in source


# 8. Rejection occurs before adapter construction and model binding.
def test_gate_rejects_before_adapter_and_model_binding(
    cli, monkeypatch
):
    calls = []

    class Governed:
        DIFFUSION_REAL_DATASET_TRAINING_ENABLED = False

        @staticmethod
        def run_governed_diffusion_training(**kwargs):
            raise AssertionError("training must not run")

    monkeypatch.setattr(
        cli, "_load_governed_module", lambda: Governed
    )
    monkeypatch.setattr(
        cli, "require_environment", lambda args: fake_environment()
    )
    monkeypatch.setattr(
        cli,
        "require_module_binding",
        lambda **kwargs: calls.append("binding")
        or fake_binding(),
    )
    monkeypatch.setattr(
        cli,
        "build_adapter",
        lambda args: calls.append("adapter") or object(),
    )

    with pytest.raises(PermissionError, match="not authorized"):
        cli.run(make_args())

    assert calls == []


# 9. Governed function remains fail-closed directly.
def test_governed_function_remains_fail_closed_directly(cli):
    pytest.importorskip("tensorflow")

    governed = cli._load_governed_module()

    class NeverAccessAdapter:
        def split_size(self, split):
            raise AssertionError(
                "dataset accessed without authorization"
            )

        def canonical_batch(self, split, positions):
            raise AssertionError(
                "dataset accessed without authorization"
            )

    assert governed.DIFFUSION_REAL_DATASET_TRAINING_ENABLED is False

    with pytest.raises(PermissionError, match="not authorized"):
        governed.run_governed_diffusion_training(
            adapter=NeverAccessAdapter(),
            dataset_id="ustc_tfc2016",
            experimental_seed=42,
        )


# 10. Correct Revision-2026 adapter dispatch using mocks.
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


# 11. Governed preflight call order and rejection.
def _preflight_fixture():
    return {
        "history": (
            {
                "epoch": 0,
                "training_examples": 64,
                "validation_examples": 64,
                "validation_mse": 2.0,
                "completed_updates": 1,
            },
            {
                "epoch": 1,
                "training_examples": 64,
                "validation_examples": 64,
                "validation_mse": 1.0,
                "completed_updates": 2,
            },
        ),
        "best_epoch": 1,
        "best_validation_mse": 1.0,
        "completed_epochs": 2,
        "completed_updates": 2,
        "train_size": 64,
        "validation_size": 64,
        "best_weights": (),
        "last_weights": (),
    }


class FixtureAdapter:
    def split_size(self, split):
        return 64

    def canonical_batch(self, split, positions):
        raise AssertionError(
            "training core must be mocked in this test"
        )


def test_governed_preflight_order(cli, monkeypatch):
    pytest.importorskip("tensorflow")

    governed = cli._load_governed_module()
    events = []
    observed = {}

    def identity(*, dataset_id, adapter):
        events.append("identity")

    def preflight(*, dataset_id, adapter):
        events.append("preflight")
        return {"dataset_id": dataset_id, "fixture": True}

    def provenance(*, dataset_id, verified_preflight):
        events.append("provenance")
        return {"dataset_id": dataset_id, "fixture": True}

    def core(**kwargs):
        events.append("training")
        observed["core"] = kwargs
        return _preflight_fixture()

    def builder(**kwargs):
        events.append("model")
        observed["builder"] = kwargs
        return object()

    monkeypatch.setattr(
        governed, "DIFFUSION_REAL_DATASET_TRAINING_ENABLED", True
    )
    monkeypatch.setenv("SLURM_JOB_ID", "fixture-only")
    monkeypatch.setattr(
        governed, "require_gan_adapter_identity", identity
    )
    monkeypatch.setattr(
        governed, "require_gan_dataset_provenance", preflight
    )
    monkeypatch.setattr(
        governed, "build_gan_v3_provenance", provenance
    )
    monkeypatch.setattr(
        governed, "run_diffusion_training_core", core
    )

    import diffusion.models as model_builder

    monkeypatch.setattr(
        model_builder, "build_diffusion_model", builder
    )

    result = governed.run_governed_diffusion_training(
        adapter=FixtureAdapter(),
        dataset_id="cicmaldroid2020",
        experimental_seed=42,
    )

    assert events == [
        "identity",
        "preflight",
        "provenance",
        "model",
        "training",
    ]
    assert observed["builder"] == {
        "img_shape": (40, 40, 3),
        "num_classes": 5,
        "base_filters": 64,
        "depth": 2,
        "time_emb_dim": 128,
        "learning_rate": 2e-4,
        "beta_1": 0.9,
    }
    assert observed["core"]["epochs"] == 2
    assert observed["core"]["batch_size"] == 64
    assert observed["core"]["num_classes"] == 5
    assert observed["core"]["experimental_seed"] == 42
    assert result["checkpoint_status"] == "NOT_PREPARED"
    assert result["publication_status"] == "NOT_PUBLISHED"
    assert result["scientific_configuration"]["epochs"] == 2


def test_governed_preflight_rejection_stops_before_training(
    cli, monkeypatch
):
    pytest.importorskip("tensorflow")

    governed = cli._load_governed_module()
    events = []

    def identity(*, dataset_id, adapter):
        events.append("identity")

    def preflight(*, dataset_id, adapter):
        events.append("preflight_rejected")
        raise ValueError("fixture preflight rejection")

    monkeypatch.setattr(
        governed, "DIFFUSION_REAL_DATASET_TRAINING_ENABLED", True
    )
    monkeypatch.setenv("SLURM_JOB_ID", "fixture-only")
    monkeypatch.setattr(
        governed, "require_gan_adapter_identity", identity
    )
    monkeypatch.setattr(
        governed, "require_gan_dataset_provenance", preflight
    )

    with pytest.raises(ValueError, match="fixture preflight"):
        governed.run_governed_diffusion_training(
            adapter=FixtureAdapter(),
            dataset_id="cicmaldroid2020",
            experimental_seed=42,
        )

    assert events == ["identity", "preflight_rejected"]


# 12. Existing scientific_configuration is reused.
def test_scientific_configuration_reused(cli, monkeypatch):
    result = fake_result()

    class Governed:
        DIFFUSION_REAL_DATASET_TRAINING_ENABLED = True

        @staticmethod
        def run_governed_diffusion_training(**kwargs):
            return result

    monkeypatch.setattr(
        cli, "_load_governed_module", lambda: Governed
    )
    monkeypatch.setattr(
        cli, "require_environment", lambda args: fake_environment()
    )
    monkeypatch.setattr(
        cli,
        "require_module_binding",
        lambda **kwargs: fake_binding(),
    )
    monkeypatch.setattr(cli, "build_adapter", lambda args: object())
    publication = {
        "checkpoint_directory": "/fixture/checkpoints",
        "manifest_sha256": "b" * 64,
        "best_state_sha256": "c" * 64,
        "last_state_sha256": "d" * 64,
        "checkpoint_role_primary": "best",
    }
    monkeypatch.setattr(
        cli, "publish_governed_checkpoint",
        lambda **kwargs: publication,
    )

    summary = cli.run(make_args())

    assert summary["scientific_configuration"] is (
        result["scientific_configuration"]
    )
    assert summary["dataset_provenance"] is (
        result["dataset_provenance"]
    )
    assert summary["checkpoint_status"] == "PREPARED"
    assert summary["publication_status"] == "PUBLISHED"
    assert summary["checkpoint_publication"] == publication
    assert summary["training_result"]["completed_epochs"] == 2
    assert summary["training_result"]["best_epoch"] == 1
    assert summary["supplementary_execution_policy"] == {
        "objective": "epsilon_prediction_mse",
        "validation_metric": "validation_mse",
        "selection_direction": "minimize",
        "selection_tie_break": "earliest_epoch",
        "test_split_usage": "prohibited",
        "early_stopping": False,
        "retry_policy": "disabled_by_launcher",
        "resume_policy": "not_supported",
    }
    assert summary["expected_source_commit"] == COMMIT
    assert summary["actual_source_commit"] == COMMIT
    assert summary["model_builder_path"] == str(
        MODEL_BUILDER_PATH
    )


# 13. Accepted scientific controls remain unchanged.
def test_actual_scientific_constants_unchanged():
    governed = GOVERNED_PATH.read_text(encoding="utf-8")
    model = MODEL_BUILDER_PATH.read_text(encoding="utf-8")

    assert "DIFFUSION_REAL_DATASET_TRAINING_ENABLED = False" in governed
    assert "APPROVED_IMAGE_SHAPE = (40, 40, 3)" in governed
    assert "APPROVED_DEPTH = 2" in governed
    assert "APPROVED_BASE_FILTERS = 64" in governed
    assert "APPROVED_TIME_EMBED_DIM = 128" in governed
    assert "APPROVED_TIMESTEPS = 1000" in governed
    assert "APPROVED_BETA_START = 1e-4" in governed
    assert "APPROVED_BETA_END = 2e-2" in governed
    assert "APPROVED_EPOCHS = 2" in governed
    assert "APPROVED_BATCH_SIZE = 64" in governed
    assert "APPROVED_LEARNING_RATE = 2e-4" in governed
    assert "APPROVED_BETA_1 = 0.9" in governed
    assert "def build_diffusion_model(" in model


# 14. Test-split access is rejected.
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


# 15. No legacy fallback, automatic retry or resume.
def test_no_legacy_or_implicit_automation_paths():
    source = CLI_PATH.read_text(encoding="utf-8")

    for token in (
        "diffusion.train",
        "diffusion.pipeline",
        "diffusion.sample",
        "app.main",
        "slurm/oneoffs/run_diffusion.slurm",
        "paper3",
        "sbatch",
        "srun",
    ):
        assert token not in source

    assert "try:" not in source
    assert "except " not in source
    assert "while " not in source
    assert "for attempt" not in source


def test_governed_training_exception_propagates_once(cli, monkeypatch):
    calls = []

    class Governed:
        DIFFUSION_REAL_DATASET_TRAINING_ENABLED = True

        @staticmethod
        def run_governed_diffusion_training(**kwargs):
            calls.append("training")
            raise RuntimeError("fixture failure")

    monkeypatch.setattr(
        cli, "_load_governed_module", lambda: Governed
    )
    monkeypatch.setattr(
        cli, "require_environment", lambda args: fake_environment()
    )
    monkeypatch.setattr(
        cli,
        "require_module_binding",
        lambda **kwargs: fake_binding(),
    )
    monkeypatch.setattr(cli, "build_adapter", lambda args: object())

    with pytest.raises(RuntimeError, match="fixture failure"):
        cli.run(make_args())

    assert calls == ["training"]


def test_launcher_contract():
    source = SLURM_PATH.read_text(encoding="utf-8")

    assert "#SBATCH --no-requeue" in source
    assert "scripts/train_paper01_diffusion_revision_production.py" in source
    assert "PROVISIONAL" in source
    for token in (
        "DIFFUSION_EXECUTION_CHECKOUT",
        "DIFFUSION_EXPECTED_SOURCE_COMMIT",
        "DIFFUSION_DATASET_ID",
        "DIFFUSION_EXPERIMENTAL_SEED",
        "DIFFUSION_ATTEMPT",
    ):
        assert token in source

    for token in (
        "app.main",
        "diffusion/train",
        "diffusion/pipeline",
        "paper3",
        "sbatch ",
        "srun",
    ):
        assert token not in source


# 16. No synthesis or evaluation automation; Stage 4 publication is expected.
def test_no_synthesis_or_evaluation_automation_in_source():
    source = CLI_PATH.read_text(encoding="utf-8")

    for token in (
        "restore_diffusion_checkpoint_state",
        "synthesize_diffusion_batch",
        "sample_governed_diffusion",
        "evaluate",
        "classifier",
    ):
        assert token not in source

    for token in (
        "prepare_diffusion_checkpoint_states",
        "publish_diffusion_checkpoint_bundle",
    ):
        assert token in source
