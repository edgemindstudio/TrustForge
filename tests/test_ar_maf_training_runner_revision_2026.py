import hashlib
from types import SimpleNamespace
import subprocess

import pytest

from trustforge.models.paper01_revision_2026 import ar_maf_training_runner as runner


def test_disabled_gate_prevents_source_or_adapter_access(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("disabled runner crossed execution boundary")

    monkeypatch.setattr(runner, "capture_source_identity", forbidden)
    with pytest.raises(RuntimeError, match="remains disabled"):
        runner.run_training_and_prepare_checkpoint(
            repository_root="unused", family="ar", dataset_id="malex",
            experimental_seed=42, adapter=object(),
        )


@pytest.fixture
def repository(tmp_path):
    root = tmp_path / "repository"
    root.mkdir()

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(root), *args],
            stderr=subprocess.STDOUT,
        )

    git("init", "-q")
    git("config", "user.name", "Fixture")
    git("config", "user.email", "fixture@example.invalid")
    recipe = root / runner.RECIPE_PATH
    recipe.parent.mkdir(parents=True)
    recipe.write_bytes(b"fixture recipe")
    (root / "model.py").write_text("value = 1\n")
    git("add", "--", "model.py", runner.RECIPE_PATH)
    git("commit", "-qm", "Fixture source")
    return root, git


def test_source_capture_hashes_tracked_python_and_recipe(repository):
    root, git = repository
    head, files = runner.capture_source_identity(root)
    assert head == git("rev-parse", "HEAD").decode().strip()
    assert set(files) == {"model.py", runner.RECIPE_PATH}
    assert files["model.py"] == hashlib.sha256(
        (root / "model.py").read_bytes()
    ).hexdigest()


@pytest.mark.parametrize("staged", [False, True])
def test_source_capture_rejects_tracked_changes(repository, staged):
    root, git = repository
    (root / "model.py").write_text("value = 2\n")
    if staged:
        git("add", "--", "model.py")
    with pytest.raises(subprocess.CalledProcessError):
        runner.capture_source_identity(root)


@pytest.fixture
def execution(tmp_path, monkeypatch):
    raw = b"fixture recipe"
    path = tmp_path / runner.RECIPE_PATH
    path.parent.mkdir(parents=True)
    path.write_bytes(raw)
    source = ("a" * 40, {
        runner.RECIPE_PATH: hashlib.sha256(raw).hexdigest(),
    })
    events = []
    count = {"value": 0}
    checked = object()
    original = object()
    model = object()
    evidence = {
        "identity": {"fixture": True},
        "training_examples": 65,
        "validation_examples": 3,
    }
    trained = {
        "completed_epochs": 2,
        "completed_training_steps": 4,
        "best_weights": ["best"],
        "last_weights": ["last"],
        "history": [{
            "training_examples": 65, "validation_examples": 3,
        }] * 2,
    }

    def capture(root):
        events.append("capture")
        return source

    def setup(**kwargs):
        assert kwargs["adapter"] is original
        events.append("setup")
        return {
            "adapter": checked, "evidence": evidence, "model": model,
            "optimizer": SimpleNamespace(iterations=SimpleNamespace(
                numpy=lambda: count["value"],
            )),
            "training_step": "train", "validation_step": "validate",
        }

    def core(**kwargs):
        assert kwargs["adapter"] is checked
        assert kwargs["model"] is model
        events.append("train")
        count["value"] = 4
        return trained

    def encode(**kwargs):
        events.append("encode")
        return [{"fixture": True}], kwargs["weights"][0].encode()

    def manifest(**kwargs):
        events.append("manifest")
        assert kwargs["source_commit"] == source[0]
        assert kwargs["identity"] is evidence["identity"]
        return {"fixture": True}

    monkeypatch.setattr(runner, "TRAINING_EXECUTION_ENABLED", True)
    monkeypatch.setattr(runner, "capture_source_identity", capture)
    monkeypatch.setattr(runner, "prepare_training_setup", setup)
    monkeypatch.setattr(runner, "run_training_core", core)
    monkeypatch.setattr(runner, "encode_model_snapshot", encode)
    monkeypatch.setattr(runner, "prepare_checkpoint_manifest", manifest)
    args = {
        "repository_root": tmp_path, "family": "ar",
        "dataset_id": "malex", "experimental_seed": 42,
        "adapter": original,
    }
    return args, events, count, trained, source


def test_orchestration_uses_checked_adapter_and_prepares_only(execution):
    args, events, _, _, _ = execution
    result = runner.run_training_and_prepare_checkpoint(**args)
    assert events == [
        "capture", "setup", "train", "encode", "encode", "capture", "manifest",
    ]
    assert result["publication_status"] == "NOT_PUBLISHED"
    assert result["best_raw"] == b"best"
    assert result["last_raw"] == b"last"


def test_nonfresh_optimizer_blocks_training(execution):
    args, events, count, _, _ = execution
    count["value"] = 1
    with pytest.raises(RuntimeError, match="fresh optimizer"):
        runner.run_training_and_prepare_checkpoint(**args)
    assert "train" not in events


@pytest.mark.parametrize("defect", ["epochs", "updates", "exposure"])
def test_inconsistent_training_result_blocks_preparation(execution, defect):
    args, events, _, trained, _ = execution
    if defect == "epochs":
        trained["completed_epochs"] = 1
    elif defect == "updates":
        trained["completed_training_steps"] = 3
    else:
        trained["history"][0]["training_examples"] = 64
    with pytest.raises(RuntimeError):
        runner.run_training_and_prepare_checkpoint(**args)
    assert "encode" not in events
    assert "manifest" not in events


def test_source_change_blocks_manifest(execution, monkeypatch):
    args, events, _, _, source = execution
    calls = []

    def capture(root):
        calls.append(1)
        return source if len(calls) == 1 else ("b" * 40, source[1])

    monkeypatch.setattr(runner, "capture_source_identity", capture)
    with pytest.raises(RuntimeError, match="source identity changed"):
        runner.run_training_and_prepare_checkpoint(**args)
    assert "manifest" not in events


@pytest.mark.parametrize("family,dataset,seed", [
    ("gan", "cicmaldroid2020", 42),
    ("ar", "malex", 42),
    ("maf", "ustc_tfc2016", 42),
    ("ar", "cicmaldroid2020", 43),
    ("maf", "cicmaldroid2020", True),
    ("ar", "cicmaldroid2020", 42.0),
])
def test_bounded_allowance_rejects_other_scope_before_source_access(
    monkeypatch, family, dataset, seed,
):
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid pilot crossed source boundary")

    monkeypatch.setattr(runner, "TRAINING_EXECUTION_ENABLED", False)
    monkeypatch.setattr(runner, "capture_source_identity", forbidden)
    with pytest.raises(RuntimeError, match="bounded CIC seed42 scope"):
        runner.run_training_and_prepare_checkpoint(
            repository_root="unused", family=family, dataset_id=dataset,
            experimental_seed=seed, adapter=object(),
            bounded_integration=True,
        )


@pytest.mark.parametrize("family", ["ar", "maf"])
def test_bounded_allowance_reaches_source_check_with_general_gate_disabled(
    monkeypatch, family,
):
    def stop_at_source(*args, **kwargs):
        raise ValueError("source boundary reached")

    monkeypatch.setattr(runner, "TRAINING_EXECUTION_ENABLED", False)
    monkeypatch.setattr(runner, "capture_source_identity", stop_at_source)
    with pytest.raises(ValueError, match="source boundary reached"):
        runner.run_training_and_prepare_checkpoint(
            repository_root="unused", family=family,
            dataset_id="cicmaldroid2020", experimental_seed=42,
            adapter=object(), bounded_integration=True,
        )


def test_nonboolean_bounded_flag_rejected():
    with pytest.raises(TypeError, match="must be boolean"):
        runner.run_training_and_prepare_checkpoint(
            repository_root="unused", family="ar",
            dataset_id="cicmaldroid2020", experimental_seed=42,
            adapter=object(), bounded_integration=1,
        )
