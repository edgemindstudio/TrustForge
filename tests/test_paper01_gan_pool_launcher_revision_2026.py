"""Focused tests for the guarded CIC GAN seed-42 pool launcher wrapper."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from trustforge.models.paper01_revision_2026 import (
    gan_synthesis_authorization as authorization,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts" / "run_paper01_gan_seed42_pool.py"
LAUNCHER = (
    REPOSITORY_ROOT
    / "slurm"
    / "paper01_revision_2026"
    / "gan_pool_production.slurm"
)

SPEC = importlib.util.spec_from_file_location(
    "paper01_gan_pool_launcher",
    SCRIPT,
)
WRAPPER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WRAPPER)


def grant(tmp_path):
    return SimpleNamespace(
        dataset_id="cicmaldroid2020",
        experimental_seed=42,
        attempt=1,
        source_commit="a" * 40,
        launcher_sha256="b" * 64,
        checkpoint_directory=tmp_path / "checkpoint",
        checkpoint_run_identity="commit42c40fc_attempt01",
        checkpoint_sha256="c" * 64,
        destination=tmp_path / "pool",
        run_identity="commit" + "a" * 7 + "_attempt01",
        recipe_sha256=authorization.RECIPE_SHA256,
        authorization_id="test-authorization",
        resources=dict(authorization.EXPECTED_RESOURCES),
    )


class FakeView:
    calls = []

    def __init__(self, manifest_path, **kwargs):
        self.manifest_path = manifest_path
        self.kwargs = kwargs
        FakeView.calls.append((manifest_path, kwargs))

    def preflight(self):
        return {
            "verified_batches": 625,
            "verified_images": 10000,
        }


def install_success(monkeypatch, tmp_path, *, fail_publisher=False,
                    fail_preflight=False):
    value = grant(tmp_path)
    calls = {
        "publisher": [],
        "guard": [],
        "preflight": [],
        "launcher": [],
        "allocation": [],
        "metadata": [],
    }

    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setenv("SLURM_JOB_NODELIST", "talon08")
    monkeypatch.setattr(
        WRAPPER, "load_gan_synthesis_grant", lambda: value,
    )
    monkeypatch.setattr(
        WRAPPER,
        "verify_launcher_sha256",
        lambda root, digest: calls["launcher"].append((root, digest)),
    )
    monkeypatch.setattr(
        WRAPPER,
        "capture_gan_slurm_allocation",
        lambda: {"JobId": "12345"},
    )
    monkeypatch.setattr(
        WRAPPER,
        "validate_gan_slurm_allocation",
        lambda allocation, resources: calls["allocation"].append(
            (allocation, resources)
        ),
    )
    monkeypatch.setattr(
        WRAPPER,
        "verify_checkpoint_metadata_digest",
        lambda checkpoint, digest: calls["metadata"].append(
            (checkpoint, digest)
        ),
    )
    monkeypatch.setattr(
        WRAPPER,
        "require_gan_synthesis_authorization",
        lambda **kwargs: calls["guard"].append(kwargs) or value,
    )

    def publish(destination, **kwargs):
        calls["publisher"].append((destination, kwargs))

        if fail_publisher:
            raise RuntimeError("publisher failed")

        destination = Path(destination)
        destination.mkdir()
        manifest = destination / "manifest.json"
        manifest.write_bytes(b'{"fixture": true}\n')
        return manifest

    monkeypatch.setattr(WRAPPER, "publish_gan_cic_pool", publish)
    FakeView.calls = []

    class View(FakeView):
        def preflight(self):
            if fail_preflight:
                raise RuntimeError("preflight failed")
            return super().preflight()

    monkeypatch.setattr(WRAPPER, "GANPoolView", View)

    return value, calls, View


def test_successful_wrapper_records_execution(monkeypatch, tmp_path, capsys):
    value, calls, view = install_success(monkeypatch, tmp_path)

    result = WRAPPER.run(tmp_path)

    assert len(calls["publisher"]) == 1
    assert len(calls["guard"]) == 1
    assert len(calls["launcher"]) == 1
    assert len(calls["allocation"]) == 1
    assert len(calls["metadata"]) == 1
    assert len(view.calls) == 1
    assert result["status"] == "COMPLETE"
    assert result["slurm_job_id"] == "12345"
    assert result["source_commit"] == value.source_commit
    assert result["authorization_id"] == value.authorization_id
    assert result["checkpoint_sha256"] == value.checkpoint_sha256
    assert result["verified_batches"] == 625
    assert result["verified_images"] == 10000

    execution = json.loads(
        (value.destination / "execution.json").read_text(
            encoding="utf-8"
        )
    )
    assert execution == result
    assert not (
        value.destination / "ACCEPTANCE_FAILURE.json"
    ).exists()
    assert "GAN_POOL_EXECUTION_COMPLETE" in capsys.readouterr().out


def test_missing_authorization_blocks_publisher(monkeypatch, tmp_path):
    value = grant(tmp_path)
    publisher_calls = []

    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setenv("SLURM_JOB_NODELIST", "talon08")
    monkeypatch.setattr(
        WRAPPER,
        "load_gan_synthesis_grant",
        lambda: (_ for _ in ()).throw(
            authorization.GANSynthesisAuthorizationError("absent")
        ),
    )
    monkeypatch.setattr(
        WRAPPER,
        "publish_gan_cic_pool",
        lambda *args, **kwargs: publisher_calls.append(args),
    )

    with pytest.raises(authorization.GANSynthesisAuthorizationError):
        WRAPPER.run(tmp_path)

    assert publisher_calls == []
    assert not value.destination.exists()


def test_launcher_hash_mismatch_blocks_publisher(monkeypatch, tmp_path):
    value = grant(tmp_path)
    publisher_calls = []

    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setenv("SLURM_JOB_NODELIST", "talon08")
    monkeypatch.setattr(
        WRAPPER, "load_gan_synthesis_grant", lambda: value,
    )
    monkeypatch.setattr(
        WRAPPER,
        "verify_launcher_sha256",
        lambda root, digest: (_ for _ in ()).throw(
            PermissionError("launcher mismatch")
        ),
    )
    monkeypatch.setattr(
        WRAPPER,
        "publish_gan_cic_pool",
        lambda *args, **kwargs: publisher_calls.append(args),
    )

    with pytest.raises(PermissionError, match="launcher"):
        WRAPPER.run(tmp_path)

    assert publisher_calls == []


def test_allocation_mismatch_blocks_publisher(monkeypatch, tmp_path):
    value = grant(tmp_path)
    publisher_calls = []

    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setenv("SLURM_JOB_NODELIST", "talon08")
    monkeypatch.setattr(
        WRAPPER, "load_gan_synthesis_grant", lambda: value,
    )
    monkeypatch.setattr(
        WRAPPER,
        "verify_launcher_sha256",
        lambda root, digest: None,
    )
    monkeypatch.setattr(
        WRAPPER,
        "capture_gan_slurm_allocation",
        lambda: {"JobId": "12345"},
    )
    monkeypatch.setattr(
        WRAPPER,
        "validate_gan_slurm_allocation",
        lambda allocation, resources: (_ for _ in ()).throw(
            PermissionError("allocation mismatch")
        ),
    )
    monkeypatch.setattr(
        WRAPPER,
        "publish_gan_cic_pool",
        lambda *args, **kwargs: publisher_calls.append(args),
    )

    with pytest.raises(PermissionError, match="allocation"):
        WRAPPER.run(tmp_path)

    assert publisher_calls == []


def test_publisher_failure_has_no_execution_record(
    monkeypatch, tmp_path,
):
    value, calls, _ = install_success(
        monkeypatch, tmp_path, fail_publisher=True,
    )

    with pytest.raises(RuntimeError, match="publisher failed"):
        WRAPPER.run(tmp_path)

    assert len(calls["publisher"]) == 1
    assert not (value.destination / "execution.json").exists()


def test_preflight_failure_has_no_execution_record(
    monkeypatch, tmp_path,
):
    value, calls, _ = install_success(
        monkeypatch, tmp_path, fail_preflight=True,
    )

    with pytest.raises(RuntimeError, match="preflight failed"):
        WRAPPER.run(tmp_path)

    assert len(calls["publisher"]) == 1
    assert not (value.destination / "execution.json").exists()
    failure = json.loads(
        (value.destination / "ACCEPTANCE_FAILURE.json").read_text(
            encoding="utf-8"
        )
    )
    assert failure["status"] == "FAILED"


def test_requires_slurm_context(monkeypatch, tmp_path):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)

    with pytest.raises(RuntimeError, match="compute node"):
        WRAPPER.run(tmp_path)


def test_training_checkout_is_rejected(tmp_path, monkeypatch):
    training = tmp_path / "training"
    training.mkdir()
    monkeypatch.setattr(WRAPPER, "TRAINING_CHECKOUT", training)

    with pytest.raises(RuntimeError, match="separate"):
        WRAPPER.run(training)


def test_launcher_hash_binding(tmp_path):
    target = (
        tmp_path
        / "slurm"
        / "paper01_revision_2026"
        / "gan_pool_production.slurm"
    )
    target.parent.mkdir(parents=True)
    target.write_bytes(LAUNCHER.read_bytes())
    digest = hashlib.sha256(target.read_bytes()).hexdigest()

    WRAPPER.verify_launcher_sha256(tmp_path, digest)

    with pytest.raises(PermissionError, match="digest"):
        WRAPPER.verify_launcher_sha256(tmp_path, "0" * 64)


def test_launcher_declares_bounded_resources():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "#SBATCH --partition=talon" in text
    assert "#SBATCH --nodes=1" in text
    assert "#SBATCH --ntasks=1" in text
    assert "#SBATCH --cpus-per-task=8" in text
    assert "#SBATCH --mem=32G" in text
    assert "#SBATCH --time=12:00:00" in text
    assert "#SBATCH --no-requeue" in text
    assert "run_paper01_gan_seed42_pool.py" in text
