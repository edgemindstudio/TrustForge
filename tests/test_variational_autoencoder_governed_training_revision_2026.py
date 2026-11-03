"""Isolated contract tests for the governed VAE training boundary.

All authorization and Slurm state changes are fixture-local.
No real dataset, optimizer training or checkpoint publication occurs.
"""

import pytest

from trustforge.models.paper01_revision_2026 import (
    variational_autoencoder_governed_training as governed,
)


DATASET = "ustc_tfc2016"
SEED = 42
TRAIN_SIZE = 129


class Adapter:
    def split_size(self, split):
        if split == "train":
            return TRAIN_SIZE
        raise AssertionError("unexpected split request")


def install_controlled_path(monkeypatch, *, events, train=None):
    adapter = Adapter()

    monkeypatch.setattr(
        governed,
        "VAE_REAL_DATASET_TRAINING_ENABLED",
        True,
    )
    monkeypatch.setenv("SLURM_JOB_ID", "synthetic-test-only")

    def check_identity(*, dataset_id, adapter):
        events.append("identity")
        assert dataset_id == DATASET
        assert isinstance(adapter, Adapter)

    def preflight(*, dataset_id, adapter):
        events.append("preflight")
        assert dataset_id == DATASET
        return {"dataset_id": dataset_id, "source_contract": "fixture"}

    def build_provenance(*, dataset_id, verified_preflight):
        events.append("provenance")
        assert verified_preflight["dataset_id"] == dataset_id
        return {"dataset_id": dataset_id, "fixture_only": True}

    def core(**kwargs):
        events.append("training")
        assert kwargs["adapter"] is adapter
        assert kwargs["num_classes"] == 9
        assert kwargs["experimental_seed"] == SEED
        assert kwargs["epochs"] == 2
        assert kwargs["batch_size"] == 64
        assert kwargs["learning_rate"] == 0.001

        result = {
            "epochs_completed": 2,
            "completed_updates": 6,
            "publication_status": "NOT_PUBLISHED",
        }
        if train is not None:
            result.update(train)
        return result

    monkeypatch.setattr(
        governed, "require_gan_adapter_identity", check_identity
    )
    monkeypatch.setattr(
        governed, "require_gan_dataset_provenance", preflight
    )
    monkeypatch.setattr(
        governed, "build_gan_v3_provenance", build_provenance
    )
    monkeypatch.setattr(
        governed, "_run_vae_training_core", core
    )

    return adapter


def test_production_gate_rejects_before_dataset_access(monkeypatch):
    monkeypatch.setattr(
        governed, "VAE_REAL_DATASET_TRAINING_ENABLED", False
    )
    monkeypatch.setenv("SLURM_JOB_ID", "synthetic-test-only")

    with pytest.raises(PermissionError, match="not authorized"):
        governed.run_governed_vae_training(
            adapter=object(),
            dataset_id=DATASET,
            experimental_seed=SEED,
        )


def test_slurm_allocation_required_before_preflight(monkeypatch):
    monkeypatch.setattr(
        governed, "VAE_REAL_DATASET_TRAINING_ENABLED", True
    )
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)

    with pytest.raises(PermissionError, match="Slurm"):
        governed.run_governed_vae_training(
            adapter=object(),
            dataset_id=DATASET,
            experimental_seed=SEED,
        )


def test_governed_path_order_and_fixed_controls(monkeypatch):
    events = []
    adapter = install_controlled_path(monkeypatch, events=events)

    result = governed.run_governed_vae_training(
        adapter=adapter,
        dataset_id=DATASET,
        experimental_seed=SEED,
    )

    assert events == [
        "identity",
        "preflight",
        "provenance",
        "training",
    ]
    assert result["dataset_id"] == DATASET
    assert result["experimental_seed"] == SEED
    assert result["class_count"] == 9
    assert result["dataset_provenance"]["fixture_only"] is True
    assert result["checkpoint_status"] == "NOT_PREPARED"
    assert result["publication_status"] == "NOT_PUBLISHED"


def test_failed_preflight_never_reaches_training(monkeypatch):
    events = []
    adapter = install_controlled_path(monkeypatch, events=events)

    def reject(*, dataset_id, adapter):
        events.append("preflight_rejected")
        raise ValueError("source integrity mismatch")

    monkeypatch.setattr(
        governed, "require_gan_dataset_provenance", reject
    )

    with pytest.raises(ValueError, match="source integrity"):
        governed.run_governed_vae_training(
            adapter=adapter,
            dataset_id=DATASET,
            experimental_seed=SEED,
        )

    assert events == ["identity", "preflight_rejected"]


def test_failed_identity_never_reaches_preflight(monkeypatch):
    events = []
    adapter = install_controlled_path(monkeypatch, events=events)

    def reject(*, dataset_id, adapter):
        events.append("identity_rejected")
        raise TypeError("adapter identity mismatch")

    monkeypatch.setattr(
        governed, "require_gan_adapter_identity", reject
    )

    with pytest.raises(TypeError, match="identity mismatch"):
        governed.run_governed_vae_training(
            adapter=adapter,
            dataset_id=DATASET,
            experimental_seed=SEED,
        )

    assert events == ["identity_rejected"]


@pytest.mark.parametrize(
    "override, expected_message",
    [
        ({"epochs_completed": 1}, "incomplete"),
        ({"completed_updates": 5}, "update count"),
        ({"publication_status": "PUBLISHED"}, "publication"),
    ],
)
def test_invalid_training_result_rejected(
    monkeypatch, override, expected_message
):
    events = []
    adapter = install_controlled_path(
        monkeypatch,
        events=events,
        train=override,
    )

    with pytest.raises(RuntimeError, match=expected_message):
        governed.run_governed_vae_training(
            adapter=adapter,
            dataset_id=DATASET,
            experimental_seed=SEED,
        )

    assert events[-1] == "training"


def test_production_gate_remains_disabled_in_source():
    from pathlib import Path

    path = Path(governed.__file__)
    source = path.read_text(encoding="utf-8")

    assert "VAE_REAL_DATASET_TRAINING_ENABLED = False" in source
