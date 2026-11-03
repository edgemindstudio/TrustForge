"""No-real-data tests of Diffusion's governed entry boundary."""

import pytest

from trustforge.models.paper01_revision_2026 import (
    diffusion_governed_training as governed,
)


class NeverAccessAdapter:
    def split_size(self, split):
        raise AssertionError("dataset accessed without authorization")

    def canonical_batch(self, split, positions):
        raise AssertionError("dataset accessed without authorization")


def test_runtime_gate_is_disabled(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    assert governed.DIFFUSION_REAL_DATASET_TRAINING_ENABLED is False

    with pytest.raises(PermissionError, match="not authorized"):
        governed.run_governed_diffusion_training(
            adapter=NeverAccessAdapter(),
            dataset_id="ustc_tfc2016",
            experimental_seed=42,
        )


def test_slurm_required_even_when_gate_is_mocked(monkeypatch):
    monkeypatch.setattr(
        governed, "DIFFUSION_REAL_DATASET_TRAINING_ENABLED", True
    )
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)

    with pytest.raises(PermissionError, match="Slurm"):
        governed.run_governed_diffusion_training(
            adapter=NeverAccessAdapter(),
            dataset_id="ustc_tfc2016",
            experimental_seed=42,
        )


def test_invalid_seed_precedes_dataset_access(monkeypatch):
    monkeypatch.setattr(
        governed, "DIFFUSION_REAL_DATASET_TRAINING_ENABLED", True
    )
    monkeypatch.setenv("SLURM_JOB_ID", "synthetic-test-only")

    with pytest.raises(ValueError, match="seed"):
        governed.run_governed_diffusion_training(
            adapter=NeverAccessAdapter(),
            dataset_id="ustc_tfc2016",
            experimental_seed=-1,
        )


def test_no_runtime_activation_in_source():
    assert governed.APPROVED_IMAGE_SHAPE == (40, 40, 3)
    assert governed.APPROVED_TIMESTEPS == 1000
    assert governed.APPROVED_EPOCHS == 2
    assert governed.APPROVED_BATCH_SIZE == 64
    assert governed.APPROVED_LEARNING_RATE == 2e-4
    assert governed.APPROVED_BETA_1 == 0.9
    assert governed.DIFFUSION_REAL_DATASET_TRAINING_ENABLED is False
