"""Synthetic-only Diffusion checkpoint-state tests."""

from copy import deepcopy

import numpy as np
import pytest

from trustforge.models.paper01_revision_2026 import (
    diffusion_governed_training as policy,
)
from trustforge.models.paper01_revision_2026.diffusion_checkpoint_state import (
    prepare_diffusion_checkpoint_states,
    restore_diffusion_checkpoint_state,
)


class DummyModel:
    def __init__(self):
        self.values = [np.zeros((2, 3), dtype=np.float32)]

    def get_weights(self):
        return [v.copy() for v in self.values]

    def set_weights(self, weights):
        self.values = [v.copy() for v in weights]


def fixture():
    config = {
        "image_shape": [40, 40, 3], "depth": 2,
        "base_filters": 64, "time_embedding_dim": 128,
        "timesteps": 1000, "schedule": "linear",
        "beta_start": 1e-4, "beta_end": 2e-2,
        "epochs": 2, "batch_size": 64,
        "learning_rate": 2e-4, "beta_1": 0.9,
    }
    history = (
        {
            "epoch": 0, "training_examples": 3,
            "validation_examples": 2, "completed_updates": 1,
            "train_mse": 1.2, "validation_mse": 0.7,
        },
        {
            "epoch": 1, "training_examples": 3,
            "validation_examples": 2, "completed_updates": 2,
            "train_mse": 1.0, "validation_mse": 0.8,
        },
    )
    return {
        "dataset_id": "malex",
        "experimental_seed": 42,
        "class_count": 2,
        "dataset_provenance": {"synthetic": True},
        "scientific_configuration": config,
        "checkpoint_status": "NOT_PREPARED",
        "publication_status": "NOT_PUBLISHED",
        "training_result": {
            "history": history, "best_epoch": 0,
            "best_validation_mse": 0.7,
            "best_weights": (np.full((2, 3), 1, dtype=np.float32),),
            "last_weights": (np.full((2, 3), 2, dtype=np.float32),),
            "completed_epochs": 2, "completed_updates": 2,
            "train_size": 3, "validation_size": 2,
        },
    }


MAPPING = ("benign", "malware")


def prepared():
    return prepare_diffusion_checkpoint_states(
        governed_result=fixture(), class_mapping=MAPPING
    )


def test_prepare_distinct_best_and_last():
    records = prepared()
    assert set(records) == {"best", "last"}
    assert records["best"]["selected_epoch"] == 0
    assert records["last"]["selected_epoch"] == 1
    assert records["best"]["state_sha256"] != records["last"]["state_sha256"]


@pytest.mark.parametrize("role,value", [("best", 1), ("last", 2)])
def test_isolated_restoration(role, value):
    model = restore_diffusion_checkpoint_state(
        prepared=prepared()[role],
        model_factory=DummyModel,
        requested_dataset_id="malex",
        requested_experimental_seed=42,
        requested_class_mapping=MAPPING,
        requested_role=role,
    )
    np.testing.assert_array_equal(
        model.get_weights()[0],
        np.full((2, 3), value, dtype=np.float32),
    )


def test_rejects_wrong_identity():
    record = prepared()["best"]
    with pytest.raises(ValueError, match="identity mismatch"):
        restore_diffusion_checkpoint_state(
            prepared=record, model_factory=DummyModel,
            requested_dataset_id="malex",
            requested_experimental_seed=43,
            requested_class_mapping=MAPPING,
            requested_role="best",
        )


def test_rejects_tampered_weights():
    record = prepared()["best"]
    record["weights"][0][0, 0] = 9.0
    with pytest.raises(ValueError, match="digest mismatch"):
        restore_diffusion_checkpoint_state(
            prepared=record, model_factory=DummyModel,
            requested_dataset_id="malex",
            requested_experimental_seed=42,
            requested_class_mapping=MAPPING,
            requested_role="best",
        )


def test_rejects_unapproved_configuration():
    result = fixture()
    result["scientific_configuration"]["epochs"] = 3
    with pytest.raises(ValueError, match="configuration"):
        prepare_diffusion_checkpoint_states(
            governed_result=result, class_mapping=MAPPING
        )


def test_rejects_incorrect_selection():
    result = fixture()
    result["training_result"]["best_epoch"] = 1
    with pytest.raises(ValueError, match="selection mismatch"):
        prepare_diffusion_checkpoint_states(
            governed_result=result, class_mapping=MAPPING
        )


def test_runtime_gate_remains_disabled():
    assert policy.DIFFUSION_REAL_DATASET_TRAINING_ENABLED is False
