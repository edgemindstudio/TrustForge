"""Focused tests for governed revision-2026 RBM training orchestration."""

from __future__ import annotations

import numpy as np
import pytest

from trustforge.models.paper01_revision_2026.restricted_boltzmann_training import (
    RBMTrainingResult,
    train_shared_conditional_rbm,
)


class TinyAdapter:
    def __init__(self):
        self.train_images = np.stack(
            [
                np.full((40, 40, 3), 0.20, dtype=np.float32),
                np.full((40, 40, 3), 0.30, dtype=np.float32),
                np.full((40, 40, 3), 0.70, dtype=np.float32),
                np.full((40, 40, 3), 0.80, dtype=np.float32),
            ],
            axis=0,
        )

        self.train_labels = np.array(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )

        self.validation_images = np.stack(
            [
                np.full((40, 40, 3), 0.25, dtype=np.float32),
                np.full((40, 40, 3), 0.75, dtype=np.float32),
            ],
            axis=0,
        )

        self.validation_labels = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )

        self.access_log = []

    def split_size(
        self,
        split,
    ):
        self.access_log.append(
            ("split_size", split)
        )

        if split == "train":
            return 4

        if split == "validation":
            return 2

        raise AssertionError(
            f"unexpected split access: {split}"
        )

    def canonical_batch(
        self,
        split,
        logical_positions,
    ):
        positions = np.asarray(
            logical_positions,
            dtype=np.int64,
        )

        self.access_log.append(
            (
                "canonical_batch",
                split,
                tuple(
                    positions.tolist()
                ),
            )
        )

        if split == "train":
            return (
                self.train_images[
                    positions
                ],
                self.train_labels[
                    positions
                ],
            )

        if split == "validation":
            return (
                self.validation_images[
                    positions
                ],
                self.validation_labels[
                    positions
                ],
            )

        raise AssertionError(
            f"unexpected split access: {split}"
        )


def _positions():
    return {
        0: np.array(
            [0, 1],
            dtype=np.int64,
        ),
        1: np.array(
            [2, 3],
            dtype=np.int64,
        ),
    }


def _train(
    *,
    adapter=None,
    dataset_id="tiny-rbm-test",
    experimental_seed=42,
    positions=None,
):
    return train_shared_conditional_rbm(
        adapter=adapter or TinyAdapter(),
        dataset_id=dataset_id,
        experimental_seed=experimental_seed,
        train_class_positions=(
            _positions()
            if positions is None
            else positions
        ),
    )


def test_training_returns_governed_result():
    result = _train()

    assert isinstance(
        result,
        RBMTrainingResult,
    )


def test_tiny_training_budget_is_exactly_two_updates():
    result = _train()

    assert result.budget.batches_per_nominal_epoch == 1
    assert result.budget.max_updates == 2
    assert result.optimizer_steps_completed == 2


def test_validation_occurs_exactly_at_zero_B_two_B():
    result = _train()

    assert tuple(
        item.step
        for item in result.validation_history
    ) == (
        0,
        1,
        2,
    )


def test_final_validation_is_authoritative_step():
    result = _train()

    assert result.budget.authoritative_step == 2
    assert result.validation_history[-1].step == 2


def test_exactly_three_validation_observations_are_recorded():
    result = _train()

    assert len(
        result.validation_history
    ) == 3


def test_validation_diagnostics_are_finite():
    result = _train()

    for observation in result.validation_history:
        assert np.isfinite(
            observation.diagnostics.reconstruction_mse
        )


def test_training_accesses_only_train_and_validation_splits():
    adapter = TinyAdapter()

    _train(
        adapter=adapter
    )

    observed_splits = set()

    for entry in adapter.access_log:
        observed_splits.add(
            entry[1]
        )

    assert observed_splits == {
        "train",
        "validation",
    }


def test_training_never_accesses_test_split():
    adapter = TinyAdapter()

    _train(
        adapter=adapter
    )

    assert all(
        entry[1] != "test"
        for entry in adapter.access_log
    )


def test_same_identity_reproduces_exact_final_state():
    first = _train(
        dataset_id="same-dataset",
        experimental_seed=42,
    )

    second = _train(
        dataset_id="same-dataset",
        experimental_seed=42,
    )

    first_state = first.model.state_dict()
    second_state = second.model.state_dict()

    for name in first_state:
        np.testing.assert_array_equal(
            first_state[name],
            second_state[name],
        )


def test_changed_seed_changes_final_state():
    first = _train(
        experimental_seed=42
    )

    second = _train(
        experimental_seed=43
    )

    first_state = first.model.state_dict()
    second_state = second.model.state_dict()

    assert any(
        not np.array_equal(
            first_state[name],
            second_state[name],
        )
        for name in first_state
    )


def test_changed_dataset_identity_changes_final_state():
    first = _train(
        dataset_id="dataset-A"
    )

    second = _train(
        dataset_id="dataset-B"
    )

    first_state = first.model.state_dict()
    second_state = second.model.state_dict()

    assert any(
        not np.array_equal(
            first_state[name],
            second_state[name],
        )
        for name in first_state
    )


def test_training_class_inventory_must_cover_exact_train_split():
    bad_positions = {
        0: np.array(
            [0],
            dtype=np.int64,
        ),
        1: np.array(
            [2, 3],
            dtype=np.int64,
        ),
    }

    with pytest.raises(ValueError):
        _train(
            positions=bad_positions
        )


def test_training_requires_zero_based_contiguous_class_ids():
    bad_positions = {
        1: np.array(
            [0, 1],
            dtype=np.int64,
        ),
        2: np.array(
            [2, 3],
            dtype=np.int64,
        ),
    }

    with pytest.raises(ValueError):
        _train(
            positions=bad_positions
        )


def test_training_rejects_adapter_label_disagreement_with_traversal():
    adapter = TinyAdapter()

    adapter.train_labels = adapter.train_labels.copy()
    adapter.train_labels[
        0
    ] = np.array(
        [0.0, 1.0],
        dtype=np.float32,
    )

    with pytest.raises(ValueError):
        _train(
            adapter=adapter
        )


def test_training_rejects_non_float32_training_images():
    adapter = TinyAdapter()

    adapter.train_images = adapter.train_images.astype(
        np.float64
    )

    with pytest.raises(TypeError):
        _train(
            adapter=adapter
        )


def test_training_rejects_non_float32_training_labels():
    adapter = TinyAdapter()

    adapter.train_labels = adapter.train_labels.astype(
        np.float64
    )

    with pytest.raises(TypeError):
        _train(
            adapter=adapter
        )


def test_training_rejects_empty_dataset_id():
    with pytest.raises(ValueError):
        _train(
            dataset_id="   "
        )


def test_training_does_not_create_checkpoint_files(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(
        tmp_path
    )

    _train()

    assert list(
        tmp_path.iterdir()
    ) == []
