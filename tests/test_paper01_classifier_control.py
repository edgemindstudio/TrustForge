"""Control-loop checks with stand-in models; no TensorFlow execution."""

from pathlib import Path

import numpy as np
import pytest

from trustforge.evaluation.paper01_classifier_control import (
    fit_classifier_control,
)


RECIPE = (
    Path(__file__).resolve().parents[1]
    / "studies/paper01_benchmark/revision_2026/classifier_recipe.json"
).read_bytes()


class View:
    def __init__(self, split):
        self.identity = {
            "dataset_id": "cicmaldroid2020",
            "split": split, "class_count": 5,
        }
        self.size = 129
        self.visits = []

    def batch(self, positions):
        self.visits.append(positions.copy())
        return (
            positions.copy(),
            np.zeros((len(positions), 40, 40, 3), dtype=np.float32),
            np.eye(5, dtype=np.float32)[positions % 5],
        )


def run(*, fault=None, recipe=RECIPE):
    train = View("train")
    validation = View("validation")
    weights = [np.array([0.0], dtype=np.float32)]
    updates = [0]

    def train_step(images, labels):
        updates[0] += 2 if fault == "double_update" else 1
        weights[0][0] += 1
        return float("nan") if fault == "nonfinite" else (
            1.0 if len(images) == 128 else 3.0
        )

    def validation_step(images, labels):
        if fault == "validation_mutation":
            weights[0][0] += 1
        epoch = updates[0] // 2 - 1
        return 3.0 if epoch == 0 else 1.0

    def set_weights(selected):
        weights[:] = [value.copy() for value in selected]

    result = fit_classifier_control(
        training=train, validation=validation,
        evaluator_seed=42, recipe_raw=recipe,
        train_step=train_step, validation_step=validation_step,
        get_weights=lambda: weights,
        set_weights=set_weights,
        get_execution_state=lambda: [*weights, np.array(updates[0], dtype=np.int64)],
        optimizer_updates=lambda: updates[0],
    )
    return result, train, validation, weights


def test_weighted_loss_earliest_tie_patience_and_restoration():
    result, training, validation, weights = run()
    assert result["selected_epoch_index"] == 1
    assert result["completed_epochs"] == 7
    assert result["optimizer_updates"] == 14
    assert result["stopping_outcome"] == "patience"
    assert weights[0][0] == 4
    assert result["history"][0]["training_loss"] == pytest.approx(131 / 129)
    for epoch in range(7):
        positions = np.concatenate(training.visits[2 * epoch:2 * epoch + 2])
        np.testing.assert_array_equal(np.sort(positions), np.arange(129))
        expected = np.random.default_rng(
            np.random.SeedSequence([42, epoch])
        ).permutation(129)
        np.testing.assert_array_equal(positions, expected)
    np.testing.assert_array_equal(
        np.concatenate(validation.visits[:2]), np.arange(129),
    )


@pytest.mark.parametrize("fault", [
    "double_update", "nonfinite", "validation_mutation",
])
def test_execution_integrity_failures(fault):
    with pytest.raises((ValueError, RuntimeError)):
        run(fault=fault)


def test_changed_recipe_rejected_before_execution():
    with pytest.raises(ValueError, match="recipe identity"):
        run(recipe=RECIPE + b"\n")
