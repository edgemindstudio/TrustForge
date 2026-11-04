import numpy as np
import pytest

from trustforge.models.paper01_revision_2026 import ar_maf_training_core as core


class Adapter:
    def split_size(self, split):
        assert split in ("train", "validation"), "test split accessed"
        return 130 if split == "train" else 65

    def canonical_batch(self, split, positions):
        n = len(positions)
        images = np.zeros((n, 40, 40, 3), dtype=np.float32)
        labels = np.eye(2, dtype=np.float32)[positions % 2]
        return images, labels


class Model:
    def __init__(self):
        self.value = np.zeros(1, dtype=np.float32)

    def get_weights(self):
        # Deliberately expose the original array: core must copy it.
        return [self.value]


def run(validation_losses=(1.0, 2.0)):
    model = Model()
    validation_calls = []

    def train(images, labels):
        model.value += 1
        return 10.0 if len(images) == 2 else 1.0

    def validate(images, labels):
        epoch = len(validation_calls) // 2
        validation_calls.append(len(images))
        return validation_losses[epoch]

    result = core.run_training_core(
        adapter=Adapter(), dataset_id="malex", experimental_seed=42,
        model=model, training_step=train, validation_step=validate,
    )
    return result, model, validation_calls


def test_full_epochs_sample_weighting_and_independent_snapshots():
    result, model, calls = run()
    assert result["completed_epochs"] == 2
    assert result["completed_training_steps"] == 6
    assert calls == [64, 1, 64, 1]
    for epoch, row in enumerate(result["history"]):
        assert row["epoch_index"] == epoch
        assert row["training_examples"] == 130
        assert row["validation_examples"] == 65
        assert row["training_batches"] == 3
        assert row["validation_batches"] == 2
        assert row["completed_training_steps"] == 3 * (epoch + 1)
        assert row["train_nll"] == pytest.approx((128 + 20) / 130)
    assert result["best_epoch_index"] == 0
    np.testing.assert_array_equal(result["best_weights"][0], [3])
    np.testing.assert_array_equal(result["last_weights"][0], [6])
    model.value[:] = 99
    np.testing.assert_array_equal(result["last_weights"][0], [6])
    result["best_weights"][0][:] = -1
    np.testing.assert_array_equal(result["last_weights"][0], [6])
    assert result["publication_status"] == "NOT_PUBLISHED"


@pytest.mark.parametrize("losses,expected_epoch", [
    ((1.0, 1.0), 0),
    ((1.0, 1.0 - 5e-7), 1),
    ((2.0, 1.0), 1),
])
def test_exact_ties_and_small_improvements(losses, expected_epoch):
    result, _, _ = run(losses)
    assert result["best_epoch_index"] == expected_epoch
    assert result["best_validation_nll"] == pytest.approx(
        losses[expected_epoch]
    )


def test_validation_partial_batch_is_sample_weighted():
    model = Model()
    result = core.run_training_core(
        adapter=Adapter(), dataset_id="malex", experimental_seed=42,
        model=model, training_step=lambda x, y: 1.0,
        validation_step=lambda x, y: 66.0 if len(x) == 1 else 1.0,
    )
    for row in result["history"]:
        assert row["validation_nll"] == pytest.approx(2.0)


def batches(*positions):
    for values in positions:
        p = np.asarray(values, dtype=np.int64)
        yield p, None, None


@pytest.mark.parametrize("positions,message", [
    (([0, 1],), "incomplete"),
    (([0, 1], [1, 2]), "repeated"),
    (([0, 0], [1, 2]), "repeated"),
    (([-1, 0, 1],), "invalid"),
    (([0, 1, 3],), "invalid"),
    (([],), "invalid"),
])
def test_invalid_or_incomplete_traversal_rejected(positions, message):
    with pytest.raises(RuntimeError, match=message):
        core._aggregate(batches(*positions), lambda x, y: 1.0, 3)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, [1.0]])
def test_invalid_step_loss_rejected(value):
    with pytest.raises(RuntimeError, match="finite scalar"):
        core._aggregate(batches([0]), lambda x, y: value, 1)


@pytest.mark.parametrize("defect", ["nonfinite", "shape", "dtype"])
def test_corrupted_or_changed_weight_state_rejected(defect):
    model = Model()

    def train(images, labels):
        if defect == "nonfinite":
            model.value[:] = np.nan
        elif defect == "shape":
            model.value = np.zeros(2, dtype=np.float32)
        else:
            model.value = np.zeros(1, dtype=np.float64)
        return 1.0

    with pytest.raises(RuntimeError, match="weight"):
        core.run_training_core(
            adapter=Adapter(), dataset_id="malex", experimental_seed=42,
            model=model, training_step=train,
            validation_step=lambda x, y: 1.0,
        )


def test_incomplete_training_prevents_validation(monkeypatch):
    monkeypatch.setattr(
        core, "canonical_training_batches",
        lambda **kwargs: batches([0]),
    )

    def forbidden_validation(*args):
        raise AssertionError("validation reached after incomplete training")

    with pytest.raises(RuntimeError, match="incomplete"):
        core.run_training_core(
            adapter=Adapter(), dataset_id="malex", experimental_seed=42,
            model=Model(), training_step=lambda x, y: 1.0,
            validation_step=forbidden_validation,
        )
