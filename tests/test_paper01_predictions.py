from pathlib import Path
import numpy as np
import pytest

from trustforge.evaluation.paper01_classifier_control import state_digest
from trustforge.evaluation.paper01_inputs import RealSplitView
from trustforge.evaluation.paper01_predictions import collect_test_predictions

RECIPE = (
    Path(__file__).resolve().parents[1]
    / "studies/paper01_benchmark/revision_2026/downstream_metric_recipe.json"
).read_bytes()


class Adapter:
    def __init__(self, fault=None):
        self.fault = fault
        self.requests = []
        self.size = 257

    def split_size(self, split):
        return self.size

    def canonical_batch(self, split, positions):
        self.requests.append(positions.copy())
        if self.fault == "size":
            self.size += 1
        classes = positions % 5
        if self.fault == "missing_class":
            classes = np.zeros(len(positions), dtype=np.int64)
        images = np.zeros((len(positions), 40, 40, 3), dtype=np.float32)
        images[:, 0, 0, 0] = classes.astype(np.float32) / np.float32(4)
        return images, np.eye(5, dtype=np.float32)[classes]


class Model:
    def __init__(self, fault=None):
        self.fault = fault
        self.weights = np.array([1.0], dtype=np.float32)
        self.counter = np.array([0], dtype=np.int64)

    def get_weights(self):
        return [self.weights.copy()]

    def execution_state(self):
        return [self.weights.copy(), self.counter.copy()]

    def predict(self, images):
        labels = np.rint(images[:, 0, 0, 0] * 4).astype(np.int64)
        probabilities = np.eye(5, dtype=np.float32)[labels]
        if self.fault == "weights":
            self.weights[0] += 1
        if self.fault == "state":
            self.counter[0] += 1
        if self.fault == "nan":
            probabilities[0, 0] = np.nan
        if self.fault == "range":
            probabilities[0, 0] = -1
        if self.fault == "sum":
            probabilities *= np.float32(0.5)
        if self.fault == "dtype":
            probabilities = probabilities.astype(np.float64)
        if self.fault == "shape":
            probabilities = probabilities[:-1]
        if self.fault == "exception":
            raise RuntimeError("fixture inference failure")
        return probabilities


def make_view(adapter=None, split="test", cls=RealSplitView):
    return cls(
        adapter=Adapter() if adapter is None else adapter,
        dataset_id="cicmaldroid2020", split=split,
        provenance_sha256="a" * 64,
    )


def run(view=None, model=None, digest=None, identity=None):
    view = make_view() if view is None else view
    model = Model() if model is None else model
    return collect_test_predictions(
        test_view=view,
        expected_test_identity=view.identity if identity is None else identity,
        expected_selected_weights_sha256=(
            state_digest(model.get_weights()) if digest is None else digest
        ),
        predict_batch=model.predict,
        get_weights=model.get_weights,
        get_execution_state=model.execution_state,
        metric_recipe_raw=RECIPE,
    )


def test_complete_order_partial_final_batch_and_prediction_identity():
    adapter, model = Adapter(), Model()
    result = run(view=make_view(adapter), model=model)
    assert [len(p) for p in adapter.requests] == [128, 128, 1]
    np.testing.assert_array_equal(
        np.concatenate(adapter.requests), np.arange(257)
    )
    np.testing.assert_array_equal(result["labels"], np.arange(257) % 5)
    np.testing.assert_array_equal(
        result["probabilities"], np.eye(5, dtype=np.float32)[np.arange(257) % 5]
    )
    report = result["report"]
    assert report["sample_count"] == 257
    assert report["maximum_image_batch"] == 128
    assert report["inference_state_unchanged"] is True
    assert report["complete_evaluation"] is False
    assert report["metrics"]["macro_f1"] == 1
    assert report["ordered_prediction_state_sha256"] == state_digest([
        result["labels"], result["probabilities"],
    ])
    assert model.weights[0] == 1
    assert model.counter[0] == 0


def test_wrong_selected_weights_rejected_before_test_access():
    adapter = Adapter()
    with pytest.raises(ValueError):
        run(view=make_view(adapter), digest="0" * 64)
    assert adapter.requests == []


@pytest.mark.parametrize("fault", [
    "weights", "state", "nan", "range", "sum", "dtype", "shape", "exception",
])
def test_inference_failures_rejected(fault):
    with pytest.raises((ValueError, RuntimeError)):
        run(model=Model(fault))


@pytest.mark.parametrize("split", ["train", "validation"])
def test_non_test_split_rejected(split):
    with pytest.raises(ValueError):
        run(view=make_view(split=split))


def test_wrong_expected_test_identity_rejected():
    view = make_view()
    identity = view.identity
    identity["declared_provenance_sha256"] = "b" * 64
    with pytest.raises(ValueError):
        run(view=view, identity=identity)


@pytest.mark.parametrize("fault", ["size", "missing_class"])
def test_changed_split_or_missing_class_rejected(fault):
    with pytest.raises((ValueError, RuntimeError)):
        run(view=make_view(Adapter(fault)))


class BrokenTraversal(RealSplitView):
    mode = "truncated"

    def iter_ordered(self, *, batch_size=128):
        for index, batch in enumerate(super().iter_ordered(batch_size=batch_size)):
            if self.mode == "truncated" and index == 1:
                return
            positions, images, labels = batch
            if self.mode == "reordered":
                positions = positions[::-1]
            yield positions, images, labels


@pytest.mark.parametrize("mode", ["truncated", "reordered"])
def test_incomplete_or_reordered_traversal_rejected(mode):
    view = make_view(cls=BrokenTraversal)
    view.mode = mode
    with pytest.raises(RuntimeError):
        run(view=view)
