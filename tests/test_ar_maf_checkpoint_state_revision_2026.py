from copy import deepcopy

import numpy as np
import pytest

from trustforge.models.paper01_revision_2026.ar_maf_checkpoint_state import (
    capture_model_state,
    encode_model_snapshot,
    restore_model_snapshot,
)


class Variable:
    def __init__(self, value, trainable=True):
        self.value = np.asarray(value)
        self.trainable = trainable

    def numpy(self):
        return self.value


class Model:
    def __init__(self, offset=0):
        self.weights = [
            Variable(np.array([1, 2], dtype=np.float32) + offset),
            Variable(np.array([[3]], dtype=np.float32) + offset, False),
        ]
        self.assignments = 0

    def set_weights(self, values):
        self.assignments += 1
        for variable, value in zip(self.weights, values):
            variable.value = np.array(value, copy=True)


def snapshot(model):
    return [v.numpy().copy() for v in model.weights]


def test_capture_is_complete_ordered_and_independent():
    model = Model()
    specification, state = capture_model_state(model)
    assert specification == [
        {
            "key": "model_0000", "shape": [2],
            "dtype": "float32", "trainable": True,
        },
        {
            "key": "model_0001", "shape": [1, 1],
            "dtype": "float32", "trainable": False,
        },
    ]
    model.weights[0].value[:] = 99
    np.testing.assert_array_equal(state["model_0000"], [1, 2])


def test_snapshot_restores_supplied_weights_not_current_model_values():
    source = Model()
    selected = snapshot(source)
    source.weights[0].value[:] = 99
    specification, raw = encode_model_snapshot(
        model=source, weights=selected,
    )
    target = Model(offset=10)
    restore_model_snapshot(
        model=target, specification=specification, raw=raw,
    )
    assert target.assignments == 1
    for expected, actual in zip(selected, target.weights):
        np.testing.assert_array_equal(actual.numpy(), expected)
    selected[0][:] = -1
    np.testing.assert_array_equal(target.weights[0].numpy(), [1, 2])


@pytest.mark.parametrize("defect", [
    "shape", "dtype", "trainability", "order", "missing",
])
def test_specification_mismatch_rejected_before_assignment(defect):
    source = Model()
    specification, raw = encode_model_snapshot(
        model=source, weights=snapshot(source),
    )
    specification = deepcopy(specification)
    if defect == "shape":
        specification[0]["shape"] = [3]
    elif defect == "dtype":
        specification[0]["dtype"] = "float64"
    elif defect == "trainability":
        specification[0]["trainable"] = False
    elif defect == "order":
        specification.reverse()
    else:
        specification.pop()
    target = Model(offset=10)
    before = snapshot(target)
    with pytest.raises(ValueError, match="specification mismatch"):
        restore_model_snapshot(
            model=target, specification=specification, raw=raw,
        )
    assert target.assignments == 0
    for expected, actual in zip(before, target.weights):
        np.testing.assert_array_equal(actual.numpy(), expected)


def test_corrupt_archive_rejected_before_assignment():
    source = Model()
    specification, raw = encode_model_snapshot(
        model=source, weights=snapshot(source),
    )
    target = Model()
    with pytest.raises(ValueError):
        restore_model_snapshot(
            model=target, specification=specification, raw=raw[:-20],
        )
    assert target.assignments == 0


@pytest.mark.parametrize("defect", [
    "empty", "scalar", "dtype", "nonfinite", "trainability",
])
def test_unsupported_model_state_rejected(defect):
    model = Model()
    if defect == "empty":
        model.weights = []
    elif defect == "scalar":
        model.weights[0].value = np.array(1, dtype=np.float32)
    elif defect == "dtype":
        model.weights[0].value = np.zeros(2, dtype=np.float64)
    elif defect == "nonfinite":
        model.weights[0].value[0] = np.nan
    else:
        model.weights[0].trainable = 1
    with pytest.raises((ValueError, TypeError)):
        capture_model_state(model)


@pytest.mark.parametrize("defect", ["count", "shape", "dtype", "nonfinite"])
def test_invalid_snapshot_rejected(defect):
    model = Model()
    values = snapshot(model)
    if defect == "count":
        values.pop()
    elif defect == "shape":
        values[0] = np.zeros(3, dtype=np.float32)
    elif defect == "dtype":
        values[0] = values[0].astype(np.float64)
    else:
        values[0][0] = np.nan
    with pytest.raises((ValueError, TypeError)):
        encode_model_snapshot(model=model, weights=values)


def test_failed_assignment_verification_is_reported():
    class BrokenModel(Model):
        def set_weights(self, values):
            self.assignments += 1

    source = Model()
    specification, raw = encode_model_snapshot(
        model=source, weights=snapshot(source),
    )
    target = BrokenModel(offset=10)
    with pytest.raises(RuntimeError, match="verification failed"):
        restore_model_snapshot(
            model=target, specification=specification, raw=raw,
        )
    assert target.assignments == 1
