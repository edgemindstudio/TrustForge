import numpy as np
import pytest

from trustforge.models.paper01_revision_2026.ar_maf_family_steps import (
    checked_steps,
)


class Variable:
    def __init__(self, value):
        self.value = np.asarray(value)

    def numpy(self):
        return self.value


class Model:
    def __init__(self):
        self.weights = [Variable(np.array([1.0], dtype=np.float32))]


class Optimizer:
    def __init__(self):
        self.iterations = Variable(np.array(0, dtype=np.int64))
        self.slot = Variable(np.array([0.0], dtype=np.float32))
        self.variables = [self.iterations, self.slot]


def bind(model, optimizer, train=None, validate=None):
    return checked_steps(
        model=model, optimizer=optimizer,
        training_step=train or (lambda x, y: 1.0),
        validation_step=validate or (lambda x, y: 2.0),
    )


def test_valid_update_and_read_only_validation():
    model, optimizer = Model(), Optimizer()

    def update(x, y):
        optimizer.iterations.value += 1
        optimizer.slot.value += 0.5
        model.weights[0].value += 0.25
        return 1.5

    train, validate = bind(model, optimizer, train=update)
    assert train(None, None) == 1.5
    assert validate(None, None) == 2.0
    assert int(optimizer.iterations.numpy()) == 1
    np.testing.assert_array_equal(model.weights[0].numpy(), [1.25])


@pytest.mark.parametrize("increment", [0, 2])
def test_missing_or_double_update_rejected(increment):
    model, optimizer = Model(), Optimizer()

    def update(x, y):
        optimizer.iterations.value += increment
        return 1.0

    train, _ = bind(model, optimizer, train=update)
    with pytest.raises(RuntimeError, match="exactly once"):
        train(None, None)


@pytest.mark.parametrize("target", [
    "model", "optimizer_slot", "iteration",
    "model_structure", "optimizer_structure", "model_dtype",
])
def test_validation_mutation_rejected(target):
    model, optimizer = Model(), Optimizer()

    def mutate(x, y):
        if target == "model":
            model.weights[0].value += 1
        elif target == "optimizer_slot":
            optimizer.slot.value += 1
        elif target == "iteration":
            optimizer.iterations.value += 1
        elif target == "model_structure":
            model.weights.append(Variable(np.zeros(1, dtype=np.float32)))
        elif target == "optimizer_structure":
            optimizer.variables.append(
                Variable(np.zeros(1, dtype=np.float32))
            )
        else:
            model.weights[0].value = model.weights[0].value.astype(
                np.float64
            )
        return 2.0

    _, validate = bind(model, optimizer, validate=mutate)
    with pytest.raises(RuntimeError, match="validation changed"):
        validate(None, None)


@pytest.mark.parametrize("phase", ["train", "validate"])
@pytest.mark.parametrize("loss", [np.nan, np.inf, [1.0]])
def test_invalid_loss_rejected(phase, loss):
    model, optimizer = Model(), Optimizer()

    def update(x, y):
        optimizer.iterations.value += 1
        return loss

    train, validate = bind(
        model, optimizer, train=update,
        validate=lambda x, y: loss,
    )
    with pytest.raises(RuntimeError, match="finite scalar"):
        (train if phase == "train" else validate)(None, None)


@pytest.mark.parametrize("target", ["model", "optimizer"])
def test_nonfinite_state_after_update_rejected(target):
    model, optimizer = Model(), Optimizer()

    def update(x, y):
        optimizer.iterations.value += 1
        variable = model.weights[0] if target == "model" else optimizer.slot
        variable.value[:] = np.nan
        return 1.0

    train, _ = bind(model, optimizer, train=update)
    with pytest.raises(RuntimeError, match="invalid model or optimizer state"):
        train(None, None)
