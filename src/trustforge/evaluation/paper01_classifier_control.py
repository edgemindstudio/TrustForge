"""Selected classifier traversal, accounting and validation selection.

Callbacks own model mathematics. No model construction, provenance
authentication, test-split access or production execution authorization.
"""

import hashlib
import json
import math

import numpy as np

from .paper01_inputs import validate_batch


RECIPE_SHA256 = (
    "1d989ec3b9f2c1cf33ff4a765dc4618ec114b826080033bf4be02276fdf77982"
)


def selected_recipe(raw):
    if type(raw) is not bytes or hashlib.sha256(raw).hexdigest() != RECIPE_SHA256:
        raise ValueError("classifier recipe identity mismatch")
    return json.loads(raw)


def snapshot(values):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("nonempty numeric state required")
    result = []
    for value in values:
        array = np.asarray(value)
        if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
            raise ValueError("nonfinite or unsupported state")
        result.append(array.copy())
    return tuple(result)


def state_digest(values):
    digest = hashlib.sha256()
    for array in snapshot(values):
        metadata = json.dumps(
            {"shape": list(array.shape), "dtype": array.dtype.str},
            sort_keys=True, separators=(",", ":"),
        ).encode()
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        raw = array.tobytes(order="C")
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _count(value):
    if type(value) is not int or value < 0:
        raise ValueError("optimizer update count must be a nonnegative integer")
    return value


def _batch(view, positions, classes):
    observed, images, labels = view.batch(positions.copy())
    if not np.array_equal(observed, positions):
        raise RuntimeError("view changed requested positions")
    validate_batch(images, labels, count=len(positions), class_count=classes)
    return images, labels


def _finite_loss(value):
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in "fiu":
        raise ValueError("batch loss must be a numeric scalar")
    value = float(array)
    if not math.isfinite(value):
        raise ValueError("nonfinite batch loss")
    return value


def fit_classifier_control(
    *, training, validation, evaluator_seed, recipe_raw,
    train_step, validation_step, get_weights, set_weights,
    get_execution_state, optimizer_updates,
):
    """Fit through callbacks and restore earliest minimum-validation weights.

    get_execution_state must include model and optimizer variables, including
    optimizer iteration state. Callbacks return mean per-example batch loss.
    Input provenance and callback implementation require outer verification.
    """
    recipe = selected_recipe(recipe_raw)
    if type(evaluator_seed) is not int or evaluator_seed not in recipe["experimental_seeds"]:
        raise ValueError("unsupported evaluator seed")
    training_identity = training.identity
    validation_identity = validation.identity
    real_identity = training_identity.get("real", training_identity)
    if (
        real_identity.get("split") != "train"
        or validation_identity.get("split") != "validation"
        or real_identity.get("dataset_id") != validation_identity.get("dataset_id")
        or real_identity.get("class_count") != validation_identity.get("class_count")
    ):
        raise ValueError("incompatible training/validation views")
    for view in (training, validation):
        if type(view.size) is not int or view.size <= 0:
            raise ValueError("nonempty input views required")
    if _count(optimizer_updates()) != 0:
        raise RuntimeError("fresh classifier optimizer required")

    classes = validation_identity["class_count"]
    batch_size = recipe["training"]["batch_size"]
    controls = recipe["implementation_selection"]["validation_and_selection"]
    initial_digest = state_digest(get_weights())
    best_loss = math.inf
    best_weights = None
    best_epoch = None
    wait = 0
    history = []

    for epoch in range(controls["maximum_epochs"]):
        if training.identity != training_identity or validation.identity != validation_identity:
            raise RuntimeError("input identity changed")
        positions = np.random.default_rng(
            np.random.SeedSequence([evaluator_seed, epoch])
        ).permutation(training.size).astype(np.int64, copy=False)
        train_total = np.float64(0)
        train_count = 0
        for start in range(0, training.size, batch_size):
            selected = positions[start:start + batch_size]
            images, labels = _batch(training, selected, classes)
            before = _count(optimizer_updates())
            loss = _finite_loss(train_step(images, labels))
            if _count(optimizer_updates()) != before + 1:
                raise RuntimeError("training must perform exactly one update")
            snapshot(get_execution_state())
            train_total += np.float64(len(selected)) * np.float64(loss)
            train_count += len(selected)

        before_validation = state_digest(get_execution_state())
        validation_total = np.float64(0)
        validation_count = 0
        for start in range(0, validation.size, batch_size):
            selected = np.arange(
                start, min(start + batch_size, validation.size), dtype=np.int64,
            )
            images, labels = _batch(validation, selected, classes)
            loss = _finite_loss(validation_step(images, labels))
            validation_total += np.float64(len(selected)) * np.float64(loss)
            validation_count += len(selected)
        if state_digest(get_execution_state()) != before_validation:
            raise RuntimeError("validation changed model or optimizer state")
        if train_count != training.size or validation_count != validation.size:
            raise RuntimeError("incomplete epoch coverage")
        train_loss = float(train_total / train_count)
        val_loss = float(validation_total / validation_count)
        if not math.isfinite(train_loss) or not math.isfinite(val_loss):
            raise RuntimeError("nonfinite epoch loss")

        expected_updates = (epoch + 1) * (
            (training.size + batch_size - 1) // batch_size
        )
        if _count(optimizer_updates()) != expected_updates:
            raise RuntimeError("epoch update accounting mismatch")
        history.append({
            "epoch_index": epoch,
            "training_loss": train_loss,
            "validation_loss": val_loss,
            "training_examples": train_count,
            "validation_examples": validation_count,
            "optimizer_updates": expected_updates,
        })
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_weights = snapshot(get_weights())
            wait = 0
        else:
            wait += 1
        if wait >= controls["patience"]:
            break

    if best_weights is None:
        raise RuntimeError("no selected classifier state")
    set_weights([value.copy() for value in best_weights])
    selected_digest = state_digest(best_weights)
    if state_digest(get_weights()) != selected_digest:
        raise RuntimeError("selected classifier weights were not restored")
    return {
        "recipe_sha256": RECIPE_SHA256,
        "evaluator_seed": evaluator_seed,
        "initial_weights_sha256": initial_digest,
        "selected_weights_sha256": selected_digest,
        "selected_epoch_index": best_epoch,
        "selected_validation_loss": best_loss,
        "completed_epochs": len(history),
        "optimizer_updates": _count(optimizer_updates()),
        "stopping_outcome": (
            "patience" if wait >= controls["patience"] else "maximum_epochs"
        ),
        "history": history,
    }
