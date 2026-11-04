"""Internal two-epoch mechanics; no real-data authorization or publication.

The caller supplies a constructed model and family-specific step functions.
Step functions return scalar batch-mean NLL; training updates are caller-owned.
Dataset authenticity and optimizer accounting require an outer boundary.
"""
import numpy as np

from .ar_maf_execution_inputs import (
    DATASET_CLASSES, SEEDS, require_integer,
    canonical_training_batches, canonical_validation_batches,
)

def _loss(value):
    array = np.asarray(value)
    if array.shape != () or not np.isfinite(array).item():
        raise RuntimeError("step must return finite scalar batch-mean NLL")
    return float(array)

def _weights(model, expected=None):
    values = model.get_weights()
    if not isinstance(values, (list, tuple)) or not values:
        raise RuntimeError("model must expose nonempty weight state")
    result = []
    for value in values:
        array = np.asarray(value)
        if array.dtype.kind != "f" or not np.isfinite(array).all():
            raise RuntimeError("invalid model weight state")
        result.append(np.array(array, copy=True))
    signature = tuple((x.shape, x.dtype.str) for x in result)
    if expected is not None and signature != expected:
        raise RuntimeError("model weight structure changed")
    return tuple(result), signature

def _aggregate(batches, step, size):
    seen = np.zeros(size, dtype=bool)
    total = 0.0
    count = 0
    batch_count = 0
    for positions, images, labels in batches:
        # Positions are trusted only after explicit validation.
        if (
            positions.dtype != np.int64 or positions.ndim != 1
            or positions.size == 0
            or np.any(positions < 0) or np.any(positions >= size)
            or np.unique(positions).size != positions.size
            or np.any(seen[positions])
        ):
            raise RuntimeError("invalid or repeated traversal positions")
        expected_positions = positions.copy()
        value = _loss(step(images, labels))
        if not np.array_equal(positions, expected_positions):
            raise RuntimeError("step mutated traversal positions")
        seen[positions] = True
        n = len(positions)
        total += n * value
        count += n
        batch_count += 1
    if count != size or not seen.all():
        raise RuntimeError("incomplete split traversal")
    mean = total / count
    if not np.isfinite(mean):
        raise RuntimeError("nonfinite aggregate NLL")
    return mean, count, batch_count

def run_training_core(*, adapter, dataset_id, experimental_seed, model,
                      training_step, validation_step):
    """Return two-epoch history and independent in-memory best/last state.

    No files are written. This function is not a production entry point.
    """
    if dataset_id not in DATASET_CLASSES:
        raise ValueError("unsupported dataset")
    seed = require_integer(experimental_seed, "experimental_seed")
    if seed not in SEEDS:
        raise ValueError("experimental seed outside selected recipe")
    if not callable(training_step) or not callable(validation_step):
        raise TypeError("training and validation steps must be callable")
    train_size = require_integer(adapter.split_size("train"), "train_size")
    val_size = require_integer(adapter.split_size("validation"), "val_size")
    if min(train_size, val_size) <= 0:
        raise ValueError("training and validation splits must be nonempty")

    _, signature = _weights(model)
    history = []
    best_loss = float("inf")
    best_epoch = None
    best_weights = None
    updates = 0

    for epoch in range(2):
        train_loss, train_count, train_batches = _aggregate(
            canonical_training_batches(
                adapter=adapter, dataset_id=dataset_id,
                experimental_seed=seed, epoch=epoch,
            ),
            training_step, train_size,
        )
        # Detect nonfinite state before validation.
        _weights(model, signature)
        val_loss, val_count, val_batches = _aggregate(
            canonical_validation_batches(
                adapter=adapter, dataset_id=dataset_id,
            ),
            validation_step, val_size,
        )
        current, _ = _weights(model, signature)
        updates += train_batches
        history.append({
            "epoch_index": epoch,
            "train_nll": train_loss,
            "validation_nll": val_loss,
            "training_examples": train_count,
            "validation_examples": val_count,
            "training_batches": train_batches,
            "validation_batches": val_batches,
            "completed_training_steps": updates,
        })
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_weights = current

    last_weights, _ = _weights(model, signature)
    return {
        "dataset_id": dataset_id,
        "experimental_seed": seed,
        "history": tuple(history),
        "completed_epochs": 2,
        "completed_training_steps": updates,
        "best_epoch_index": best_epoch,
        "best_validation_nll": best_loss,
        "best_weights": best_weights,
        "last_weights": last_weights,
        "publication_status": "NOT_PUBLISHED",
    }
