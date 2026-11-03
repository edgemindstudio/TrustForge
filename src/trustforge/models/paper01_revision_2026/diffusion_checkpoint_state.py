"""Unpublished in-memory Diffusion best/last state contract.

No filesystem access, checkpoint publication or training authorization.
State is bound to experimental identity and approved scientific policy.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np

from .gan_checkpoint_manifest import DATASET_CLASS_COUNTS
from . import diffusion_governed_training as policy


def _approved_configuration():
    return {
        "image_shape": list(policy.APPROVED_IMAGE_SHAPE),
        "depth": policy.APPROVED_DEPTH,
        "base_filters": policy.APPROVED_BASE_FILTERS,
        "time_embedding_dim": policy.APPROVED_TIME_EMBED_DIM,
        "timesteps": policy.APPROVED_TIMESTEPS,
        "schedule": "linear",
        "beta_start": policy.APPROVED_BETA_START,
        "beta_end": policy.APPROVED_BETA_END,
        "epochs": policy.APPROVED_EPOCHS,
        "batch_size": policy.APPROVED_BATCH_SIZE,
        "learning_rate": policy.APPROVED_LEARNING_RATE,
        "beta_1": policy.APPROVED_BETA_1,
    }


def _weights(values):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("missing weight tensors")
    result = []
    for value in values:
        if not isinstance(value, np.ndarray):
            raise TypeError("checkpoint weights must be numpy arrays")
        if value.dtype.kind != "f" or not np.all(np.isfinite(value)):
            raise ValueError("invalid checkpoint weight tensor")
        result.append(np.array(value, copy=True))
    return tuple(result)


def _digest(weights):
    hasher = hashlib.sha256()
    for array in weights:
        hasher.update(array.dtype.str.encode("ascii"))
        hasher.update(repr(array.shape).encode("ascii"))
        hasher.update(np.ascontiguousarray(array).tobytes())
    return hasher.hexdigest()


def _identity(dataset_id, seed, mapping):
    if dataset_id not in DATASET_CLASS_COUNTS:
        raise ValueError("unsupported dataset identity")
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError("invalid experimental seed")
    count = DATASET_CLASS_COUNTS[dataset_id]
    if (
        not isinstance(mapping, (list, tuple))
        or len(mapping) != count
        or any(not isinstance(v, (str, int)) or type(v) is bool
               for v in mapping)
        or len(set(mapping)) != count
    ):
        raise ValueError("invalid class mapping")
    return tuple(mapping), count


def prepare_diffusion_checkpoint_states(*, governed_result, class_mapping):
    """Prepare identity-bound best/last snapshots without publishing."""
    if not isinstance(governed_result, dict):
        raise TypeError("governed_result must be a dictionary")
    if governed_result.get("checkpoint_status") != "NOT_PREPARED":
        raise ValueError("unexpected checkpoint status")
    if governed_result.get("publication_status") != "NOT_PUBLISHED":
        raise ValueError("checkpoint publication is not permitted")

    dataset_id = governed_result.get("dataset_id")
    seed = governed_result.get("experimental_seed")
    mapping, count = _identity(dataset_id, seed, class_mapping)

    if governed_result.get("class_count") != count:
        raise ValueError("class count mismatch")
    if governed_result.get("dataset_provenance") is None:
        raise ValueError("missing dataset provenance")
    if governed_result.get("scientific_configuration") != _approved_configuration():
        raise ValueError("unapproved scientific configuration")

    trained = governed_result.get("training_result")
    if not isinstance(trained, dict):
        raise TypeError("missing completed training result")

    history = trained.get("history")
    if not isinstance(history, (tuple, list)) or len(history) != policy.APPROVED_EPOCHS:
        raise ValueError("incomplete training history")
    size = trained.get("train_size")
    validation_size = trained.get("validation_size")
    if type(size) is not int or size <= 0:
        raise ValueError("invalid training size")
    if type(validation_size) is not int or validation_size <= 0:
        raise ValueError("invalid validation size")

    steps_per_epoch = (size + policy.APPROVED_BATCH_SIZE - 1) // policy.APPROVED_BATCH_SIZE
    for epoch, record in enumerate(history):
        if not isinstance(record, dict):
            raise ValueError("invalid training history record")
        if (
            record.get("epoch") != epoch
            or record.get("training_examples") != size
            or record.get("validation_examples") != validation_size
            or record.get("completed_updates") != (epoch + 1) * steps_per_epoch
        ):
            raise ValueError("inconsistent epoch accounting")
        for key in ("train_mse", "validation_mse"):
            value = record.get(key)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("invalid epoch metric")

    expected_updates = policy.APPROVED_EPOCHS * steps_per_epoch
    if (
        trained.get("completed_epochs") != policy.APPROVED_EPOCHS
        or trained.get("completed_updates") != expected_updates
    ):
        raise ValueError("incomplete training result")

    best_epoch = min(
        range(len(history)),
        key=lambda i: history[i]["validation_mse"],
    )
    if (
        trained.get("best_epoch") != best_epoch
        or trained.get("best_validation_mse") != history[best_epoch]["validation_mse"]
    ):
        raise ValueError("best checkpoint selection mismatch")

    best = _weights(trained.get("best_weights"))
    last = _weights(trained.get("last_weights"))
    if len(best) != len(last) or any(
        a.shape != b.shape or a.dtype != b.dtype
        for a, b in zip(best, last)
    ):
        raise ValueError("best/last architecture mismatch")

    prepared = {}
    for role, epoch, weights in (
        ("best", best_epoch, best),
        ("last", policy.APPROVED_EPOCHS - 1, last),
    ):
        prepared[role] = {
            "dataset_id": dataset_id,
            "experimental_seed": seed,
            "class_mapping": mapping,
            "class_count": count,
            "scientific_configuration": _approved_configuration(),
            "checkpoint_role": role,
            "selected_epoch": epoch,
            "completed_updates": history[epoch]["completed_updates"],
            "validation_mse": history[epoch]["validation_mse"],
            "weights": weights,
            "state_sha256": _digest(weights),
            "publication_status": "NOT_PUBLISHED",
        }
    return prepared


def restore_diffusion_checkpoint_state(
    *, prepared, model_factory, requested_dataset_id,
    requested_experimental_seed, requested_class_mapping,
    requested_role,
):
    """Restore verified weights into an isolated caller-built model."""
    if requested_role not in ("best", "last"):
        raise ValueError("invalid checkpoint role")
    if not isinstance(prepared, dict):
        raise TypeError("prepared checkpoint must be a dictionary")

    mapping, count = _identity(
        requested_dataset_id,
        requested_experimental_seed,
        requested_class_mapping,
    )
    expected = {
        "dataset_id": requested_dataset_id,
        "experimental_seed": requested_experimental_seed,
        "class_mapping": mapping,
        "class_count": count,
        "checkpoint_role": requested_role,
        "scientific_configuration": _approved_configuration(),
        "publication_status": "NOT_PUBLISHED",
    }
    for key, value in expected.items():
        if prepared.get(key) != value:
            raise ValueError(f"checkpoint identity mismatch: {key}")

    weights = _weights(prepared.get("weights"))
    if _digest(weights) != prepared.get("state_sha256"):
        raise ValueError("checkpoint state digest mismatch")
    if not callable(model_factory):
        raise TypeError("model_factory must be callable")

    model = model_factory()
    if not callable(getattr(model, "get_weights", None)):
        raise TypeError("model must expose get_weights")
    if not callable(getattr(model, "set_weights", None)):
        raise TypeError("model must expose set_weights")

    current = _weights(model.get_weights())
    if len(current) != len(weights) or any(
        a.shape != b.shape or a.dtype != b.dtype
        for a, b in zip(current, weights)
    ):
        raise ValueError("checkpoint architecture mismatch")

    model.set_weights([np.array(w, copy=True) for w in weights])
    restored = _weights(model.get_weights())
    if _digest(restored) != prepared["state_sha256"]:
        raise RuntimeError("checkpoint restoration mismatch")
    return model


__all__ = [
    "prepare_diffusion_checkpoint_states",
    "restore_diffusion_checkpoint_state",
]
