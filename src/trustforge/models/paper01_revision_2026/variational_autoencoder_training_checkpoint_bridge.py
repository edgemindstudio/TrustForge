"""In-memory best/last checkpoint bridge for Revision-2026 VAE.

This module does not authorize training, create checkpoint
directories, write artifacts, or publish checkpoints.

Caller-supplied provenance and metadata remain subject to the
existing checkpoint validators and independent execution controls.
"""

from __future__ import annotations

import math

import numpy as np

from .variational_autoencoder_model import build_models
from .variational_autoencoder_checkpoint_preparation import (
    prepare_vae_checkpoint_artifacts,
)

APPROVED_EPOCHS = 2
APPROVED_BATCH_SIZE = 64
APPROVED_LEARNING_RATE = 0.001


def _checked_history(trained):
    if not isinstance(trained, dict):
        raise TypeError("training result must be a dictionary")

    history = trained.get("history")
    if not isinstance(history, list) or len(history) != APPROVED_EPOCHS:
        raise ValueError("expected exactly two completed epochs")

    if trained.get("epochs_completed") != APPROVED_EPOCHS:
        raise ValueError("training duration mismatch")

    previous_updates = 0

    for index, record in enumerate(history):
        if not isinstance(record, dict) or record.get("epoch") != index:
            raise ValueError("invalid training epoch sequence")

        updates = record.get("completed_updates")
        if type(updates) is not int or updates <= previous_updates:
            raise ValueError("invalid completed update sequence")

        previous_updates = updates

        validation = record.get("validation")
        if not isinstance(validation, dict):
            raise ValueError("missing validation record")

        loss = validation.get("total_loss")
        if type(loss) not in (int, float) or not math.isfinite(loss):
            raise ValueError("invalid validation loss")

    if trained.get("completed_updates") != previous_updates:
        raise ValueError("final update count mismatch")

    return history


def _verify_snapshot(models, weights):
    if not isinstance(weights, dict):
        raise TypeError("checkpoint weights must be a dictionary")

    encoder, decoder = models

    for name, model in (("encoder", encoder), ("decoder", decoder)):
        arrays = weights.get(name)
        if not isinstance(arrays, (tuple, list)):
            raise TypeError("missing checkpoint network weights")

        expected = model.get_weights()
        if len(arrays) != len(expected):
            raise ValueError("checkpoint weight count mismatch")

        for actual, reference in zip(arrays, expected, strict=True):
            value = np.asarray(actual)

            if value.shape != reference.shape:
                raise ValueError("checkpoint weight shape mismatch")
            if value.dtype != reference.dtype:
                raise ValueError("checkpoint weight dtype mismatch")
            if not np.all(np.isfinite(value)):
                raise ValueError("nonfinite checkpoint weight")

        model.set_weights(list(arrays))

        restored = model.get_weights()
        for actual, expected_value in zip(
            restored, arrays, strict=True
        ):
            if not np.array_equal(actual, expected_value):
                raise RuntimeError("checkpoint weight readback mismatch")


def prepare_vae_training_checkpoints(
    *,
    governed_result,
    class_mapping,
    manifest_evidence_by_role,
):
    """Prepare best and last checkpoint bundles without publishing.

    Manifest evidence must reflect the actual trained dataset,
    experimental seed, selected epoch, validation objective,
    and effective approved scientific controls.
    """
    if not isinstance(governed_result, dict):
        raise TypeError("governed result must be a dictionary")

    if governed_result.get("publication_status") != "NOT_PUBLISHED":
        raise ValueError("training result already reports publication")

    if governed_result.get("checkpoint_status") != "NOT_PREPARED":
        raise ValueError("unexpected training checkpoint status")

    trained = governed_result["training_result"]
    history = _checked_history(trained)

    selection = trained.get("epoch_selection")
    best = trained.get("best_weights")
    last = trained.get("last_weights")

    if not isinstance(selection, dict) or not isinstance(best, dict):
        raise TypeError("missing best-checkpoint selection")
    if not isinstance(last, dict):
        raise TypeError("missing last-checkpoint weights")

    best_epoch = min(
        range(APPROVED_EPOCHS),
        key=lambda epoch: history[epoch]["validation"]["total_loss"],
    )

    if selection.get("best_epoch") != best_epoch:
        raise ValueError("selected best epoch mismatch")
    if best.get("best_epoch") != best_epoch:
        raise ValueError("captured best epoch mismatch")

    best_loss = float(
        history[best_epoch]["validation"]["total_loss"]
    )
    if best.get("best_validation_total_loss") != best_loss:
        raise ValueError("best snapshot validation mismatch")

    if set(manifest_evidence_by_role) != {"best", "last"}:
        raise ValueError("both checkpoint roles are required")

    dataset_id = governed_result["dataset_id"]
    seed = governed_result["experimental_seed"]
    count = governed_result["class_count"]

    if not isinstance(class_mapping, (tuple, list)):
        raise TypeError("class mapping must be a sequence")
    if len(class_mapping) != count:
        raise ValueError("class count mismatch")

    snapshots = {
        "best": {
            "encoder": best["encoder_weights"],
            "decoder": best["decoder_weights"],
        },
        "last": last,
    }

    role_epochs = {"best": best_epoch, "last": APPROVED_EPOCHS - 1}
    prepared = {}

    for role in ("best", "last"):
        epoch = role_epochs[role]
        record = history[epoch]
        evidence = manifest_evidence_by_role[role]

        if not isinstance(evidence, dict):
            raise TypeError("manifest evidence must be a dictionary")

        expected_fields = {
            "dataset_id": dataset_id,
            "experimental_seed": seed,
            "class_count": count,
            "class_mapping": list(class_mapping),
            "checkpoint_role": role,
            "completed_epochs": epoch + 1,
            "completed_updates": record["completed_updates"],
        }

        for field, expected in expected_fields.items():
            if evidence.get(field) != expected:
                raise ValueError(
                    f"{role} checkpoint identity mismatch: {field}"
                )

        criterion = evidence.get("validation_criterion")
        if criterion != {
            "metric": "validation_total_loss",
            "direction": "minimize",
            "value": float(record["validation"]["total_loss"]),
        }:
            raise ValueError("checkpoint validation evidence mismatch")

        config = evidence.get("effective_configuration")
        if not isinstance(config, dict):
            raise TypeError("effective configuration must be a mapping")

        required_config = {
            "batch_size": APPROVED_BATCH_SIZE,
            "learning_rate": APPROVED_LEARNING_RATE,
            "epochs": APPROVED_EPOCHS,
        }

        for field, value in required_config.items():
            if config.get(field) != value:
                raise ValueError(
                    f"unapproved VAE effective configuration: {field}"
                )

        if evidence.get("selection_policy_id") != (
            "min_validation_total_loss_earliest_epoch_v1"
        ):
            raise ValueError("unexpected checkpoint selection policy")

        if evidence.get("validation_seed_policy_id") != (
            "fixed_per_batch_across_epochs_v1"
        ):
            raise ValueError("unexpected validation seed policy")

        encoder, decoder = build_models(num_classes=count)

        _verify_snapshot(
            (encoder, decoder),
            snapshots[role],
        )

        prepared[role] = prepare_vae_checkpoint_artifacts(
            encoder=encoder,
            decoder=decoder,
            manifest_evidence=evidence,
            dataset_provenance=governed_result["dataset_provenance"],
        )

        if prepared[role]["metadata"]["checkpoint_role"] != role:
            raise RuntimeError("prepared checkpoint role mismatch")

    return prepared


__all__ = ["prepare_vae_training_checkpoints"]
