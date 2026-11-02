"""Deterministic validation-based VAE epoch selection.

Scientific-policy candidate only. This module does not capture
weights, prepare checkpoints, or authorize production training.
"""

from __future__ import annotations

import math


def select_best_vae_epoch(history):
    """Select the earliest epoch attaining minimum validation total loss.

    The history must be a nonempty sequence of complete, consecutively
    numbered epoch records beginning at epoch zero.
    """
    if not isinstance(history, (tuple, list)) or not history:
        raise ValueError("history must be a nonempty sequence")

    best_epoch = None
    best_loss = None
    best_updates = None
    previous_updates = 0

    for expected_epoch, record in enumerate(history):
        if not isinstance(record, dict):
            raise TypeError("epoch record must be a dictionary")

        epoch = record.get("epoch")
        if type(epoch) is not int or epoch != expected_epoch:
            raise ValueError("epoch numbering must be consecutive")

        updates = record.get("completed_updates")
        if type(updates) is not int or updates <= previous_updates:
            raise ValueError("completed updates must increase")

        previous_updates = updates

        training_examples = record.get("training_examples")
        if type(training_examples) is not int or training_examples <= 0:
            raise ValueError("invalid training example count")

        validation = record.get("validation")
        if not isinstance(validation, dict):
            raise TypeError("validation record must be a dictionary")

        count = validation.get("count")
        if type(count) is not int or count <= 0:
            raise ValueError("invalid validation example count")

        loss = validation.get("total_loss")
        if type(loss) not in (int, float):
            raise TypeError("validation total loss must be numeric")

        loss = float(loss)
        if not math.isfinite(loss):
            raise ValueError("validation total loss must be finite")

        if best_loss is None or loss < best_loss:
            best_loss = loss
            best_epoch = epoch
            best_updates = updates

    return {
        "selection_metric": "validation_total_loss",
        "selection_direction": "minimize",
        "tie_break": "earliest_epoch",
        "best_epoch": best_epoch,
        "best_validation_total_loss": best_loss,
        "best_completed_updates": best_updates,
        "last_epoch": len(history) - 1,
        "last_completed_updates": previous_updates,
        "checkpoint_status": "NOT_PREPARED",
        "publication_status": "NOT_PUBLISHED",
    }


__all__ = ["select_best_vae_epoch"]
