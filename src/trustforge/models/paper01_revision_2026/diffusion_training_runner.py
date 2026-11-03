"""Synthetic-development DDPM training runner.

Consumes a caller-supplied governed adapter and preconstructed model.
Does not authenticate provenance, authorize real-data execution,
load datasets by path, publish checkpoints, or modify global gates.

No early stopping. Every epoch traverses all training and validation
positions. Best and last weights remain separate in-memory copies.
"""

from __future__ import annotations

import math

import numpy as np
import tensorflow as tf

from .variational_autoencoder_traversal import (
    training_position_batches,
    validation_position_batches,
)
from .variational_autoencoder_data_integration import canonical_vae_batch
from .diffusion_training_step import (
    diffusion_optimizer_step,
    diffusion_validation_step,
)


def _positive(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _weights(model):
    return tuple(np.array(w, copy=True) for w in model.get_weights())


def run_diffusion_training_core(
    *,
    adapter,
    model,
    optimizer,
    schedule,
    num_classes,
    experimental_seed,
    epochs,
    batch_size,
):
    """Run complete epochs and return metrics plus weight snapshots.

    This is an internal training mechanic, not an execution entrypoint.
    The caller owns identity and provenance authorization.
    """
    _positive(num_classes, "num_classes")
    if num_classes < 2:
        raise ValueError("num_classes must be >= 2")
    _positive(epochs, "epochs")
    _positive(batch_size, "batch_size")
    if (
        type(experimental_seed) is not int
        or not 0 <= experimental_seed < 2**31
    ):
        raise ValueError("invalid experimental seed")
    if not callable(model):
        raise TypeError("model must be callable")
    if not callable(getattr(model, "get_weights", None)):
        raise TypeError("model must expose get_weights")
    if not callable(getattr(optimizer, "apply_gradients", None)):
        raise TypeError("optimizer must support apply_gradients")
    if not tf.executing_eagerly():
        raise RuntimeError("Diffusion training requires eager TensorFlow")

    train_size = _positive(
        adapter.split_size("train"), "training split size"
    )
    validation_size = _positive(
        adapter.split_size("validation"), "validation split size"
    )

    history = []
    global_step = 0
    best_weights = None
    best_epoch = None
    best_mse = math.inf

    for epoch in range(epochs):
        training_total = 0.0
        training_count = 0

        for positions in training_position_batches(
            adapter=adapter,
            batch_size=batch_size,
            experimental_seed=experimental_seed,
            epoch=epoch,
        ):
            if global_step >= 2**31:
                raise RuntimeError("training seed stream exhausted")

            images, labels = canonical_vae_batch(
                adapter=adapter,
                split="train",
                logical_positions=positions,
                num_classes=num_classes,
            )

            loss = diffusion_optimizer_step(
                model=model,
                optimizer=optimizer,
                images=images,
                labels=labels,
                schedule=schedule,
                seed=(experimental_seed, global_step),
            )
            if not math.isfinite(loss):
                raise RuntimeError("nonfinite training loss")

            training_total += len(positions) * loss
            training_count += len(positions)
            global_step += 1

        if training_count != train_size:
            raise RuntimeError("incomplete training traversal")

        validation_total = 0.0
        validation_count = 0

        for batch_index, positions in enumerate(
            validation_position_batches(
                adapter=adapter, batch_size=batch_size
            )
        ):
            if batch_index >= 2**31 - 1:
                raise RuntimeError("validation seed stream exhausted")

            images, labels = canonical_vae_batch(
                adapter=adapter,
                split="validation",
                logical_positions=positions,
                num_classes=num_classes,
            )

            loss = diffusion_validation_step(
                model=model,
                images=images,
                labels=labels,
                schedule=schedule,
                seed=(experimental_seed, -(batch_index + 1)),
            )
            if not math.isfinite(loss):
                raise RuntimeError("nonfinite validation loss")

            validation_total += len(positions) * loss
            validation_count += len(positions)

        if validation_count != validation_size:
            raise RuntimeError("incomplete validation traversal")

        train_mse = training_total / training_count
        validation_mse = validation_total / validation_count

        if not all(math.isfinite(x) for x in (train_mse, validation_mse)):
            raise RuntimeError("nonfinite epoch metrics")

        history.append({
            "epoch": epoch,
            "training_examples": training_count,
            "validation_examples": validation_count,
            "train_mse": train_mse,
            "validation_mse": validation_mse,
            "completed_updates": global_step,
        })

        # Strict comparison chooses the earliest epoch on exact ties.
        if validation_mse < best_mse:
            best_mse = validation_mse
            best_epoch = epoch
            best_weights = _weights(model)

    if best_weights is None:
        raise RuntimeError("best model was not selected")

    return {
        "history": tuple(history),
        "best_epoch": best_epoch,
        "best_validation_mse": best_mse,
        "best_weights": best_weights,
        "last_weights": _weights(model),
        "completed_epochs": epochs,
        "completed_updates": global_step,
        "train_size": train_size,
        "validation_size": validation_size,
    }


__all__ = ["run_diffusion_training_core"]
