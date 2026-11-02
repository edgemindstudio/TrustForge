"""Synthetic-development VAE training and validation runner.

Consumes a caller-supplied canonical adapter. Does not load real
datasets, write checkpoints, publish commits, or authorize production
training. An interrupted run returns no completed training result.

The experimental training schedule remains subject to scientific
approval for Paper 1 revision-2026.
"""

from __future__ import annotations

import math

import tensorflow as tf

from .variational_autoencoder_model import build_models
from .variational_autoencoder_data_integration import canonical_vae_batch
from .variational_autoencoder_traversal import (
    training_position_batches,
    validation_position_batches,
)
from .variational_autoencoder_training_step import vae_training_step
from .variational_autoencoder_epoch_selection import (
    select_best_vae_epoch,
)
from .variational_autoencoder_weight_selection import (
    capture_vae_weights,
    consider_vae_best_weights,
)
from .variational_autoencoder_validation_step import (
    vae_validation_batch,
    aggregate_validation_batches,
)


def _positive_int(value, name):
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _bounded_seed(value):
    if type(value) is not int:
        raise TypeError("experimental_seed must be an integer")
    if not 0 <= value <= 2**31 - 1:
        raise ValueError("experimental_seed exceeds supported range")
    return value


def run_synthetic_vae_training(
    *,
    adapter,
    num_classes,
    experimental_seed,
    epochs,
    batch_size,
    learning_rate,
    development_mode,
):
    """Train and validate fully in memory using a synthetic adapter.

    A development_mode string does not authenticate the adapter's data.
    The caller remains responsible for synthetic-only provenance.
    """
    if development_mode != "synthetic_only":
        raise PermissionError(
            "VAE runner restricted to synthetic development"
        )

    num_classes = _positive_int(num_classes, "num_classes")
    if num_classes < 2:
        raise ValueError("num_classes must be at least two")

    epochs = _positive_int(epochs, "epochs")
    batch_size = _positive_int(batch_size, "batch_size")
    seed = _bounded_seed(experimental_seed)

    if type(learning_rate) not in (int, float):
        raise TypeError("learning_rate must be numeric")
    learning_rate = float(learning_rate)
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")

    if not tf.executing_eagerly():
        raise RuntimeError("VAE training requires eager TensorFlow")

    if not callable(getattr(adapter, "split_size", None)):
        raise TypeError("adapter must provide split_size")
    if not callable(getattr(adapter, "canonical_batch", None)):
        raise TypeError("adapter must provide canonical_batch")

    train_size = _positive_int(
        adapter.split_size("train"), "training split size"
    )
    validation_size = _positive_int(
        adapter.split_size("validation"), "validation split size"
    )

    tf.keras.utils.set_random_seed(seed)

    encoder, decoder = build_models(num_classes=num_classes)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate
    )

    history = []
    global_step = 0
    best_weights = None

    for epoch in range(epochs):
        weighted = {
            "total_loss": 0.0,
            "reconstruction_loss": 0.0,
            "kl_loss": 0.0,
        }
        observed_train = 0

        for positions in training_position_batches(
            adapter=adapter,
            batch_size=batch_size,
            experimental_seed=seed,
            epoch=epoch,
        ):
            if global_step >= 2**31:
                raise RuntimeError(
                    "VAE stateless training seed range exhausted"
                )

            images, labels = canonical_vae_batch(
                adapter=adapter,
                split="train",
                logical_positions=positions,
                num_classes=num_classes,
            )

            losses = vae_training_step(
                encoder=encoder,
                decoder=decoder,
                optimizer=optimizer,
                canonical_images=images,
                class_labels=labels,
                seed=tf.constant(
                    [seed, global_step],
                    dtype=tf.int32,
                ),
            )

            count = len(positions)
            observed_train += count

            for key in weighted:
                weighted[key] += count * losses[key]

            global_step += 1

        if observed_train != train_size:
            raise RuntimeError("incomplete training traversal")

        validation_results = []
        observed_validation = 0

        for batch_index, positions in enumerate(
            validation_position_batches(
                adapter=adapter,
                batch_size=batch_size,
            )
        ):
            if batch_index >= 2**31 - 1:
                raise RuntimeError(
                    "VAE validation seed range exhausted"
                )

            images, labels = canonical_vae_batch(
                adapter=adapter,
                split="validation",
                logical_positions=positions,
                num_classes=num_classes,
            )

            validation_results.append(
                vae_validation_batch(
                    encoder=encoder,
                    decoder=decoder,
                    canonical_images=images,
                    class_labels=labels,
                    seed=tf.constant(
                        [seed, -(batch_index + 1)],
                        dtype=tf.int32,
                    ),
                )
            )

            observed_validation += len(positions)

        if observed_validation != validation_size:
            raise RuntimeError("incomplete validation traversal")

        validation = aggregate_validation_batches(
            validation_results
        )

        if validation["count"] != validation_size:
            raise RuntimeError(
                "validation aggregation count mismatch"
            )

        training = {
            key: value / observed_train
            for key, value in weighted.items()
        }

        if not all(
            math.isfinite(value)
            for value in training.values()
        ):
            raise RuntimeError("nonfinite training metrics")

        history.append({
            "epoch": epoch,
            "training_examples": observed_train,
            "training": training,
            "validation": validation,
            "completed_updates": global_step,
        })

        best_weights = consider_vae_best_weights(
            current=best_weights,
            epoch=epoch,
            validation_total_loss=validation["total_loss"],
            encoder=encoder,
            decoder=decoder,
        )

    selection = select_best_vae_epoch(history)

    if best_weights is None:
        raise RuntimeError("VAE best weights were not captured")

    if (
        best_weights["best_epoch"] != selection["best_epoch"]
        or best_weights["best_validation_total_loss"]
        != selection["best_validation_total_loss"]
    ):
        raise RuntimeError("VAE best-weight selection mismatch")

    last_weights = capture_vae_weights(
        encoder=encoder,
        decoder=decoder,
    )

    return {
        "encoder": encoder,
        "decoder": decoder,
        "optimizer": optimizer,
        "experimental_seed": seed,
        "epochs_completed": len(history),
        "completed_updates": global_step,
        "history": history,
        "epoch_selection": selection,
        "best_weights": best_weights,
        "last_weights": last_weights,
        "checkpoint_status": "NOT_PREPARED",
        "publication_status": "NOT_PUBLISHED",
    }


__all__ = ["run_synthetic_vae_training"]
