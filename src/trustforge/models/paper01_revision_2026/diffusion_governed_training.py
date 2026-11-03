"""Governed real-dataset boundary for Paper 1 Revision-2026 Diffusion.

Runtime activation is disabled. Methodological acceptance does not
authorize execution, checkpoint publication, or production synthesis.
"""

from __future__ import annotations

import math
import os

import tensorflow as tf


from .gan_adapter_identity import require_gan_adapter_identity
from .gan_dataset_provenance import require_gan_dataset_provenance
from .gan_v3_provenance_adapter import build_gan_v3_provenance
from .gan_checkpoint_manifest import DATASET_CLASS_COUNTS
from .diffusion_ddpm_math import linear_ddpm_schedule
from .diffusion_training_runner import run_diffusion_training_core


DIFFUSION_REAL_DATASET_TRAINING_ENABLED = False

APPROVED_IMAGE_SHAPE = (40, 40, 3)
APPROVED_DEPTH = 2
APPROVED_BASE_FILTERS = 64
APPROVED_TIME_EMBED_DIM = 128
APPROVED_TIMESTEPS = 1000
APPROVED_BETA_START = 1e-4
APPROVED_BETA_END = 2e-2
APPROVED_EPOCHS = 2
APPROVED_BATCH_SIZE = 64
APPROVED_LEARNING_RATE = 2e-4
APPROVED_BETA_1 = 0.9


def run_governed_diffusion_training(
    *, adapter, dataset_id, experimental_seed
):
    """Return completed training evidence without writing artifacts."""
    if DIFFUSION_REAL_DATASET_TRAINING_ENABLED is not True:
        raise PermissionError(
            "Diffusion real-dataset training is not authorized"
        )

    if not os.environ.get("SLURM_JOB_ID"):
        raise PermissionError(
            "Diffusion real-dataset training requires Slurm allocation"
        )

    if (
        type(experimental_seed) is not int
        or not 0 <= experimental_seed < 2**31
    ):
        raise ValueError("invalid experimental seed")

    require_gan_adapter_identity(
        dataset_id=dataset_id, adapter=adapter
    )

    verified_preflight = require_gan_dataset_provenance(
        dataset_id=dataset_id, adapter=adapter
    )

    provenance = build_gan_v3_provenance(
        dataset_id=dataset_id,
        verified_preflight=verified_preflight,
    )

    num_classes = DATASET_CLASS_COUNTS[dataset_id]

    # Load the historical model only after authorization and preflight.
    from diffusion.models import build_diffusion_model

    tf.keras.utils.set_random_seed(experimental_seed)

    model = build_diffusion_model(
        img_shape=APPROVED_IMAGE_SHAPE,
        num_classes=num_classes,
        base_filters=APPROVED_BASE_FILTERS,
        depth=APPROVED_DEPTH,
        time_emb_dim=APPROVED_TIME_EMBED_DIM,
        learning_rate=APPROVED_LEARNING_RATE,
        beta_1=APPROVED_BETA_1,
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=APPROVED_LEARNING_RATE,
        beta_1=APPROVED_BETA_1,
    )

    schedule = linear_ddpm_schedule(
        timesteps=APPROVED_TIMESTEPS,
        beta_start=APPROVED_BETA_START,
        beta_end=APPROVED_BETA_END,
    )

    trained = run_diffusion_training_core(
        adapter=adapter,
        model=model,
        optimizer=optimizer,
        schedule=schedule,
        num_classes=num_classes,
        experimental_seed=experimental_seed,
        epochs=APPROVED_EPOCHS,
        batch_size=APPROVED_BATCH_SIZE,
    )

    if trained["completed_epochs"] != APPROVED_EPOCHS:
        raise RuntimeError("incomplete governed Diffusion training")

    expected_updates = APPROVED_EPOCHS * (
        (
            adapter.split_size("train")
            + APPROVED_BATCH_SIZE - 1
        ) // APPROVED_BATCH_SIZE
    )
    if trained["completed_updates"] != expected_updates:
        raise RuntimeError("Diffusion update count mismatch")

    if (
        len(trained["history"]) != APPROVED_EPOCHS
        or trained["best_epoch"] not in range(APPROVED_EPOCHS)
    ):
        raise RuntimeError("invalid Diffusion epoch selection")

    for index, record in enumerate(trained["history"]):
        if record["epoch"] != index:
            raise RuntimeError("invalid Diffusion epoch numbering")
        if record["training_examples"] != trained["train_size"]:
            raise RuntimeError("incomplete Diffusion training coverage")
        if record["validation_examples"] != trained["validation_size"]:
            raise RuntimeError("incomplete Diffusion validation coverage")
        if not math.isfinite(record["validation_mse"]):
            raise RuntimeError("nonfinite Diffusion validation")

    expected_best = min(
        range(APPROVED_EPOCHS),
        key=lambda i: trained["history"][i]["validation_mse"],
    )

    if trained["best_epoch"] != expected_best:
        raise RuntimeError("Diffusion best epoch mismatch")

    if trained["best_validation_mse"] != (
        trained["history"][expected_best]["validation_mse"]
    ):
        raise RuntimeError("Diffusion best validation MSE mismatch")

    return {
        "dataset_id": dataset_id,
        "experimental_seed": experimental_seed,
        "class_count": num_classes,
        "dataset_provenance": provenance,
        "scientific_configuration": {
            "image_shape": list(APPROVED_IMAGE_SHAPE),
            "depth": APPROVED_DEPTH,
            "base_filters": APPROVED_BASE_FILTERS,
            "time_embedding_dim": APPROVED_TIME_EMBED_DIM,
            "timesteps": APPROVED_TIMESTEPS,
            "schedule": "linear",
            "beta_start": APPROVED_BETA_START,
            "beta_end": APPROVED_BETA_END,
            "epochs": APPROVED_EPOCHS,
            "batch_size": APPROVED_BATCH_SIZE,
            "learning_rate": APPROVED_LEARNING_RATE,
            "beta_1": APPROVED_BETA_1,
        },
        "training_result": trained,
        "checkpoint_status": "NOT_PREPARED",
        "publication_status": "NOT_PUBLISHED",
    }


__all__ = [
    "DIFFUSION_REAL_DATASET_TRAINING_ENABLED",
    "run_governed_diffusion_training",
]
