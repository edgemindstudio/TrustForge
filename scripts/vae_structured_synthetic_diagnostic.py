"""Bounded synthetic-only VAE latent-utilization experiment.

No real dataset loading, filesystem checkpoint publication,
or production training authorization.

Descriptive diagnostics are not production acceptance criteria.
"""

import json
import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

from trustforge.models.paper01_revision_2026.variational_autoencoder_model import (
    build_models,
)
from trustforge.models.paper01_revision_2026.variational_autoencoder_training_runner import (
    run_synthetic_vae_training,
)
from trustforge.models.paper01_revision_2026.variational_autoencoder_structured_synthetic_adapter import (
    StructuredSyntheticVAEAdapter,
)


TRAIN_COUNT = 512
VALIDATION_COUNT = 128
DATA_SEED = 2026
EXPERIMENTAL_SEED = 42
EPOCHS = 5
BATCH_SIZE = 64
LEARNING_RATE = 0.001
ACTIVE_THRESHOLD = 0.01


def observations(adapter):
    train_x, train_y = adapter.canonical_batch(
        "train", np.arange(TRAIN_COUNT, dtype=np.int64)
    )
    valid_x, valid_y = adapter.canonical_batch(
        "validation", np.arange(VALIDATION_COUNT, dtype=np.int64)
    )
    return train_x, train_y, valid_x, valid_y


def class_template_baseline(train_x, train_y, valid_x, valid_y):
    train_ids = train_y.argmax(axis=1)
    valid_ids = valid_y.argmax(axis=1)

    templates = np.stack([
        train_x[train_ids == class_id].mean(axis=0)
        for class_id in range(2)
    ])

    predicted = templates[valid_ids]

    return float(np.mean((valid_x - predicted) ** 2))


def assess(encoder, decoder, valid_x, valid_y):
    internal_x = tf.convert_to_tensor(
        valid_x * 2.0 - 1.0, dtype=tf.float32
    )
    labels = tf.convert_to_tensor(valid_y, dtype=tf.float32)

    mean, logvar = encoder(
        [internal_x, labels], training=False
    )
    mean_np = mean.numpy()
    logvar_np = logvar.numpy()

    kl = 0.5 * np.sum(
        mean_np.astype(np.float64) ** 2
        + np.exp(logvar_np.astype(np.float64))
        - 1.0
        - logvar_np.astype(np.float64),
        axis=1,
    )

    # Use posterior means for a deterministic reconstruction
    # diagnostic. This is not an ELBO estimate.
    reconstructed = decoder(
        [mean, labels], training=False
    ).numpy()

    recon_canonical = (reconstructed + 1.0) / 2.0

    reconstruction_mse = float(
        np.mean((valid_x - recon_canonical) ** 2)
    )

    posterior_mean_variance = np.var(
        mean_np.astype(np.float64), axis=0
    )

    # Separate decoder sensitivities at fixed, common inputs.
    fixed_z = tf.zeros((VALIDATION_COUNT, 100), tf.float32)
    shifted_z = tf.ones((VALIDATION_COUNT, 100), tf.float32)
    labels_zero = tf.one_hot(
        np.zeros(VALIDATION_COUNT, dtype=np.int32), 2
    )
    labels_one = tf.one_hot(
        np.ones(VALIDATION_COUNT, dtype=np.int32), 2
    )

    decoded_zero = decoder(
        [fixed_z, labels_zero], training=False
    ).numpy()
    decoded_shifted = decoder(
        [shifted_z, labels_zero], training=False
    ).numpy()
    decoded_class_one = decoder(
        [fixed_z, labels_one], training=False
    ).numpy()

    return {
        "reconstruction_mse_posterior_mean": reconstruction_mse,
        "posterior_kl_mean": float(kl.mean()),
        "posterior_mean_variance_mean": float(
            posterior_mean_variance.mean()
        ),
        "active_latent_dimensions_descriptive": int(
            np.count_nonzero(
                posterior_mean_variance > ACTIVE_THRESHOLD
            )
        ),
        "active_dimension_variance_threshold": ACTIVE_THRESHOLD,
        "decoder_latent_shift_mse": float(
            np.mean((decoded_zero - decoded_shifted) ** 2)
        ),
        "decoder_class_shift_mse": float(
            np.mean((decoded_zero - decoded_class_one) ** 2)
        ),
    }


def main():
    if tf.config.list_logical_devices("GPU"):
        raise RuntimeError("CPU-only diagnostic required")

    adapter = StructuredSyntheticVAEAdapter(
        train_count=TRAIN_COUNT,
        validation_count=VALIDATION_COUNT,
        seed=DATA_SEED,
    )
    train_x, train_y, valid_x, valid_y = observations(adapter)

    tf.keras.utils.set_random_seed(EXPERIMENTAL_SEED)
    initial_encoder, initial_decoder = build_models(
        num_classes=2
    )

    initial_metrics = assess(
        initial_encoder, initial_decoder, valid_x, valid_y
    )
    template_mse = class_template_baseline(
        train_x, train_y, valid_x, valid_y
    )

    start = time.perf_counter()

    trained = run_synthetic_vae_training(
        adapter=adapter,
        num_classes=2,
        experimental_seed=EXPERIMENTAL_SEED,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        development_mode="synthetic_only",
    )

    elapsed = time.perf_counter() - start

    if trained["epochs_completed"] != EPOCHS:
        raise RuntimeError("incomplete diagnostic epochs")
    if trained["completed_updates"] != EPOCHS * 8:
        raise RuntimeError("unexpected update count")
    if trained["publication_status"] != "NOT_PUBLISHED":
        raise RuntimeError("unexpected checkpoint publication")

    last_metrics = assess(
        trained["encoder"], trained["decoder"],
        valid_x, valid_y
    )

    best = trained["best_weights"]
    best_encoder, best_decoder = build_models(num_classes=2)
    best_encoder.set_weights(best["encoder_weights"])
    best_decoder.set_weights(best["decoder_weights"])

    best_metrics = assess(
        best_encoder, best_decoder, valid_x, valid_y
    )

    result = {
        "diagnostic": "structured_synthetic_only",
        "configuration": {
            "train_count": TRAIN_COUNT,
            "validation_count": VALIDATION_COUNT,
            "data_seed": DATA_SEED,
            "experimental_seed": EXPERIMENTAL_SEED,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
        },
        "class_template_validation_mse": template_mse,
        "initialized": initial_metrics,
        "trained_last": last_metrics,
        "trained_best": best_metrics,
        "best_epoch": trained["epoch_selection"]["best_epoch"],
        "history": trained["history"],
        "completed_updates": trained["completed_updates"],
        "runner_seconds": elapsed,
        "checkpoint_status": "NOT_PREPARED",
        "publication_status": "NOT_PUBLISHED",
        "real_data_training": "NOT_AUTHORIZED",
    }

    print(
        json.dumps(result, sort_keys=True, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
