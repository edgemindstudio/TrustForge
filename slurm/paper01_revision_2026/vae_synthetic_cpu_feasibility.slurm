#!/bin/bash
#SBATCH --job-name=p1r1-vae-cpu-pilot
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00

set -euo pipefail
cd "$HOME/TrustForge"

export CUDA_VISIBLE_DEVICES=-1
export TF_CPP_MIN_LOG_LEVEL=2
export OMP_NUM_THREADS=2
export TF_NUM_INTEROP_THREADS=2
export TF_NUM_INTRAOP_THREADS=2
export PYTHONPATH=src

/home/bruno.fonkeng/miniforge3/envs/gcs_py310/bin/python - <<'PYTHON'
import math
import time
import numpy as np

import optree
import keras
import tensorflow as tf

from trustforge.models.paper01_revision_2026.variational_autoencoder_training_runner import (
    run_synthetic_vae_training,
)

class SyntheticAdapter:
    def __init__(self):
        rng = np.random.default_rng(2026)
        self.data = {}

        for split, count in (
            ("train", 512),
            ("validation", 64),
        ):
            images = rng.uniform(
                0.0, 1.0, (count, 40, 40, 3)
            ).astype(np.float32)

            labels = np.eye(2, dtype=np.float32)[
                np.arange(count) % 2
            ]

            self.data[split] = (images, labels)

    def split_size(self, split):
        return self.data[split][0].shape[0]

    def canonical_batch(self, split, positions):
        images, labels = self.data[split]
        return (
            images[positions].copy(),
            labels[positions].copy(),
        )

assert not tf.config.list_logical_devices("GPU")

print("===== SLURM VAE SYNTHETIC CPU PILOT =====", flush=True)
print("SLURM_JOB_ID:", __import__("os").environ.get("SLURM_JOB_ID"))
print("DATASET: SYNTHETIC ONLY")
print("TRAINING EXAMPLES: 512")
print("VALIDATION EXAMPLES: 64")
print("LEARNING RATE: 0.001")
print("BATCH SIZE: 64")
print("EPOCHS: 1", flush=True)

start = time.perf_counter()

result = run_synthetic_vae_training(
    adapter=SyntheticAdapter(),
    num_classes=2,
    experimental_seed=42,
    epochs=1,
    batch_size=64,
    learning_rate=0.001,
    development_mode="synthetic_only",
)

elapsed = time.perf_counter() - start
epoch = result["history"][0]

assert result["completed_updates"] == 8
assert epoch["training_examples"] == 512
assert epoch["validation"]["count"] == 64

for section in ("training", "validation"):
    for key in (
        "total_loss",
        "reconstruction_loss",
        "kl_loss",
    ):
        assert math.isfinite(epoch[section][key])

assert result["checkpoint_status"] == "NOT_PREPARED"
assert result["publication_status"] == "NOT_PUBLISHED"

print("PILOT STATUS: PASS")
print("WALL TIME SECONDS:", round(elapsed, 3))
print("COMPLETED UPDATES:", result["completed_updates"])
print("TRAINING METRICS:", epoch["training"])
print("VALIDATION METRICS:", epoch["validation"])
print("CHECKPOINTS PUBLISHED: NO")
print("REAL DATASET TRAINING: NO")
PYTHON
