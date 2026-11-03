"""Synthetic-only complete traversal tests for DDPM runner."""

import numpy as np
import pytest
import tensorflow as tf

from trustforge.models.paper01_revision_2026.diffusion_ddpm_math import (
    linear_ddpm_schedule,
)
from trustforge.models.paper01_revision_2026.diffusion_training_runner import (
    run_diffusion_training_core,
)


class SyntheticAdapter:
    def __init__(self):
        self.visits = {"train": [], "validation": []}

    def split_size(self, split):
        return {"train": 5, "validation": 3}[split]

    def canonical_batch(self, split, logical_positions):
        indices = np.asarray(logical_positions, dtype=np.int64)
        self.visits[split].extend(indices.tolist())
        images = np.full(
            (len(indices), 40, 40, 3), 0.25, dtype=np.float32
        )
        labels = np.eye(3, dtype=np.float32)[indices % 3]
        return images, labels


class TinyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv = tf.keras.layers.Conv2D(
            3, 1, use_bias=False, kernel_initializer="zeros"
        )

    def call(self, inputs, training=False):
        x, labels, timesteps = inputs
        return self.conv(x) + (
            tf.reshape(
                tf.reduce_sum(labels, axis=1)
                * 0.0 + tf.cast(timesteps, tf.float32) * 0.0,
                (-1, 1, 1, 1),
            )
        )


def test_complete_two_epoch_training():
    adapter = SyntheticAdapter()
    model = TinyModel()
    result = run_diffusion_training_core(
        adapter=adapter,
        model=model,
        optimizer=tf.keras.optimizers.SGD(0.01),
        schedule=linear_ddpm_schedule(timesteps=4),
        num_classes=3,
        experimental_seed=42,
        epochs=2,
        batch_size=2,
    )
    assert result["completed_epochs"] == 2
    assert result["completed_updates"] == 6
    assert len(result["history"]) == 2
    assert result["best_epoch"] in (0, 1)
    assert len(result["best_weights"]) > 0
    assert len(result["last_weights"]) > 0
    assert sorted(adapter.visits["train"][:5]) == list(range(5))
    assert sorted(adapter.visits["train"][5:]) == list(range(5))
    assert adapter.visits["validation"] == list(range(3)) * 2
    assert all(
        np.isfinite(row["validation_mse"])
        for row in result["history"]
    )


def test_rejects_zero_epochs_before_data_access():
    adapter = SyntheticAdapter()
    with pytest.raises(ValueError, match="epochs"):
        run_diffusion_training_core(
            adapter=adapter,
            model=TinyModel(),
            optimizer=tf.keras.optimizers.SGD(0.01),
            schedule=linear_ddpm_schedule(timesteps=4),
            num_classes=3,
            experimental_seed=42,
            epochs=0,
            batch_size=2,
        )
    assert not adapter.visits["train"]
    assert not adapter.visits["validation"]
