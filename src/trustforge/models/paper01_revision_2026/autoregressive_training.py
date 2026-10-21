"""Training primitives for the governed revision-2026 Conditional PixelCNN.

This module is revision-specific. Historical autoregressive training code
remains unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf

from .autoregressive_observation import (
    coupled_rgb_censored_logistic_nll,
)


@dataclass(frozen=True)
class AutoregressiveTrainingConfig:
    """Consumed training controls for the revision-2026 PixelCNN."""

    epochs: int
    patience: int
    save_every: int
    min_log_scale: float
    max_log_scale: float

    def validate(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")

        if self.patience <= 0:
            raise ValueError("patience must be positive")

        if self.save_every <= 0:
            raise ValueError("save_every must be positive")

        if self.min_log_scale >= self.max_log_scale:
            raise ValueError(
                "min_log_scale must be smaller than max_log_scale"
            )


def _assert_canonical_batch(
    x: tf.Tensor,
    y_onehot: tf.Tensor,
) -> None:
    """Validate the revision-2026 model-facing batch contract."""
    x = tf.convert_to_tensor(x)
    y_onehot = tf.convert_to_tensor(y_onehot)

    tf.debugging.assert_rank(
        x,
        4,
        message="canonical images must be rank-4 NHWC tensors",
    )

    tf.debugging.assert_equal(
        tf.shape(x)[-1],
        3,
        message="canonical images must have exactly three channels",
    )

    tf.debugging.assert_type(
        x,
        tf.float32,
        message="canonical images must be float32",
    )

    tf.debugging.assert_greater_equal(
        x,
        tf.constant(0.0, dtype=tf.float32),
        message="canonical images contain values below 0",
    )

    tf.debugging.assert_less_equal(
        x,
        tf.constant(1.0, dtype=tf.float32),
        message="canonical images contain values above 1",
    )

    tf.debugging.assert_rank(
        y_onehot,
        2,
        message="class conditioning must be rank-2 one-hot data",
    )

    tf.debugging.assert_equal(
        tf.shape(x)[0],
        tf.shape(y_onehot)[0],
        message="image and label batch sizes differ",
    )

    tf.debugging.assert_type(
        y_onehot,
        tf.float32,
        message="one-hot class conditioning must be float32",
    )

    tf.debugging.assert_all_finite(
        x,
        "canonical images contain non-finite values",
    )

    tf.debugging.assert_all_finite(
        y_onehot,
        "class conditioning contains non-finite values",
    )


def _compute_nll(
    model: tf.keras.Model,
    x: tf.Tensor,
    y_onehot: tf.Tensor,
    *,
    config: AutoregressiveTrainingConfig,
    training: bool,
) -> tf.Tensor:
    """Evaluate the governed canonical-space censored-logistic NLL."""
    _assert_canonical_batch(x, y_onehot)

    params = model(
        [x, y_onehot],
        training=training,
    )

    loss = coupled_rgb_censored_logistic_nll(
        x,
        params,
        min_log_scale=config.min_log_scale,
        max_log_scale=config.max_log_scale,
    )

    tf.debugging.assert_all_finite(
        loss,
        "non-finite revision-2026 autoregressive loss",
    )

    return loss


def train_step(
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    x: tf.Tensor,
    y_onehot: tf.Tensor,
    *,
    config: AutoregressiveTrainingConfig,
) -> tf.Tensor:
    """Execute one optimizer step and return the scalar training NLL."""
    with tf.GradientTape() as tape:
        loss = _compute_nll(
            model,
            x,
            y_onehot,
            config=config,
            training=True,
        )

    gradients = tape.gradient(
        loss,
        model.trainable_variables,
    )

    missing = [
        variable.name
        for variable, gradient in zip(
            model.trainable_variables,
            gradients,
        )
        if gradient is None
    ]

    if missing:
        raise RuntimeError(
            "missing autoregressive gradients for: "
            + ", ".join(missing)
        )

    for variable, gradient in zip(
        model.trainable_variables,
        gradients,
    ):
        tf.debugging.assert_all_finite(
            gradient,
            f"non-finite gradient for {variable.name}",
        )

    optimizer.apply_gradients(
        zip(gradients, model.trainable_variables)
    )

    return loss


def validation_step(
    model: tf.keras.Model,
    x: tf.Tensor,
    y_onehot: tf.Tensor,
    *,
    config: AutoregressiveTrainingConfig,
) -> tf.Tensor:
    """Evaluate one validation batch without updating model state."""
    return _compute_nll(
        model,
        x,
        y_onehot,
        config=config,
        training=False,
    )


def fit_revision_pixelcnn(
    *,
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    checkpoint_dir: Path | str,
    config: AutoregressiveTrainingConfig,
    writer: Optional[tf.summary.SummaryWriter] = None,
) -> list[dict[str, float]]:
    """Train with validation-based early stopping and governed checkpoints."""
    config.validate()

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_path = checkpoint_dir / "AR_revision2026_best.weights.h5"
    last_path = checkpoint_dir / "AR_revision2026_last.weights.h5"

    best_val = np.inf
    no_improve = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        train_metric = tf.keras.metrics.Mean()
        val_metric = tf.keras.metrics.Mean()

        train_batches = 0

        for x_batch, y_batch in train_ds:
            loss = train_step(
                model,
                optimizer,
                x_batch,
                y_batch,
                config=config,
            )

            train_metric.update_state(loss)
            train_batches += 1

        if train_batches == 0:
            raise RuntimeError(
                "training dataset produced zero batches"
            )

        val_batches = 0

        for x_batch, y_batch in val_ds:
            loss = validation_step(
                model,
                x_batch,
                y_batch,
                config=config,
            )

            val_metric.update_state(loss)
            val_batches += 1

        if val_batches == 0:
            raise RuntimeError(
                "validation dataset produced zero batches"
            )

        train_loss = float(train_metric.result())
        val_loss = float(val_metric.result())

        if not np.isfinite(train_loss):
            raise RuntimeError(
                "non-finite epoch training loss"
            )

        if not np.isfinite(val_loss):
            raise RuntimeError(
                "non-finite epoch validation loss"
            )

        record = {
            "epoch": float(epoch),
            "train_nll": train_loss,
            "val_nll": val_loss,
        }

        history.append(record)

        print(
            f"[revision2026][autoregressive] "
            f"epoch={epoch:04d} "
            f"train_nll={train_loss:.6f} "
            f"val_nll={val_loss:.6f}"
        )

        if writer is not None:
            with writer.as_default():
                tf.summary.scalar(
                    "autoregressive/train_nll",
                    train_loss,
                    step=epoch,
                )
                tf.summary.scalar(
                    "autoregressive/val_nll",
                    val_loss,
                    step=epoch,
                )
                writer.flush()

        if (
            epoch == 1
            or epoch % config.save_every == 0
        ):
            snapshot = (
                checkpoint_dir
                / f"AR_revision2026_epoch_{epoch:04d}.weights.h5"
            )

            model.save_weights(
                str(snapshot)
            )

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            no_improve = 0

            model.save_weights(
                str(best_path)
            )
        else:
            no_improve += 1

            if no_improve >= config.patience:
                print(
                    "[revision2026][autoregressive] "
                    f"early stop after {config.patience} "
                    "epochs without validation improvement"
                )
                break

    model.save_weights(
        str(last_path)
    )

    return history


__all__ = [
    "AutoregressiveTrainingConfig",
    "fit_revision_pixelcnn",
    "train_step",
    "validation_step",
]
