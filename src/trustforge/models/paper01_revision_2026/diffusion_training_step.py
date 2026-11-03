"""Stateless DDPM noise objective and optimizer step.

This module provides training mechanics only. It does not authenticate
datasets, authorize execution, choose training duration, load or
publish checkpoints, or submit compute jobs.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf


def _validate_inputs(*, images, labels, schedule, seed):
    if not isinstance(seed, (tuple, list)) or len(seed) != 2:
        raise ValueError("seed must contain exactly two integers")

    if any(
        type(item) is not int or not -(2**31) <= item < 2**31
        for item in seed
    ):
        raise ValueError("stateless seed components out of range")

    x = np.asarray(images)
    y = np.asarray(labels)

    if x.ndim != 4 or x.shape[0] == 0:
        raise ValueError("images must be nonempty NHWC")
    if y.ndim != 2 or y.shape[0] != x.shape[0] or y.shape[1] < 2:
        raise ValueError("invalid class-label shape")
    if x.dtype != np.float32 or y.dtype != np.float32:
        raise TypeError("images and labels must be float32")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("nonfinite canonical batch")
    if np.any(x < 0) or np.any(x > 1):
        raise ValueError("canonical images must be in [0,1]")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("class labels must be one-hot")
    if not np.all(np.sum(y, axis=1) == 1):
        raise ValueError("class labels must have one active class")

    if not isinstance(schedule, dict):
        raise TypeError("schedule must be a dictionary")

    T = schedule.get("timesteps")
    if type(T) is not int or T < 2:
        raise ValueError("invalid diffusion timestep count")
    if schedule.get("schedule") != "linear":
        raise ValueError("unsupported governed diffusion schedule")

    bars = np.asarray(schedule.get("alpha_bars"))
    betas = np.asarray(schedule.get("betas"))
    alphas = np.asarray(schedule.get("alphas"))

    if any(a.shape != (T,) for a in (bars, betas, alphas)):
        raise ValueError("schedule coefficient shape mismatch")
    if not all(np.all(np.isfinite(a)) for a in (bars, betas, alphas)):
        raise ValueError("nonfinite schedule coefficients")
    if np.any(bars <= 0) or np.any(bars >= 1):
        raise ValueError("invalid cumulative schedule coefficients")
    if np.any(betas <= 0) or np.any(betas >= 1):
        raise ValueError("invalid beta coefficients")
    if not np.allclose(alphas + betas, 1.0, atol=1e-10):
        raise ValueError("inconsistent alpha/beta coefficients")
    if not np.allclose(
        bars, np.cumprod(alphas), rtol=1e-7, atol=1e-10
    ):
        raise ValueError("inconsistent cumulative schedule")

    return x, y, bars, T


def stateless_ddpm_batch(
    *, images, labels, schedule, seed
):
    """Prepare deterministic noised inputs and noise targets.

    The first seed component identifies the experiment; the second
    identifies the caller-assigned training or validation batch stream.
    Callers must keep training and validation stream identifiers distinct.
    """
    x, y, bars, T = _validate_inputs(
        images=images,
        labels=labels,
        schedule=schedule,
        seed=seed,
    )

    seed_tensor = tf.constant(seed, dtype=tf.int32)

    timestep_seed = tf.random.experimental.stateless_fold_in(
        seed_tensor, tf.constant(0, dtype=tf.int32)
    )
    noise_seed = tf.random.experimental.stateless_fold_in(
        seed_tensor, tf.constant(1, dtype=tf.int32)
    )

    t = tf.random.stateless_uniform(
        shape=(len(x),),
        seed=timestep_seed,
        minval=0,
        maxval=T,
        dtype=tf.int32,
    )

    noise = tf.random.stateless_normal(
        shape=x.shape,
        seed=noise_seed,
        dtype=tf.float32,
    )

    bar_t = tf.gather(
        tf.convert_to_tensor(bars, dtype=tf.float32),
        t,
    )
    sqrt_bar = tf.reshape(tf.sqrt(bar_t), (-1, 1, 1, 1))
    sqrt_rest = tf.reshape(
        tf.sqrt(1.0 - bar_t), (-1, 1, 1, 1)
    )

    noisy = (
        sqrt_bar * tf.convert_to_tensor(x)
        + sqrt_rest * noise
    )
    tf.debugging.assert_all_finite(noisy, "nonfinite noised images")

    return {
        "noisy_images": noisy,
        "class_labels": tf.convert_to_tensor(y),
        "timesteps": t,
        "noise_target": noise,
    }


def diffusion_noise_loss(*, model, batch, training):
    """Evaluate the standard epsilon-prediction mean squared error."""
    if type(training) is not bool:
        raise TypeError("training must be boolean")
    if not callable(model):
        raise TypeError("model must be callable")

    prediction = tf.convert_to_tensor(
        model(
            [
                batch["noisy_images"],
                batch["class_labels"],
                batch["timesteps"],
            ],
            training=training,
        )
    )

    target = batch["noise_target"]

    if prediction.dtype != tf.float32:
        raise TypeError("noise prediction must be float32")
    if prediction.shape != target.shape:
        raise ValueError("noise prediction shape mismatch")

    tf.debugging.assert_all_finite(
        prediction, "nonfinite noise prediction"
    )

    loss = tf.reduce_mean(tf.square(target - prediction))
    tf.debugging.assert_all_finite(
        loss, "nonfinite DDPM noise loss"
    )

    return loss


def diffusion_optimizer_step(
    *, model, optimizer, images, labels, schedule, seed
):
    """Execute exactly one governed DDPM optimizer update."""
    if not callable(getattr(optimizer, "apply_gradients", None)):
        raise TypeError("optimizer must support apply_gradients")

    batch = stateless_ddpm_batch(
        images=images,
        labels=labels,
        schedule=schedule,
        seed=seed,
    )

    with tf.GradientTape() as tape:
        loss = diffusion_noise_loss(
            model=model, batch=batch, training=True
        )

    variables = list(model.trainable_variables)
    if not variables:
        raise RuntimeError("model has no trainable variables")

    gradients = tape.gradient(loss, variables)
    if any(gradient is None for gradient in gradients):
        raise RuntimeError("missing model gradient")

    for gradient in gradients:
        tf.debugging.assert_all_finite(
            gradient, "nonfinite DDPM model gradient"
        )

    optimizer.apply_gradients(zip(gradients, variables))
    return float(loss.numpy())


def diffusion_validation_step(
    *, model, images, labels, schedule, seed
):
    """Compute forward-only validation MSE without optimizer changes."""
    batch = stateless_ddpm_batch(
        images=images,
        labels=labels,
        schedule=schedule,
        seed=seed,
    )
    loss = diffusion_noise_loss(
        model=model, batch=batch, training=False
    )
    return float(loss.numpy())


__all__ = [
    "stateless_ddpm_batch",
    "diffusion_noise_loss",
    "diffusion_optimizer_step",
    "diffusion_validation_step",
]
