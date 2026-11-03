"""Synthetic-only governed DDPM training primitive tests."""

import numpy as np
import pytest
import tensorflow as tf

from trustforge.models.paper01_revision_2026.diffusion_ddpm_math import (
    linear_ddpm_schedule,
)
from trustforge.models.paper01_revision_2026.diffusion_training_step import (
    stateless_ddpm_batch,
    diffusion_optimizer_step,
    diffusion_validation_step,
)


class TinyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv = tf.keras.layers.Conv2D(
            1, 1, use_bias=False, kernel_initializer="ones"
        )

    def call(self, inputs, training=False):
        x, labels, timesteps = inputs
        extra = (
            tf.reduce_sum(labels, axis=1)
            + 0.0 * tf.cast(timesteps, tf.float32)
        )
        return self.conv(x) + tf.reshape(extra * 0.0, (-1, 1, 1, 1))


def fixture():
    x = np.full((3, 4, 4, 1), 0.25, dtype=np.float32)
    y = np.eye(3, dtype=np.float32)
    schedule = linear_ddpm_schedule(timesteps=8)
    return x, y, schedule


def test_stateless_batch_repeats_exactly():
    x, y, schedule = fixture()
    a = stateless_ddpm_batch(
        images=x, labels=y, schedule=schedule, seed=(42, 0)
    )
    b = stateless_ddpm_batch(
        images=x, labels=y, schedule=schedule, seed=(42, 0)
    )
    for key in a:
        np.testing.assert_array_equal(
            a[key].numpy(), b[key].numpy()
        )


def test_training_and_validation_streams_differ():
    x, y, schedule = fixture()
    train = stateless_ddpm_batch(
        images=x, labels=y, schedule=schedule, seed=(42, 0)
    )
    val = stateless_ddpm_batch(
        images=x, labels=y, schedule=schedule, seed=(42, -1)
    )
    assert not np.array_equal(
        train["noise_target"].numpy(),
        val["noise_target"].numpy(),
    )


def test_optimizer_performs_one_finite_update():
    x, y, schedule = fixture()
    model = TinyModel()
    model([
        tf.convert_to_tensor(x),
        tf.convert_to_tensor(y),
        tf.zeros((3,), tf.int32),
    ])
    before = [v.numpy().copy() for v in model.trainable_variables]
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

    loss = diffusion_optimizer_step(
        model=model,
        optimizer=optimizer,
        images=x,
        labels=y,
        schedule=schedule,
        seed=(42, 0),
    )

    assert np.isfinite(loss)
    assert int(optimizer.iterations.numpy()) == 1
    assert any(
        not np.array_equal(old, new.numpy())
        for old, new in zip(
            before, model.trainable_variables, strict=True
        )
    )


def test_validation_does_not_update_weights():
    x, y, schedule = fixture()
    model = TinyModel()
    model([
        tf.convert_to_tensor(x),
        tf.convert_to_tensor(y),
        tf.zeros((3,), tf.int32),
    ])
    before = [v.numpy().copy() for v in model.trainable_variables]

    first = diffusion_validation_step(
        model=model, images=x, labels=y,
        schedule=schedule, seed=(42, -1),
    )
    second = diffusion_validation_step(
        model=model, images=x, labels=y,
        schedule=schedule, seed=(42, -1),
    )

    assert np.isfinite(first)
    assert first == second

    for old, new in zip(
        before, model.trainable_variables, strict=True
    ):
        np.testing.assert_array_equal(old, new.numpy())


def test_invalid_labels_rejected_before_training():
    x, y, schedule = fixture()
    y[0] = 0.0
    with pytest.raises(ValueError, match="one active class"):
        stateless_ddpm_batch(
            images=x, labels=y,
            schedule=schedule, seed=(42, 0),
        )


def test_invalid_canonical_range_rejected():
    x, y, schedule = fixture()
    x[0, 0, 0, 0] = -0.1
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        stateless_ddpm_batch(
            images=x, labels=y,
            schedule=schedule, seed=(42, 0),
        )


def test_inconsistent_schedule_rejected():
    x, y, schedule = fixture()
    bad = dict(schedule)
    bad["alpha_bars"] = schedule["alpha_bars"].copy()
    bad["alpha_bars"][2] *= 0.5

    with pytest.raises(ValueError, match="cumulative schedule"):
        stateless_ddpm_batch(
            images=x, labels=y,
            schedule=bad, seed=(42, 0),
        )
