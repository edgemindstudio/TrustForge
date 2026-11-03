"""Isolated tests for governed class-conditional diffusion sampling."""

import numpy as np
import pytest

from trustforge.models.paper01_revision_2026.diffusion_ddpm_math import (
    linear_ddpm_schedule,
)
from trustforge.models.paper01_revision_2026.diffusion_governed_sampling import (
    sample_governed_diffusion,
)


class ZeroNoiseModel:
    def __init__(self):
        self.calls = []

    def __call__(self, inputs, training=False):
        x, labels, times = inputs
        assert training is False
        assert x.shape == (1, 4, 4, 1)
        assert labels.shape == (1, 3)
        assert times.shape == (1,)
        assert np.sum(labels) == 1
        self.calls.append(
            (int(np.argmax(labels)), int(times[0]))
        )
        return np.zeros_like(x)


def arguments():
    return {
        "num_classes": 3,
        "image_shape": (4, 4, 1),
        "experimental_seed": 42,
        "schedule": linear_ddpm_schedule(timesteps=4),
    }


def test_sampling_is_class_conditional_and_finite():
    model = ZeroNoiseModel()
    samples = sample_governed_diffusion(
        model=model,
        class_ids=[0, 2],
        sample_indices=[0, 0],
        **arguments(),
    )

    assert samples.shape == (2, 4, 4, 1)
    assert samples.dtype == np.float32
    assert np.all(np.isfinite(samples))
    assert np.all((samples >= 0) & (samples <= 1))
    assert model.calls[:4] == [
        (0, 3), (0, 2), (0, 1), (0, 0)
    ]
    assert model.calls[4:] == [
        (2, 3), (2, 2), (2, 1), (2, 0)
    ]


def test_repeatability_and_order_independence():
    a = sample_governed_diffusion(
        model=ZeroNoiseModel(),
        class_ids=[0, 1],
        sample_indices=[7, 9],
        **arguments(),
    )
    b = sample_governed_diffusion(
        model=ZeroNoiseModel(),
        class_ids=[1, 0],
        sample_indices=[9, 7],
        **arguments(),
    )
    np.testing.assert_array_equal(a[0], b[1])
    np.testing.assert_array_equal(a[1], b[0])


def test_sample_identity_changes_noise_stream():
    a = sample_governed_diffusion(
        model=ZeroNoiseModel(),
        class_ids=[1],
        sample_indices=[0],
        **arguments(),
    )
    b = sample_governed_diffusion(
        model=ZeroNoiseModel(),
        class_ids=[1],
        sample_indices=[1],
        **arguments(),
    )
    assert not np.array_equal(a, b)


def test_duplicate_identity_is_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        sample_governed_diffusion(
            model=ZeroNoiseModel(),
            class_ids=[0, 0],
            sample_indices=[1, 1],
            **arguments(),
        )


def test_invalid_class_is_rejected():
    with pytest.raises(ValueError, match="class_id"):
        sample_governed_diffusion(
            model=ZeroNoiseModel(),
            class_ids=[3],
            sample_indices=[0],
            **arguments(),
        )


def test_nonfinite_model_prediction_is_rejected():
    class BadModel:
        def __call__(self, inputs, training=False):
            return np.full_like(inputs[0], np.nan)

    with pytest.raises(ValueError, match="nonfinite"):
        sample_governed_diffusion(
            model=BadModel(),
            class_ids=[0],
            sample_indices=[0],
            **arguments(),
        )


def test_wrong_prediction_shape_is_rejected():
    class BadShape:
        def __call__(self, inputs, training=False):
            return np.zeros((2, 4, 4, 1), dtype=np.float32)

    with pytest.raises(ValueError, match="shape mismatch"):
        sample_governed_diffusion(
            model=BadShape(),
            class_ids=[0],
            sample_indices=[0],
            **arguments(),
        )
