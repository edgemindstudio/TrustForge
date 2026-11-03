"""Isolated class-conditional DDPM sampling mechanics.

Development and engineering verification only.

No checkpoint loading, provenance assertion, artifact publication,
production authorization, or global RNG mutation occurs here.

Sample noise is derived from dataset-independent numerical identity.
The governed caller must independently authenticate the dataset,
checkpoint, experiment identity, and class mapping.
"""

from __future__ import annotations

import numpy as np

from .diffusion_ddpm_math import ddpm_reverse_step


def _require_identity(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _normal(*, seed, class_id, sample_index, step, stream, shape):
    identity = [
        seed,
        class_id,
        sample_index,
        step,
        stream,
    ]
    rng = np.random.default_rng(np.random.SeedSequence(identity))
    return rng.standard_normal(shape).astype(np.float32)


def sample_governed_diffusion(
    *,
    model,
    class_ids,
    sample_indices,
    num_classes,
    image_shape,
    experimental_seed,
    schedule,
):
    """Return samples as float32 NHWC images in [0, 1].

    This follows the historical training input convention [0, 1].
    Finite output values are bounded at the final image boundary;
    nonfinite intermediate values always raise.

    Each sample is generated independently of batch/chunk order.
    """
    if not callable(model):
        raise TypeError("model must be callable")

    if type(num_classes) is not int or num_classes < 2:
        raise ValueError("num_classes must be an integer >= 2")

    if (
        not isinstance(image_shape, (tuple, list))
        or len(image_shape) != 3
        or any(type(n) is not int or n <= 0 for n in image_shape)
    ):
        raise ValueError("invalid image shape")

    height, width, channels = image_shape

    if height % 4 or width % 4:
        raise ValueError(
            "historical depth-two U-Net requires dimensions divisible by 4"
        )

    seed = _require_identity(experimental_seed, "experimental_seed")

    if not isinstance(class_ids, (tuple, list)):
        raise TypeError("class_ids must be a list or tuple")
    if not isinstance(sample_indices, (tuple, list)):
        raise TypeError("sample_indices must be a list or tuple")
    if not class_ids or len(class_ids) != len(sample_indices):
        raise ValueError("class and sample identities must align")

    if not isinstance(schedule, dict):
        raise TypeError("schedule must be a mapping")

    timesteps = schedule.get("timesteps")
    if type(timesteps) is not int or timesteps < 2:
        raise ValueError("invalid diffusion schedule")

    identities = []

    for class_id, sample_index in zip(
        class_ids, sample_indices, strict=True
    ):
        class_id = _require_identity(class_id, "class_id")
        sample_index = _require_identity(sample_index, "sample_index")

        if class_id >= num_classes:
            raise ValueError("class_id outside class mapping")

        identity = (class_id, sample_index)
        if identity in identities:
            raise ValueError("duplicate sample identity")
        identities.append(identity)

    images = []

    for class_id, sample_index in identities:
        shape = (1, height, width, channels)

        x = _normal(
            seed=seed,
            class_id=class_id,
            sample_index=sample_index,
            step=timesteps,
            stream=0,
            shape=shape,
        )

        label = np.zeros((1, num_classes), dtype=np.float32)
        label[0, class_id] = 1.0

        for t in reversed(range(timesteps)):
            time = np.array([t], dtype=np.int32)

            predicted = np.asarray(
                model([x, label, time], training=False)
            )

            if predicted.shape != x.shape:
                raise ValueError("model noise-prediction shape mismatch")
            if predicted.dtype != np.dtype("float32"):
                raise TypeError(
                    "model noise prediction must be float32"
                )
            if not np.all(np.isfinite(predicted)):
                raise ValueError("nonfinite model noise prediction")

            transition_noise = _normal(
                seed=seed,
                class_id=class_id,
                sample_index=sample_index,
                step=t,
                stream=1,
                shape=shape,
            )

            x = ddpm_reverse_step(
                x_t=x,
                predicted_noise=predicted,
                transition_noise=transition_noise,
                timestep=t,
                schedule=schedule,
            )

        if not np.all(np.isfinite(x)):
            raise ValueError("nonfinite final sample")

        images.append(np.clip(x[0], 0.0, 1.0))

    result = np.stack(images, axis=0).astype(np.float32)

    if result.shape != (
        len(identities), height, width, channels
    ):
        raise RuntimeError("unexpected sampling output shape")

    return result


__all__ = ["sample_governed_diffusion"]
