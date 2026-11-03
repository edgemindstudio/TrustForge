"""Bounded, terminal-only descriptive observations for Paper 1 GAN.

These observations cannot select checkpoints, change optimization,
measure convergence, or authorize training.
"""

import numpy as np


IMAGE_SHAPE = (40, 40, 3)
LATENT_DIM = 100
MAX_VALIDATION_EXAMPLES = 16
DIAGNOSTIC_SEED_OFFSET = 104729


def observe_gan_terminal_diagnostics(
    *,
    generator,
    dataset,
    experimental_seed,
    class_count,
):
    """Inspect fixed-seed generation and a bounded validation batch.

    Call only after successful terminal training accounting.
    Does not modify weights, optimizers, or training random streams.
    """
    if type(experimental_seed) is not int or not (
        0 <= experimental_seed < 2**32
    ):
        raise ValueError("invalid diagnostic experimental seed")

    if type(class_count) is not int or not (1 <= class_count <= 47):
        raise ValueError("invalid diagnostic class count")

    # Independent NumPy stream: no TensorFlow RNG reset.
    rng = np.random.default_rng(
        np.random.SeedSequence(
            [experimental_seed, DIAGNOSTIC_SEED_OFFSET]
        )
    )

    class_ids = np.arange(class_count, dtype=np.int32)
    labels = np.eye(class_count, dtype=np.float32)[class_ids]
    noise = rng.standard_normal(
        (class_count, LATENT_DIM)
    ).astype(np.float32)

    generated = generator(
        [noise, labels],
        training=False,
    )

    if hasattr(generated, "numpy"):
        generated = generated.numpy()

    generated = np.asarray(generated)

    if generated.shape != (class_count, *IMAGE_SHAPE):
        raise ValueError("invalid diagnostic generation shape")
    if generated.dtype != np.float32:
        raise ValueError("invalid diagnostic generation dtype")
    if not np.isfinite(generated).all():
        raise ValueError("nonfinite diagnostic generation")

    # The GAN generator's native domain is [-1, 1].
    if np.any(generated < -1.00001) or np.any(
        generated > 1.00001
    ):
        raise ValueError("diagnostic generation outside GAN domain")

    canonical_generated = (generated + 1.0) / 2.0

    if np.any(canonical_generated < -0.00001) or np.any(
        canonical_generated > 1.00001
    ):
        raise ValueError("invalid canonical diagnostic generation")

    validation_size = dataset.split_size("validation")

    if type(validation_size) is not int or validation_size <= 0:
        raise ValueError("invalid governed validation split size")

    count = min(validation_size, MAX_VALIDATION_EXAMPLES)
    positions = np.arange(count, dtype=np.int64)

    validation_images, validation_labels = dataset.canonical_batch(
        "validation",
        positions,
    )

    validation_images = np.asarray(validation_images)
    validation_labels = np.asarray(validation_labels)

    if validation_images.shape != (count, *IMAGE_SHAPE):
        raise ValueError("invalid validation image shape")
    if validation_images.dtype != np.float32:
        raise ValueError("invalid validation image dtype")
    if not np.isfinite(validation_images).all():
        raise ValueError("nonfinite validation images")
    if np.any(validation_images < 0) or np.any(
        validation_images > 1
    ):
        raise ValueError("validation images outside canonical domain")

    if validation_labels.shape != (count, class_count):
        raise ValueError("invalid validation label shape")
    if validation_labels.dtype != np.float32:
        raise ValueError("invalid validation label dtype")
    if not np.isfinite(validation_labels).all():
        raise ValueError("nonfinite validation labels")
    if not np.all(
        (validation_labels == 0.0)
        | (validation_labels == 1.0)
    ):
        raise ValueError("invalid validation one-hot entries")
    if not np.all(validation_labels.sum(axis=1) == 1.0):
        raise ValueError("invalid validation one-hot sums")

    return {
        "diagnostic_type": "terminal_descriptive_nonselecting",
        "diagnostic_seed_offset": DIAGNOSTIC_SEED_OFFSET,
        "generated_samples_per_class": 1,
        "generated_sample_count": class_count,
        "generated_canonical_min": float(
            canonical_generated.min()
        ),
        "generated_canonical_max": float(
            canonical_generated.max()
        ),
        "generated_canonical_mean": float(
            canonical_generated.mean(dtype=np.float64)
        ),
        "validation_split": "validation",
        "validation_position_policy": "first_bounded_positions",
        "validation_example_count": count,
        "validation_canonical_min": float(
            validation_images.min()
        ),
        "validation_canonical_max": float(
            validation_images.max()
        ),
        "validation_canonical_mean": float(
            validation_images.mean(dtype=np.float64)
        ),
        "validation_label_ids": [
            int(x) for x in np.argmax(
                validation_labels, axis=1
            )
        ],
        "checkpoint_selection_influenced": False,
        "convergence_claimed": False,
        "semantic_class_fidelity_claimed": False,
    }
