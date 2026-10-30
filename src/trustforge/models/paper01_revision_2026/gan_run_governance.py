"""Run governance for Paper 1 revision-2026 GAN feasibility.

Planning and accounting only. This module cannot execute training.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import operator

from trustforge.models.paper01_revision_2026.gan_checkpoint_manifest import (
    DATASET_CLASS_COUNTS,
)


BATCH_SIZE = 256
MIN_ITERATIONS = 500
MAX_ITERATIONS = 10000
NOMINAL_PASSES = 2


def exact_integer(value, name, minimum=0):
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return int(result)


@dataclass(frozen=True)
class GANRunPlan:
    dataset_id: str
    experimental_seed: int
    class_mapping: tuple[str, ...]
    train_size: int
    batch_size: int
    steps_per_pass: int
    planned_generator_iterations: int
    planned_discriminator_updates: int

    @property
    def nominal_real_exposures(self):
        return self.planned_generator_iterations * self.batch_size

    @property
    def nominal_passes(self):
        return self.nominal_real_exposures / self.train_size


def plan_gan_run(
    *,
    dataset_id,
    experimental_seed,
    class_mapping,
    train_size,
):
    """Establish the provisional budget before any model construction."""
    if dataset_id not in DATASET_CLASS_COUNTS:
        raise ValueError("unsupported governed dataset")

    seed = exact_integer(experimental_seed, "experimental_seed")
    if seed > 2**32 - 1:
        raise ValueError("experimental_seed exceeds uint32 range")

    size = exact_integer(train_size, "train_size", 1)

    if isinstance(class_mapping, (str, bytes)):
        raise TypeError("class_mapping must be an ordered label sequence")

    labels = tuple(class_mapping)

    if (
        len(labels) != DATASET_CLASS_COUNTS[dataset_id]
        or any(
            not isinstance(label, str)
            or not label
            or label != label.strip()
            for label in labels
        )
        or len(set(labels)) != len(labels)
    ):
        raise ValueError("invalid authoritative class mapping")

    steps = math.ceil(size / BATCH_SIZE)
    iterations = min(
        MAX_ITERATIONS,
        max(MIN_ITERATIONS, NOMINAL_PASSES * steps),
    )

    return GANRunPlan(
        dataset_id=dataset_id,
        experimental_seed=seed,
        class_mapping=labels,
        train_size=size,
        batch_size=BATCH_SIZE,
        steps_per_pass=steps,
        planned_generator_iterations=iterations,
        planned_discriminator_updates=2 * iterations,
    )


@dataclass
class GANRunAccounting:
    """Count only completely successful adversarial iterations."""

    plan: GANRunPlan
    generator_iterations: int = 0
    discriminator_updates: int = 0

    def record_completed_iteration(self):
        if self.generator_iterations >= self.plan.planned_generator_iterations:
            raise ValueError("predeclared generator budget exhausted")

        self.generator_iterations += 1
        self.discriminator_updates += 2

    def require_terminal(self):
        if self.generator_iterations != self.plan.planned_generator_iterations:
            raise ValueError("incomplete generator budget")
        if self.discriminator_updates != self.plan.planned_discriminator_updates:
            raise ValueError("incomplete discriminator update accounting")
