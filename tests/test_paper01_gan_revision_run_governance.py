"""Governed GAN feasibility budget and update accounting."""

import pytest

from trustforge.models.paper01_revision_2026.gan_run_governance import (
    GANRunAccounting,
    plan_gan_run,
)


def make_plan(size=1000):
    return plan_gan_run(
        dataset_id="malex",
        experimental_seed=42,
        class_mapping=("benign", "malware"),
        train_size=size,
    )


@pytest.mark.parametrize(
    "size,expected",
    [
        (1, 500),
        (1000, 500),
        (73000, 572),
        (883456, 6902),
        (2000000, 10000),
    ],
)
def test_provisional_budget(size, expected):
    plan = make_plan(size)
    assert plan.planned_generator_iterations == expected
    assert plan.planned_discriminator_updates == 2 * expected


def test_incomplete_run_cannot_be_terminal():
    accounting = GANRunAccounting(make_plan())
    with pytest.raises(ValueError):
        accounting.require_terminal()


def test_exact_terminal_accounting():
    plan = make_plan()
    accounting = GANRunAccounting(plan)

    for _ in range(plan.planned_generator_iterations):
        accounting.record_completed_iteration()

    accounting.require_terminal()

    with pytest.raises(ValueError):
        accounting.record_completed_iteration()


@pytest.mark.parametrize(
    "seed",
    [-1, True, 2**32, 0.5],
)
def test_invalid_seed_rejected(seed):
    with pytest.raises((TypeError, ValueError)):
        plan_gan_run(
            dataset_id="malex",
            experimental_seed=seed,
            class_mapping=("benign", "malware"),
            train_size=1000,
        )


@pytest.mark.parametrize(
    "mapping",
    [
        ("benign",),
        ("benign", "benign"),
        ("", "malware"),
        (" benign", "malware"),
    ],
)
def test_invalid_mapping_rejected(mapping):
    with pytest.raises(ValueError):
        plan_gan_run(
            dataset_id="malex",
            experimental_seed=42,
            class_mapping=mapping,
            train_size=1000,
        )


def test_invalid_training_size_rejected():
    with pytest.raises(ValueError):
        make_plan(0)


def test_accounting_does_not_execute_training():
    accounting = GANRunAccounting(make_plan())
    assert accounting.generator_iterations == 0
    assert accounting.discriminator_updates == 0
