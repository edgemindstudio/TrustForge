"""Focused tests for governed revision-2026 RBM training budget."""

from __future__ import annotations

import math

import pytest

from trustforge.models.paper01_revision_2026.restricted_boltzmann_training_budget import (
    BATCH_SIZE,
    RBMTrainingBudget,
    batches_per_nominal_epoch,
    build_training_budget,
    max_updates,
    validation_steps,
)


def test_governed_batch_size_is_256():
    assert BATCH_SIZE == 256


@pytest.mark.parametrize(
    "n_train,expected_B",
    [
        (1, 1),
        (255, 1),
        (256, 1),
        (257, 2),
        (72900, 285),
        (8117, 32),
        (883416, 3451),
        (731049, 2856),
    ],
)
def test_batches_per_nominal_epoch_is_ceil_n_over_256(
    n_train,
    expected_B,
):
    assert batches_per_nominal_epoch(
        n_train
    ) == expected_B

    assert expected_B == math.ceil(
        n_train / 256
    )


@pytest.mark.parametrize(
    "n_train,expected_max_updates",
    [
        (72900, 570),
        (8117, 64),
        (883416, 6902),
        (731049, 5712),
    ],
)
def test_known_dataset_max_update_budgets(
    n_train,
    expected_max_updates,
):
    assert max_updates(
        n_train
    ) == expected_max_updates


@pytest.mark.parametrize(
    "n_train",
    [
        1,
        255,
        256,
        257,
        72900,
        8117,
        883416,
        731049,
    ],
)
def test_max_updates_is_exactly_two_times_B(
    n_train,
):
    B = batches_per_nominal_epoch(
        n_train
    )

    assert max_updates(
        n_train
    ) == 2 * B


@pytest.mark.parametrize(
    "n_train",
    [
        1,
        257,
        72900,
        8117,
        883416,
        731049,
    ],
)
def test_validation_schedule_is_exactly_zero_B_two_B(
    n_train,
):
    B = batches_per_nominal_epoch(
        n_train
    )

    assert validation_steps(
        n_train
    ) == (
        0,
        B,
        2 * B,
    )


def test_build_training_budget_records_authoritative_final_step():
    budget = build_training_budget(
        72900
    )

    assert isinstance(
        budget,
        RBMTrainingBudget,
    )

    assert budget.training_example_count == 72900
    assert budget.batch_size == 256
    assert budget.batches_per_nominal_epoch == 285
    assert budget.max_updates == 570
    assert budget.validation_steps == (
        0,
        285,
        570,
    )
    assert budget.authoritative_step == 570


@pytest.mark.parametrize(
    "n_train,expected_max_updates",
    [
        (72900, 570),
        (8117, 64),
        (883416, 6902),
        (731049, 5712),
    ],
)
def test_all_four_benchmark_dataset_budgets_match_policy(
    n_train,
    expected_max_updates,
):
    budget = build_training_budget(
        n_train
    )

    assert budget.max_updates == expected_max_updates
    assert budget.authoritative_step == expected_max_updates
    assert budget.validation_steps[-1] == expected_max_updates


@pytest.mark.parametrize(
    "bad_n_train",
    [
        0,
        -1,
        True,
        1.5,
    ],
)
def test_invalid_training_example_count_fails_closed(
    bad_n_train,
):
    with pytest.raises(
        (TypeError, ValueError)
    ):
        build_training_budget(
            bad_n_train
        )


def test_no_minimum_update_floor_exists_for_tiny_dataset():
    budget = build_training_budget(
        1
    )

    assert budget.batches_per_nominal_epoch == 1
    assert budget.max_updates == 2
    assert budget.validation_steps == (
        0,
        1,
        2,
    )
    assert budget.authoritative_step == 2


def test_validation_steps_do_not_duplicate_final_evaluation():
    budget = build_training_budget(
        257
    )

    assert budget.validation_steps == (
        0,
        2,
        4,
    )

    assert len(
        budget.validation_steps
    ) == len(
        set(
            budget.validation_steps
        )
    )


def test_authoritative_step_is_never_midpoint():
    budget = build_training_budget(
        72900
    )

    assert budget.authoritative_step != (
        budget.batches_per_nominal_epoch
    )

    assert budget.authoritative_step == (
        2
        * budget.batches_per_nominal_epoch
    )
