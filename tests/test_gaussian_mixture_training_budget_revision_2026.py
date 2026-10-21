"""Tests for Paper 1 revision-2026 GMM training-budget schedule."""

from __future__ import annotations

import pytest

from trustforge.models.paper01_revision_2026.gaussian_mixture_training_budget import (
    MAXIMUM_PASS_MULTIPLIER,
    MINIMUM_UPDATE_BUDGET,
    MINIMUM_VALIDATION_INTERVAL_UPDATES,
    POST_UPDATE_VALIDATION_EVALUATIONS,
    TOTAL_VALIDATION_CHECKPOINT_OBSERVATIONS,
    is_validation_step,
    training_budget,
)


def test_governed_constants():
    assert MINIMUM_UPDATE_BUDGET == 100
    assert MAXIMUM_PASS_MULTIPLIER == 5
    assert MINIMUM_VALIDATION_INTERVAL_UPDATES == 20
    assert POST_UPDATE_VALIDATION_EVALUATIONS == 5
    assert TOTAL_VALIDATION_CHECKPOINT_OBSERVATIONS == 6


@pytest.mark.parametrize(
    (
        "class_count",
        "expected_batches",
        "expected_updates",
        "expected_interval",
        "expected_steps",
    ),
    [
        (
            79,
            1,
            100,
            20,
            (
                0,
                20,
                40,
                60,
                80,
                100,
            ),
        ),
        (
            106,
            1,
            100,
            20,
            (
                0,
                20,
                40,
                60,
                80,
                100,
            ),
        ),
        (
            206,
            2,
            100,
            20,
            (
                0,
                20,
                40,
                60,
                80,
                100,
            ),
        ),
        (
            877,
            7,
            100,
            20,
            (
                0,
                20,
                40,
                60,
                80,
                100,
            ),
        ),
        (
            8100,
            64,
            320,
            64,
            (
                0,
                64,
                128,
                192,
                256,
                320,
            ),
        ),
        (
            12221,
            96,
            480,
            96,
            (
                0,
                96,
                192,
                288,
                384,
                480,
            ),
        ),
        (
            125794,
            983,
            4915,
            983,
            (
                0,
                983,
                1966,
                2949,
                3932,
                4915,
            ),
        ),
        (
            605255,
            4729,
            23645,
            4729,
            (
                0,
                4729,
                9458,
                14187,
                18916,
                23645,
            ),
        ),
        (
            619118,
            4837,
            24185,
            4837,
            (
                0,
                4837,
                9674,
                14511,
                19348,
                24185,
            ),
        ),
    ],
)
def test_representative_training_budgets(
    class_count,
    expected_batches,
    expected_updates,
    expected_interval,
    expected_steps,
):
    budget = training_budget(
        class_count
    )

    assert budget.class_count == class_count
    assert budget.batches_per_pass == expected_batches
    assert budget.max_updates == expected_updates
    assert budget.validation_interval == expected_interval
    assert budget.validation_steps == expected_steps


@pytest.mark.parametrize(
    "class_count",
    [
        1,
        79,
        106,
        206,
        877,
        8100,
        12221,
        125794,
        605255,
        619118,
    ],
)
def test_every_budget_has_exact_six_checkpoint_observations(
    class_count,
):
    budget = training_budget(
        class_count
    )

    assert len(
        budget.validation_steps
    ) == 6

    assert budget.validation_steps[
        0
    ] == 0

    assert budget.validation_steps[
        -1
    ] == budget.max_updates

    assert len(
        set(
            budget.validation_steps
        )
    ) == 6


def test_small_class_branch_has_exact_five_positive_validations():
    budget = training_budget(
        79
    )

    assert budget.validation_steps[
        1:
    ] == (
        20,
        40,
        60,
        80,
        100,
    )


def test_large_class_branch_has_exact_five_positive_validations():
    budget = training_budget(
        8100
    )

    assert budget.validation_steps[
        1:
    ] == tuple(
        budget.batches_per_pass
        * multiplier
        for multiplier in range(
            1,
            6,
        )
    )


def test_validation_step_membership():
    budget = training_budget(
        8100
    )

    assert is_validation_step(
        0,
        budget=budget,
    )

    assert is_validation_step(
        64,
        budget=budget,
    )

    assert is_validation_step(
        320,
        budget=budget,
    )

    assert not is_validation_step(
        1,
        budget=budget,
    )

    assert not is_validation_step(
        319,
        budget=budget,
    )


@pytest.mark.parametrize(
    "step",
    [
        -1,
        321,
    ],
)
def test_invalid_validation_step_query_fails_closed(
    step,
):
    budget = training_budget(
        8100
    )

    with pytest.raises(
        ValueError
    ):
        is_validation_step(
            step,
            budget=budget,
        )


@pytest.mark.parametrize(
    "class_count",
    [
        0,
        -1,
    ],
)
def test_invalid_class_count_fails_closed(
    class_count,
):
    with pytest.raises(
        ValueError
    ):
        training_budget(
            class_count
        )
