"""Tests for the Paper 1 revision-2026 in-memory GMM class trainer."""

from __future__ import annotations

import numpy as np
import pytest

import trustforge.models.paper01_revision_2026.gaussian_mixture_training as training_module
from trustforge.models.paper01_revision_2026.gaussian_mixture_optimizer import (
    GMMOptimizerStepResult,
)
from trustforge.models.paper01_revision_2026.gaussian_mixture_validation import (
    GMMValidationResult,
)


CLASS_ID = 0
NUM_CLASSES = 2


class _SyntheticAdapter:
    def __init__(self) -> None:
        self.images = np.zeros(
            (
                10,
                40,
                40,
                3,
            ),
            dtype=np.float32,
        )

        # Five distinct governed training vectors.
        for index in range(
            5
        ):
            self.images[
                index,
                0,
                0,
                0,
            ] = np.float32(
                (
                    index
                    + 1
                )
                * 1e-4
            )

        # Validation vectors.
        for index in range(
            5,
            10,
        ):
            self.images[
                index
            ].fill(
                np.float32(
                    0.25
                    + (
                        index
                        - 5
                    )
                    * 0.05
                )
            )

        self.labels = np.zeros(
            (
                10,
                NUM_CLASSES,
            ),
            dtype=np.float32,
        )

        self.labels[
            :,
            CLASS_ID,
        ] = 1.0

        self.calls: list[
            tuple[
                str,
                np.ndarray,
            ]
        ] = []

    def canonical_batch(
        self,
        split: str,
        logical_positions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        positions = np.asarray(
            logical_positions,
            dtype=np.int64,
        )

        self.calls.append(
            (
                split,
                positions.copy(),
            )
        )

        return (
            self.images[
                positions
            ],
            self.labels[
                positions
            ],
        )


def _install_fast_optimizer(
    monkeypatch,
):
    step_counter = {
        "value": 0,
    }

    def fake_sgd_train_step(
        model,
        images,
    ):
        del images

        step_counter[
            "value"
        ] += 1

        model.means.assign_add(
            np.ones(
                model.means.shape,
                dtype=np.float64,
            )
        )

        return GMMOptimizerStepResult(
            loss=float(
                step_counter[
                    "value"
                ]
            ),
            gradient_global_norm=1.0,
        )

    monkeypatch.setattr(
        training_module,
        "sgd_train_step",
        fake_sgd_train_step,
    )

    return step_counter


def _install_validation_sequence(
    monkeypatch,
    sequence,
):
    values = iter(
        sequence
    )

    calls = {
        "count": 0,
    }

    def fake_validation(
        model,
        *,
        adapter,
        logical_positions,
        class_id,
    ):
        del model
        del adapter
        del class_id

        calls[
            "count"
        ] += 1

        return GMMValidationResult(
            mean_nll=float(
                next(
                    values
                )
            ),
            example_count=int(
                len(
                    logical_positions
                )
            ),
            batch_count=1,
        )

    monkeypatch.setattr(
        training_module,
        "evaluate_full_class_validation_nll",
        fake_validation,
    )

    return calls


def test_training_runs_exact_governed_budget_and_schedule(
    monkeypatch,
):
    adapter = _SyntheticAdapter()

    step_counter = _install_fast_optimizer(
        monkeypatch
    )

    validation_calls = _install_validation_sequence(
        monkeypatch,
        [
            10.0,
            9.0,
            8.0,
            7.0,
            6.0,
            5.0,
        ],
    )

    result = training_module.train_class_model(
        adapter=adapter,
        class_id=CLASS_ID,
        experimental_seed=42,
        train_class_positions=np.arange(
            5,
            dtype=np.int64,
        ),
        validation_class_positions=np.arange(
            5,
            10,
            dtype=np.int64,
        ),
    )

    assert result.budget.max_updates == 100

    assert result.budget.validation_steps == (
        0,
        20,
        40,
        60,
        80,
        100,
    )

    assert result.optimizer_steps_completed == 100

    assert step_counter[
        "value"
    ] == 100

    assert validation_calls[
        "count"
    ] == 6

    assert tuple(
        item.step
        for item in result.validation_history
    ) == result.budget.validation_steps

    assert result.selected_step == 100
    assert result.selected_validation_nll == 5.0


def test_earliest_step_wins_exact_validation_tie(
    monkeypatch,
):
    adapter = _SyntheticAdapter()

    _install_fast_optimizer(
        monkeypatch
    )

    _install_validation_sequence(
        monkeypatch,
        [
            10.0,
            8.0,
            8.0,
            9.0,
            7.0,
            7.0,
        ],
    )

    result = training_module.train_class_model(
        adapter=adapter,
        class_id=CLASS_ID,
        experimental_seed=42,
        train_class_positions=np.arange(
            5,
            dtype=np.int64,
        ),
        validation_class_positions=np.arange(
            5,
            10,
            dtype=np.int64,
        ),
    )

    assert result.selected_step == 80
    assert result.selected_validation_nll == 7.0


def test_returned_model_is_selected_checkpoint_not_final_state(
    monkeypatch,
):
    adapter = _SyntheticAdapter()

    _install_fast_optimizer(
        monkeypatch
    )

    _install_validation_sequence(
        monkeypatch,
        [
            10.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
        ],
    )

    result = training_module.train_class_model(
        adapter=adapter,
        class_id=CLASS_ID,
        experimental_seed=42,
        train_class_positions=np.arange(
            5,
            dtype=np.int64,
        ),
        validation_class_positions=np.arange(
            5,
            10,
            dtype=np.int64,
        ),
    )

    # Step 20 is selected even though optimization continued through step 100.
    assert result.selected_step == 20

    selected_means = result.model.state_arrays()[
        "means"
    ]

    # Fake optimizer adds exactly one to every mean per update.
    # Therefore the selected state must be initialization + 20, not +100.
    assert np.all(
        selected_means
        < 30.0
    )

    assert np.all(
        selected_means
        > 19.0
    )


def test_initialization_uses_governed_class_relative_positions(
    monkeypatch,
):
    adapter = _SyntheticAdapter()

    _install_fast_optimizer(
        monkeypatch
    )

    _install_validation_sequence(
        monkeypatch,
        [
            10.0,
            9.0,
            8.0,
            7.0,
            6.0,
            5.0,
        ],
    )

    train_positions = np.array(
        [
            4,
            3,
            2,
            1,
            0,
        ],
        dtype=np.int64,
    )

    training_module.train_class_model(
        adapter=adapter,
        class_id=CLASS_ID,
        experimental_seed=42,
        train_class_positions=train_positions,
        validation_class_positions=np.arange(
            5,
            10,
            dtype=np.int64,
        ),
    )

    first_split, first_positions = adapter.calls[
        0
    ]

    assert first_split == "train"

    np.testing.assert_array_equal(
        first_positions,
        train_positions,
    )


def test_wrong_training_class_membership_fails_closed(
    monkeypatch,
):
    adapter = _SyntheticAdapter()

    adapter.labels[
        2,
        :,
    ] = 0.0

    adapter.labels[
        2,
        1,
    ] = 1.0

    with pytest.raises(
        ValueError,
        match="outside governed class",
    ):
        training_module.train_class_model(
            adapter=adapter,
            class_id=CLASS_ID,
            experimental_seed=42,
            train_class_positions=np.arange(
                5,
                dtype=np.int64,
            ),
            validation_class_positions=np.arange(
                5,
                10,
                dtype=np.int64,
            ),
        )


def test_duplicate_train_positions_fail_closed():
    adapter = _SyntheticAdapter()

    with pytest.raises(
        ValueError,
        match="duplicates",
    ):
        training_module.train_class_model(
            adapter=adapter,
            class_id=CLASS_ID,
            experimental_seed=42,
            train_class_positions=np.array(
                [
                    0,
                    1,
                    2,
                    2,
                    4,
                ],
                dtype=np.int64,
            ),
            validation_class_positions=np.arange(
                5,
                10,
                dtype=np.int64,
            ),
        )


def test_duplicate_validation_positions_fail_closed():
    adapter = _SyntheticAdapter()

    with pytest.raises(
        ValueError,
        match="duplicates",
    ):
        training_module.train_class_model(
            adapter=adapter,
            class_id=CLASS_ID,
            experimental_seed=42,
            train_class_positions=np.arange(
                5,
                dtype=np.int64,
            ),
            validation_class_positions=np.array(
                [
                    5,
                    6,
                    7,
                    7,
                    9,
                ],
                dtype=np.int64,
            ),
        )


@pytest.mark.parametrize(
    "class_id",
    [
        0.0,
        "0",
        True,
    ],
)
def test_class_id_requires_exact_integer(
    class_id,
):
    adapter = _SyntheticAdapter()

    with pytest.raises(
        TypeError
    ):
        training_module.train_class_model(
            adapter=adapter,
            class_id=class_id,
            experimental_seed=42,
            train_class_positions=np.arange(
                5,
                dtype=np.int64,
            ),
            validation_class_positions=np.arange(
                5,
                10,
                dtype=np.int64,
            ),
        )
