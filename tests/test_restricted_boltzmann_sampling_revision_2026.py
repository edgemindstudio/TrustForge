"""Focused tests for revision-2026 RBM sampling primitives.

These tests cover governed initialization and one fixed-class Gibbs
transition only.  They deliberately do not select a production chain length.
"""

from __future__ import annotations

from dataclasses import replace
import inspect
from pathlib import Path

import numpy as np
import pytest

from trustforge.models.paper01_revision_2026.restricted_boltzmann_checkpoint import (
    RBMPersistedCheckpoint,
    persist_rbm_checkpoint,
)
from trustforge.models.paper01_revision_2026 import (
    restricted_boltzmann_sampling as sampling_module,
)
from trustforge.models.paper01_revision_2026.restricted_boltzmann_sampling import (
    RBMChainState,
    build_sampling_identity,
    gibbs_transition,
    initialize_chain,
)
from trustforge.models.paper01_revision_2026.restricted_boltzmann_training import (
    train_shared_conditional_rbm,
)


DATASET_ID = "tiny-rbm-sampling-test"
EXPERIMENTAL_SEED = 42
CLASS_MAPPING = (
    "class_0",
    "class_1",
)


class TinyAdapter:
    def __init__(self):
        self.train_images = np.stack(
            [
                np.full((40, 40, 3), 0.20, dtype=np.float32),
                np.full((40, 40, 3), 0.30, dtype=np.float32),
                np.full((40, 40, 3), 0.70, dtype=np.float32),
                np.full((40, 40, 3), 0.80, dtype=np.float32),
            ],
            axis=0,
        )

        self.train_labels = np.array(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )

        self.validation_images = np.stack(
            [
                np.full((40, 40, 3), 0.25, dtype=np.float32),
                np.full((40, 40, 3), 0.75, dtype=np.float32),
            ],
            axis=0,
        )

        self.validation_labels = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )

    def split_size(
        self,
        split,
    ):
        if split == "train":
            return 4

        if split == "validation":
            return 2

        raise AssertionError(
            split
        )

    def canonical_batch(
        self,
        split,
        logical_positions,
    ):
        positions = np.asarray(
            logical_positions,
            dtype=np.int64,
        )

        if split == "train":
            return (
                self.train_images[
                    positions
                ],
                self.train_labels[
                    positions
                ],
            )

        if split == "validation":
            return (
                self.validation_images[
                    positions
                ],
                self.validation_labels[
                    positions
                ],
            )

        raise AssertionError(
            split
        )


@pytest.fixture(scope="module")
def training_result():
    return train_shared_conditional_rbm(
        adapter=TinyAdapter(),
        dataset_id=DATASET_ID,
        experimental_seed=EXPERIMENTAL_SEED,
        train_class_positions={
            0: np.array(
                [0, 1],
                dtype=np.int64,
            ),
            1: np.array(
                [2, 3],
                dtype=np.int64,
            ),
        },
    )


@pytest.fixture
def checkpoint(
    tmp_path,
    training_result,
):
    return persist_rbm_checkpoint(
        tmp_path,
        dataset_id=DATASET_ID,
        experimental_seed=EXPERIMENTAL_SEED,
        class_mapping=CLASS_MAPPING,
        training_result=training_result,
    )


def test_initial_hidden_state_is_exact_zero_vector(
    checkpoint,
):
    state = initialize_chain(
        checkpoint=checkpoint,
        class_id=0,
        sample_index=7,
    )

    assert state.hidden.shape == (
        1,
        128,
    )

    assert state.hidden.dtype == np.float64

    assert np.array_equal(
        state.hidden,
        np.zeros(
            (
                1,
                128,
            ),
            dtype=np.float64,
        ),
    )


def test_initial_visible_parameter_equals_a(
    checkpoint,
):
    zero_hidden = np.zeros(
        (
            1,
            128,
        ),
        dtype=np.float64,
    )

    theta = checkpoint.model.visible_natural_parameter(
        zero_hidden
    ).numpy()

    expected = checkpoint.model.a.numpy()[
        None,
        :
    ]

    np.testing.assert_array_equal(
        theta,
        expected,
    )


def test_initialization_calls_governed_visible_sampler(
    checkpoint,
    monkeypatch,
):
    observed = []

    original = checkpoint.model.sample_visible_from_uniform

    def recording_sampler(
        hidden,
        uniform,
    ):
        observed.append(
            (
                np.asarray(
                    hidden
                ).copy(),
                np.asarray(
                    uniform
                ).copy(),
            )
        )

        return original(
            hidden,
            uniform,
        )

    monkeypatch.setattr(
        checkpoint.model,
        "sample_visible_from_uniform",
        recording_sampler,
    )

    initialize_chain(
        checkpoint=checkpoint,
        class_id=0,
        sample_index=7,
    )

    assert len(
        observed
    ) == 1

    hidden, uniform = observed[
        0
    ]

    assert np.array_equal(
        hidden,
        np.zeros(
            (
                1,
                128,
            ),
            dtype=np.float64,
        ),
    )

    assert uniform.dtype == np.float64
    assert uniform.shape == (
        1,
        4800,
    )


def test_initialization_is_float64_finite_and_bounded(
    checkpoint,
):
    state = initialize_chain(
        checkpoint=checkpoint,
        class_id=0,
        sample_index=7,
    )

    assert state.visible.dtype == np.float64
    assert state.hidden.dtype == np.float64

    assert np.isfinite(
        state.visible
    ).all()

    assert np.isfinite(
        state.hidden
    ).all()

    assert np.all(
        state.visible >= 0.0
    )

    assert np.all(
        state.visible <= 1.0
    )


def test_initialization_api_has_no_empirical_data_input():
    parameters = set(
        inspect.signature(
            initialize_chain
        ).parameters
    )

    assert parameters == {
        "checkpoint",
        "class_id",
        "sample_index",
    }

    forbidden = {
        "adapter",
        "training",
        "validation",
        "test",
        "real_example",
        "prototype",
    }

    assert parameters.isdisjoint(
        forbidden
    )


def test_same_governed_identity_reproduces_initialization(
    checkpoint,
):
    first = initialize_chain(
        checkpoint=checkpoint,
        class_id=0,
        sample_index=11,
    )

    second = initialize_chain(
        checkpoint=checkpoint,
        class_id=0,
        sample_index=11,
    )

    np.testing.assert_array_equal(
        first.visible,
        second.visible,
    )

    np.testing.assert_array_equal(
        first.hidden,
        second.hidden,
    )


def test_changed_sample_identity_can_change_initialization(
    checkpoint,
):
    first = initialize_chain(
        checkpoint=checkpoint,
        class_id=0,
        sample_index=11,
    )

    second = initialize_chain(
        checkpoint=checkpoint,
        class_id=0,
        sample_index=12,
    )

    assert not np.array_equal(
        first.visible,
        second.visible,
    )


def test_changed_class_identity_can_change_rng_path_but_not_theta0(
    checkpoint,
):
    class_zero = initialize_chain(
        checkpoint=checkpoint,
        class_id=0,
        sample_index=11,
    )

    class_one = initialize_chain(
        checkpoint=checkpoint,
        class_id=1,
        sample_index=11,
    )

    assert not np.array_equal(
        class_zero.visible,
        class_one.visible,
    )

    zero_hidden = np.zeros(
        (
            1,
            128,
        ),
        dtype=np.float64,
    )

    theta = checkpoint.model.visible_natural_parameter(
        zero_hidden
    ).numpy()

    np.testing.assert_array_equal(
        theta,
        checkpoint.model.a.numpy()[
            None,
            :
        ],
    )


def test_requested_class_is_clamped_before_first_hidden_draw(
    checkpoint,
    monkeypatch,
):
    observed_class_ids = []

    original = checkpoint.model.sample_hidden_from_uniform

    def recording_hidden_sampler(
        visible,
        class_ids,
        uniform,
    ):
        observed_class_ids.append(
            np.asarray(
                class_ids
            ).copy()
        )

        return original(
            visible,
            class_ids,
            uniform,
        )

    monkeypatch.setattr(
        checkpoint.model,
        "sample_hidden_from_uniform",
        recording_hidden_sampler,
    )

    state = initialize_chain(
        checkpoint=checkpoint,
        class_id=1,
        sample_index=5,
    )

    gibbs_transition(
        checkpoint=checkpoint,
        state=state,
        sample_index=5,
    )

    assert len(
        observed_class_ids
    ) == 1

    np.testing.assert_array_equal(
        observed_class_ids[
            0
        ],
        np.array(
            [
                1,
            ],
            dtype=np.int64,
        ),
    )


def test_first_hidden_draw_uses_requested_U_row(
    checkpoint,
):
    state_zero = initialize_chain(
        checkpoint=checkpoint,
        class_id=0,
        sample_index=19,
    )

    state_one = initialize_chain(
        checkpoint=checkpoint,
        class_id=1,
        sample_index=19,
    )

    logits_zero = checkpoint.model.hidden_logits(
        state_zero.visible,
        np.array(
            [
                0,
            ],
            dtype=np.int64,
        ),
    ).numpy()

    logits_one_for_same_visible = checkpoint.model.hidden_logits(
        state_zero.visible,
        np.array(
            [
                1,
            ],
            dtype=np.int64,
        ),
    ).numpy()

    expected_difference = (
        checkpoint.model.U.numpy()[
            1
        ]
        - checkpoint.model.U.numpy()[
            0
        ]
    )[
        None,
        :
    ]

    np.testing.assert_allclose(
        logits_one_for_same_visible
        - logits_zero,
        expected_difference,
        rtol=0.0,
        atol=1.0e-12,
    )

    assert state_one.class_id == 1


def test_class_identity_remains_fixed_across_transitions(
    checkpoint,
):
    state = initialize_chain(
        checkpoint=checkpoint,
        class_id=1,
        sample_index=3,
    )

    for expected_count in (
        1,
        2,
        3,
        4,
        5,
    ):
        state = gibbs_transition(
            checkpoint=checkpoint,
            state=state,
            sample_index=3,
        )

        assert state.class_id == 1
        assert state.transition_count == expected_count


def test_gibbs_transition_is_identity_reproducible(
    checkpoint,
):
    first = initialize_chain(
        checkpoint=checkpoint,
        class_id=0,
        sample_index=31,
    )

    second = initialize_chain(
        checkpoint=checkpoint,
        class_id=0,
        sample_index=31,
    )

    first = gibbs_transition(
        checkpoint=checkpoint,
        state=first,
        sample_index=31,
    )

    second = gibbs_transition(
        checkpoint=checkpoint,
        state=second,
        sample_index=31,
    )

    np.testing.assert_array_equal(
        first.hidden,
        second.hidden,
    )

    np.testing.assert_array_equal(
        first.visible,
        second.visible,
    )


def test_gibbs_transition_state_is_float64_finite_and_bounded(
    checkpoint,
):
    state = initialize_chain(
        checkpoint=checkpoint,
        class_id=0,
        sample_index=17,
    )

    state = gibbs_transition(
        checkpoint=checkpoint,
        state=state,
        sample_index=17,
    )

    assert state.hidden.dtype == np.float64
    assert state.visible.dtype == np.float64

    assert np.isfinite(
        state.hidden
    ).all()

    assert np.isfinite(
        state.visible
    ).all()

    assert np.all(
        state.hidden >= 0.0
    )

    assert np.all(
        state.hidden <= 1.0
    )

    assert np.all(
        state.visible >= 0.0
    )

    assert np.all(
        state.visible <= 1.0
    )


def test_invalid_class_identity_fails_closed(
    checkpoint,
):
    with pytest.raises(
        ValueError,
        match="class_id",
    ):
        initialize_chain(
            checkpoint=checkpoint,
            class_id=2,
            sample_index=0,
        )


def test_invalid_checkpoint_type_fails_closed():
    with pytest.raises(
        (
            TypeError,
            AttributeError,
        )
    ):
        initialize_chain(
            checkpoint=object(),
            class_id=0,
            sample_index=0,
        )


def test_corrupted_in_memory_model_state_fails_closed(
    checkpoint,
):
    checkpoint.model.a.assign(
        np.full(
            (
                4800,
            ),
            np.nan,
            dtype=np.float64,
        )
    )

    with pytest.raises(
        ValueError,
        match="non-finite",
    ):
        initialize_chain(
            checkpoint=checkpoint,
            class_id=0,
            sample_index=0,
        )


def test_dataset_identity_mismatch_between_metadata_and_sampling_object_fails(
    checkpoint,
):
    metadata = dict(
        checkpoint.metadata
    )

    metadata[
        "dataset_id"
    ] = "tampered-dataset"

    tampered = RBMPersistedCheckpoint(
        directory=checkpoint.directory,
        state_path=checkpoint.state_path,
        metadata_path=checkpoint.metadata_path,
        metadata=metadata,
        model=checkpoint.model,
    )

    identity = build_sampling_identity(
        dataset_id=DATASET_ID,
        experimental_seed=EXPERIMENTAL_SEED,
        class_id=0,
        sample_index=0,
        class_count=2,
    )

    with pytest.raises(
        ValueError,
        match="dataset identity",
    ):
        sampling_module._validate_checkpoint_identity(
            tampered,
            identity,
        )


def test_experimental_seed_mismatch_between_metadata_and_identity_fails(
    checkpoint,
):
    identity = build_sampling_identity(
        dataset_id=DATASET_ID,
        experimental_seed=43,
        class_id=0,
        sample_index=0,
        class_count=2,
    )

    with pytest.raises(
        ValueError,
        match="experimental seed",
    ):
        sampling_module._validate_checkpoint_identity(
            checkpoint,
            identity,
        )


def test_sampling_module_exposes_no_production_chain_length_constant():
    public_and_private_names = set(
        vars(
            sampling_module
        )
    )

    forbidden_names = {
        "BURN_IN",
        "BURN_IN_LENGTH",
        "CHAIN_LENGTH",
        "GIBBS_STEPS",
        "PRODUCTION_CHAIN_LENGTH",
        "THINNING",
    }

    assert public_and_private_names.isdisjoint(
        forbidden_names
    )


def test_sampling_module_does_not_reuse_training_cd_k_as_chain_length():
    source = Path(
        sampling_module.__file__
    ).read_text(
        encoding="utf-8"
    )

    assert "CD_K" not in source
    assert "restricted_boltzmann_cd" not in source


def test_sampling_state_rejects_nonfinite_visible_values(
    checkpoint,
):
    state = initialize_chain(
        checkpoint=checkpoint,
        class_id=0,
        sample_index=21,
    )

    corrupted_visible = state.visible.copy()
    corrupted_visible[
        0,
        0,
    ] = np.nan

    corrupted = replace(
        state,
        visible=corrupted_visible,
    )

    with pytest.raises(
        Exception,
    ):
        gibbs_transition(
            checkpoint=checkpoint,
            state=corrupted,
            sample_index=21,
        )


def test_sampling_state_rejects_out_of_range_visible_values(
    checkpoint,
):
    state = initialize_chain(
        checkpoint=checkpoint,
        class_id=0,
        sample_index=22,
    )

    corrupted_visible = state.visible.copy()
    corrupted_visible[
        0,
        0,
    ] = 1.01

    corrupted = replace(
        state,
        visible=corrupted_visible,
    )

    with pytest.raises(
        Exception,
    ):
        gibbs_transition(
            checkpoint=checkpoint,
            state=corrupted,
            sample_index=22,
        )


def test_sampling_state_rejects_invalid_hidden_values(
    checkpoint,
):
    state = initialize_chain(
        checkpoint=checkpoint,
        class_id=0,
        sample_index=23,
    )

    corrupted_hidden = state.hidden.copy()
    corrupted_hidden[
        0,
        0,
    ] = 2.0

    corrupted = replace(
        state,
        hidden=corrupted_hidden,
    )

    with pytest.raises(
        ValueError,
        match="hidden state lies outside",
    ):
        gibbs_transition(
            checkpoint=checkpoint,
            state=corrupted,
            sample_index=23,
        )


def test_no_historical_initialization_fallback_surface_exists():
    source = Path(
        sampling_module.__file__
    ).read_text(
        encoding="utf-8"
    ).lower()

    forbidden = (
        "bernoulli(0.5)",
        "rbm_best",
        "weights.h5",
        "training example",
        "validation example",
        "test example",
        "np.clip",
        "tf.clip",
    )

    for token in forbidden:
        assert token not in source
