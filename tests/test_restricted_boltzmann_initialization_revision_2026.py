"""Focused tests for governed revision-2026 RBM initialization."""

from __future__ import annotations

import numpy as np
import pytest

from trustforge.models.paper01_revision_2026.restricted_boltzmann_initialization import (
    INITIAL_WEIGHT_STD,
    RBMInitializationDiagnostics,
    full_training_scalar_mean,
    initialize_shared_conditional_rbm,
    solve_visible_bias_scalar,
    visible_mean_from_scalar_theta,
)
from trustforge.models.paper01_revision_2026.restricted_boltzmann_model import (
    HIDDEN_UNITS,
    VISIBLE_DIM,
)


def test_initial_weight_std_is_governed_value():
    assert INITIAL_WEIGHT_STD == 0.01


@pytest.mark.parametrize(
    "training_mean",
    [
        0.063764118469,
        0.447801859279,
        0.071689729239,
        0.399391633345,
        0.1,
        0.5,
        0.9,
    ],
)
def test_visible_bias_root_recovers_training_mean(training_mean):
    a0 = solve_visible_bias_scalar(
        training_mean
    )

    recovered = visible_mean_from_scalar_theta(
        a0
    )

    assert np.isfinite(a0)
    assert abs(
        recovered - training_mean
    ) <= 5.0e-11


def test_visible_bias_root_at_half_is_zero():
    assert solve_visible_bias_scalar(
        0.5
    ) == 0.0


@pytest.mark.parametrize(
    "bad_mean",
    [
        0.0,
        1.0,
        -1.0e-12,
        1.0 + 1.0e-12,
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_training_scalar_mean_fails_closed_outside_open_unit_interval(
    bad_mean,
):
    with pytest.raises(ValueError):
        solve_visible_bias_scalar(
            bad_mean
        )


def test_initialization_state_shapes_and_dtypes():
    model, diagnostics = initialize_shared_conditional_rbm(
        class_count=5,
        experimental_seed=42,
        training_scalar_mean=0.447801859279,
    )

    state = model.state_dict()

    assert state["W"].shape == (
        VISIBLE_DIM,
        HIDDEN_UNITS,
    )
    assert state["a"].shape == (
        VISIBLE_DIM,
    )
    assert state["b"].shape == (
        HIDDEN_UNITS,
    )
    assert state["U"].shape == (
        5,
        HIDDEN_UNITS,
    )

    for value in state.values():
        assert value.dtype == np.float64
        assert np.all(np.isfinite(value))

    assert isinstance(
        diagnostics,
        RBMInitializationDiagnostics,
    )


def test_visible_bias_is_one_global_scalar_replicated_across_all_coordinates():
    model, diagnostics = initialize_shared_conditional_rbm(
        class_count=3,
        experimental_seed=42,
        training_scalar_mean=0.399391633345,
    )

    a = model.state_dict()["a"]

    assert np.all(
        a == diagnostics.visible_bias_scalar
    )


def test_hidden_bias_is_zero():
    model, _ = initialize_shared_conditional_rbm(
        class_count=3,
        experimental_seed=42,
        training_scalar_mean=0.4,
    )

    b = model.state_dict()["b"]

    assert np.all(
        b == 0.0
    )


def test_class_interaction_is_zero():
    model, _ = initialize_shared_conditional_rbm(
        class_count=7,
        experimental_seed=42,
        training_scalar_mean=0.4,
    )

    U = model.state_dict()["U"]

    assert np.all(
        U == 0.0
    )


def test_same_seed_produces_identical_weight_initialization():
    first, _ = initialize_shared_conditional_rbm(
        class_count=5,
        experimental_seed=42,
        training_scalar_mean=0.4,
    )
    second, _ = initialize_shared_conditional_rbm(
        class_count=5,
        experimental_seed=42,
        training_scalar_mean=0.4,
    )

    np.testing.assert_array_equal(
        first.state_dict()["W"],
        second.state_dict()["W"],
    )


def test_different_seed_changes_weight_initialization():
    first, _ = initialize_shared_conditional_rbm(
        class_count=5,
        experimental_seed=42,
        training_scalar_mean=0.4,
    )
    second, _ = initialize_shared_conditional_rbm(
        class_count=5,
        experimental_seed=43,
        training_scalar_mean=0.4,
    )

    assert not np.array_equal(
        first.state_dict()["W"],
        second.state_dict()["W"],
    )


def test_empirical_weight_scale_matches_selected_std():
    model, _ = initialize_shared_conditional_rbm(
        class_count=5,
        experimental_seed=42,
        training_scalar_mean=0.4,
    )

    W = model.state_dict()["W"]

    assert abs(
        float(W.mean())
    ) < 1.0e-3

    assert abs(
        float(W.std()) - INITIAL_WEIGHT_STD
    ) < 5.0e-4


def test_diagnostics_record_governed_initialization_identity():
    _, diagnostics = initialize_shared_conditional_rbm(
        class_count=5,
        experimental_seed=42,
        training_scalar_mean=0.447801859279,
    )

    assert diagnostics.class_count == 5
    assert diagnostics.experimental_seed == 42
    assert diagnostics.training_scalar_mean == 0.447801859279
    assert diagnostics.weight_std == 0.01

    recovered = visible_mean_from_scalar_theta(
        diagnostics.visible_bias_scalar
    )

    assert abs(
        recovered
        - diagnostics.training_scalar_mean
    ) <= 5.0e-11


@pytest.mark.parametrize(
    "bad_class_count",
    [
        0,
        -1,
        True,
        2.5,
    ],
)
def test_invalid_class_count_fails_closed(
    bad_class_count,
):
    with pytest.raises(
        (TypeError, ValueError)
    ):
        initialize_shared_conditional_rbm(
            class_count=bad_class_count,
            experimental_seed=42,
            training_scalar_mean=0.4,
        )


@pytest.mark.parametrize(
    "bad_seed",
    [
        -1,
        True,
        1.5,
        2**32,
    ],
)
def test_invalid_experimental_seed_fails_closed(
    bad_seed,
):
    with pytest.raises(
        (TypeError, ValueError)
    ):
        initialize_shared_conditional_rbm(
            class_count=3,
            experimental_seed=bad_seed,
            training_scalar_mean=0.4,
        )


def test_representative_audit_values_are_not_required_constants():
    means = [
        0.2,
        0.3,
        0.6,
        0.8,
    ]

    for training_mean in means:
        a0 = solve_visible_bias_scalar(
            training_mean
        )
        recovered = visible_mean_from_scalar_theta(
            a0
        )

        assert abs(
            recovered - training_mean
        ) <= 5.0e-11


class _FakeCanonicalTrainingAdapter:
    def __init__(
        self,
        images: np.ndarray,
    ):
        self.images = images
        self.calls = []

    def split_size(
        self,
        split,
    ):
        assert split == "train"
        return self.images.shape[0]

    def canonical_batch(
        self,
        split,
        logical_positions,
    ):
        assert split == "train"

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

        labels = np.zeros(
            (positions.size,),
            dtype=np.int64,
        )

        return (
            self.images[positions],
            labels,
        )


def test_full_training_scalar_mean_streams_all_positions_exactly_once():
    images = np.stack(
        [
            np.full(
                (40, 40, 3),
                0.25,
                dtype=np.float32,
            ),
            np.full(
                (40, 40, 3),
                0.50,
                dtype=np.float32,
            ),
            np.full(
                (40, 40, 3),
                0.75,
                dtype=np.float32,
            ),
            np.full(
                (40, 40, 3),
                1.00,
                dtype=np.float32,
            ),
            np.full(
                (40, 40, 3),
                0.00,
                dtype=np.float32,
            ),
        ],
        axis=0,
    )

    adapter = _FakeCanonicalTrainingAdapter(
        images
    )

    result = full_training_scalar_mean(
        adapter,
        batch_size=2,
    )

    expected = float(
        images.astype(
            np.float64
        ).mean()
    )

    visited = np.concatenate(
        [
            positions
            for _, positions in adapter.calls
        ]
    )

    assert result == expected

    np.testing.assert_array_equal(
        visited,
        np.arange(
            images.shape[0],
            dtype=np.int64,
        ),
    )

    assert all(
        split == "train"
        for split, _ in adapter.calls
    )

    assert [
        positions.tolist()
        for _, positions in adapter.calls
    ] == [
        [0, 1],
        [2, 3],
        [4],
    ]


def test_full_training_scalar_mean_accumulates_float64_not_float32():
    values = np.array(
        [
            0.10000001,
            0.20000002,
            0.30000004,
        ],
        dtype=np.float32,
    )

    images = np.stack(
        [
            np.full(
                (40, 40, 3),
                value,
                dtype=np.float32,
            )
            for value in values
        ],
        axis=0,
    )

    adapter = _FakeCanonicalTrainingAdapter(
        images
    )

    result = full_training_scalar_mean(
        adapter,
        batch_size=1,
    )

    expected = float(
        np.sum(
            images.astype(
                np.float64
            ),
            dtype=np.float64,
        )
        / np.float64(
            images.size
        )
    )

    assert result == expected


def test_full_training_scalar_mean_never_requests_validation_or_test():
    images = np.full(
        (3, 40, 40, 3),
        0.4,
        dtype=np.float32,
    )

    adapter = _FakeCanonicalTrainingAdapter(
        images
    )

    result = full_training_scalar_mean(
        adapter,
        batch_size=2,
    )

    assert result == pytest.approx(
        0.4
    )

    assert {
        split
        for split, _ in adapter.calls
    } == {
        "train"
    }


@pytest.mark.parametrize(
    "bad_batch_size",
    [
        0,
        -1,
        True,
        1.5,
    ],
)
def test_full_training_scalar_mean_rejects_invalid_batch_size(
    bad_batch_size,
):
    images = np.full(
        (1, 40, 40, 3),
        0.4,
        dtype=np.float32,
    )

    adapter = _FakeCanonicalTrainingAdapter(
        images
    )

    with pytest.raises(
        (TypeError, ValueError)
    ):
        full_training_scalar_mean(
            adapter,
            batch_size=bad_batch_size,
        )


def test_full_training_scalar_mean_requires_split_size():
    class MissingSplitSize:
        def canonical_batch(
            self,
            split,
            logical_positions,
        ):
            raise AssertionError(
                "must not be called"
            )

    with pytest.raises(TypeError):
        full_training_scalar_mean(
            MissingSplitSize()
        )


def test_full_training_scalar_mean_requires_canonical_batch():
    class MissingCanonicalBatch:
        def split_size(
            self,
            split,
        ):
            return 1

    with pytest.raises(TypeError):
        full_training_scalar_mean(
            MissingCanonicalBatch()
        )


def test_full_training_scalar_mean_rejects_empty_training_split():
    class EmptyAdapter:
        def split_size(
            self,
            split,
        ):
            assert split == "train"
            return 0

        def canonical_batch(
            self,
            split,
            logical_positions,
        ):
            raise AssertionError(
                "empty split must not be read"
            )

    with pytest.raises(ValueError):
        full_training_scalar_mean(
            EmptyAdapter()
        )


def test_full_training_scalar_mean_requires_float32_canonical_images():
    images = np.full(
        (2, 40, 40, 3),
        0.4,
        dtype=np.float64,
    )

    adapter = _FakeCanonicalTrainingAdapter(
        images
    )

    with pytest.raises(TypeError):
        full_training_scalar_mean(
            adapter
        )


@pytest.mark.parametrize(
    "shape",
    [
        (2, 40, 40),
        (2, 39, 40, 3),
        (2, 40, 39, 3),
        (2, 40, 40, 1),
    ],
)
def test_full_training_scalar_mean_rejects_wrong_canonical_shape(
    shape,
):
    images = np.full(
        shape,
        0.4,
        dtype=np.float32,
    )

    adapter = _FakeCanonicalTrainingAdapter(
        images
    )

    with pytest.raises(ValueError):
        full_training_scalar_mean(
            adapter
        )


@pytest.mark.parametrize(
    "bad_value",
    [
        -1.0e-6,
        1.0 + 1.0e-6,
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_full_training_scalar_mean_rejects_invalid_canonical_values(
    bad_value,
):
    images = np.full(
        (2, 40, 40, 3),
        0.4,
        dtype=np.float32,
    )

    images[0, 0, 0, 0] = bad_value

    adapter = _FakeCanonicalTrainingAdapter(
        images
    )

    with pytest.raises(ValueError):
        full_training_scalar_mean(
            adapter
        )


def test_full_training_scalar_mean_rejects_batch_cardinality_mismatch():
    class BadCardinalityAdapter:
        def split_size(
            self,
            split,
        ):
            assert split == "train"
            return 3

        def canonical_batch(
            self,
            split,
            logical_positions,
        ):
            assert split == "train"

            return (
                np.zeros(
                    (1, 40, 40, 3),
                    dtype=np.float32,
                ),
                np.zeros(
                    (1,),
                    dtype=np.int64,
                ),
            )

    with pytest.raises(ValueError):
        full_training_scalar_mean(
            BadCardinalityAdapter(),
            batch_size=2,
        )


@pytest.mark.parametrize(
    "constant_value",
    [
        0.0,
        1.0,
    ],
)
def test_full_training_scalar_mean_fails_closed_at_degenerate_endpoint_mean(
    constant_value,
):
    images = np.full(
        (2, 40, 40, 3),
        constant_value,
        dtype=np.float32,
    )

    adapter = _FakeCanonicalTrainingAdapter(
        images
    )

    with pytest.raises(ValueError):
        full_training_scalar_mean(
            adapter
        )
