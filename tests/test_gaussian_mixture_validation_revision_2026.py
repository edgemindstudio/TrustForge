"""Tests for Paper 1 revision-2026 full-class GMM validation."""

from __future__ import annotations

import math

import numpy as np
import pytest
import tensorflow as tf

from trustforge.models.paper01_revision_2026.gaussian_mixture_model import (
    CensoredDiagonalGaussianMixture,
)
from trustforge.models.paper01_revision_2026.gaussian_mixture_observation import (
    CANONICAL_DIM,
    NUM_COMPONENTS,
)
from trustforge.models.paper01_revision_2026.gaussian_mixture_validation import (
    VALIDATION_BATCH_SIZE_MAX,
    evaluate_full_class_validation_nll,
)


CLASS_ID = 2
NUM_CLASSES = 5


class _SyntheticAdapter:
    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        self.images = images
        self.labels = labels
        self.calls: list[
            np.ndarray
        ] = []

    def canonical_batch(
        self,
        split: str,
        logical_positions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if split != "validation":
            raise ValueError(
                "unexpected split"
            )

        positions = np.asarray(
            logical_positions,
            dtype=np.int64,
        )

        self.calls.append(
            positions.copy()
        )

        return (
            self.images[
                positions
            ],
            self.labels[
                positions
            ],
        )


def _model() -> CensoredDiagonalGaussianMixture:
    return CensoredDiagonalGaussianMixture(
        logits=np.zeros(
            NUM_COMPONENTS,
            dtype=np.float64,
        ),
        means=np.full(
            (
                NUM_COMPONENTS,
                CANONICAL_DIM,
            ),
            0.5,
            dtype=np.float64,
        ),
        raw_scales=np.zeros(
            (
                NUM_COMPONENTS,
                CANONICAL_DIM,
            ),
            dtype=np.float64,
        ),
    )


def _fixture(
    count: int,
) -> tuple[
    _SyntheticAdapter,
    np.ndarray,
]:
    values = np.linspace(
        0.05,
        0.95,
        count,
        dtype=np.float32,
    )

    images = np.empty(
        (
            count,
            40,
            40,
            3,
        ),
        dtype=np.float32,
    )

    for index, value in enumerate(
        values
    ):
        images[
            index
        ].fill(
            value
        )

    labels = np.zeros(
        (
            count,
            NUM_CLASSES,
        ),
        dtype=np.float32,
    )

    labels[
        :,
        CLASS_ID,
    ] = 1.0

    return (
        _SyntheticAdapter(
            images,
            labels,
        ),
        np.arange(
            count,
            dtype=np.int64,
        ),
    )


def test_governed_validation_batch_maximum():
    assert VALIDATION_BATCH_SIZE_MAX == 128


@pytest.mark.parametrize(
    "count",
    [
        1,
        12,
        127,
        128,
        129,
        257,
    ],
)
def test_full_validation_covers_every_example_exactly_once(
    count,
):
    adapter, positions = _fixture(
        count
    )

    result = evaluate_full_class_validation_nll(
        _model(),
        adapter=adapter,
        logical_positions=positions,
        class_id=CLASS_ID,
    )

    observed = np.concatenate(
        adapter.calls
    )

    np.testing.assert_array_equal(
        observed,
        positions,
    )

    assert result.example_count == count

    assert result.batch_count == math.ceil(
        count
        / 128
    )

    assert max(
        len(
            batch
        )
        for batch in adapter.calls
    ) <= 128


def test_example_weighted_mean_matches_one_batch_reference():
    adapter, positions = _fixture(
        257
    )

    model = _model()

    expected = -np.mean(
        model.log_prob(
            tf.convert_to_tensor(
                adapter.images,
                dtype=tf.float32,
            )
        ).numpy(),
        dtype=np.float64,
    )

    result = evaluate_full_class_validation_nll(
        model,
        adapter=adapter,
        logical_positions=positions,
        class_id=CLASS_ID,
        batch_size_max=128,
    )

    assert result.batch_count == 3

    assert result.mean_nll == pytest.approx(
        float(
            expected
        ),
        rel=1e-13,
        abs=1e-10,
    )


def test_weighting_is_independent_of_validation_batch_partition():
    adapter_a, positions = _fixture(
        257
    )

    adapter_b = _SyntheticAdapter(
        adapter_a.images.copy(),
        adapter_a.labels.copy(),
    )

    model_a = _model()
    model_b = _model()

    result_128 = evaluate_full_class_validation_nll(
        model_a,
        adapter=adapter_a,
        logical_positions=positions,
        class_id=CLASS_ID,
        batch_size_max=128,
    )

    result_37 = evaluate_full_class_validation_nll(
        model_b,
        adapter=adapter_b,
        logical_positions=positions,
        class_id=CLASS_ID,
        batch_size_max=37,
    )

    assert result_128.mean_nll == pytest.approx(
        result_37.mean_nll,
        rel=1e-13,
        abs=1e-10,
    )


def test_validation_does_not_mutate_model_state():
    adapter, positions = _fixture(
        12
    )

    model = _model()

    before = model.state_arrays()

    evaluate_full_class_validation_nll(
        model,
        adapter=adapter,
        logical_positions=positions,
        class_id=CLASS_ID,
    )

    after = model.state_arrays()

    for key in before:
        np.testing.assert_array_equal(
            after[
                key
            ],
            before[
                key
            ],
        )


def test_wrong_class_membership_fails_closed():
    adapter, positions = _fixture(
        12
    )

    adapter.labels[
        5,
        :,
    ] = 0.0

    adapter.labels[
        5,
        1,
    ] = 1.0

    with pytest.raises(
        ValueError,
        match="outside governed class",
    ):
        evaluate_full_class_validation_nll(
            _model(),
            adapter=adapter,
            logical_positions=positions,
            class_id=CLASS_ID,
        )


def test_duplicate_validation_positions_fail_closed():
    adapter, _ = _fixture(
        12
    )

    positions = np.array(
        [
            0,
            1,
            2,
            2,
        ],
        dtype=np.int64,
    )

    with pytest.raises(
        ValueError,
        match="duplicates",
    ):
        evaluate_full_class_validation_nll(
            _model(),
            adapter=adapter,
            logical_positions=positions,
            class_id=CLASS_ID,
        )


def test_empty_validation_positions_fail_closed():
    adapter, _ = _fixture(
        1
    )

    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        evaluate_full_class_validation_nll(
            _model(),
            adapter=adapter,
            logical_positions=np.empty(
                0,
                dtype=np.int64,
            ),
            class_id=CLASS_ID,
        )


def test_noninteger_validation_positions_fail_closed():
    adapter, _ = _fixture(
        3
    )

    with pytest.raises(
        TypeError
    ):
        evaluate_full_class_validation_nll(
            _model(),
            adapter=adapter,
            logical_positions=np.array(
                [
                    0.0,
                    1.0,
                ],
                dtype=np.float64,
            ),
            class_id=CLASS_ID,
        )


def test_invalid_one_hot_row_fails_closed():
    adapter, positions = _fixture(
        4
    )

    adapter.labels[
        0,
        CLASS_ID,
    ] = 0.5

    with pytest.raises(
        ValueError,
        match="one-hot",
    ):
        evaluate_full_class_validation_nll(
            _model(),
            adapter=adapter,
            logical_positions=positions,
            class_id=CLASS_ID,
        )


def test_validation_image_dtype_fails_closed():
    adapter, positions = _fixture(
        4
    )

    adapter.images = adapter.images.astype(
        np.float64
    )

    with pytest.raises(
        TypeError,
        match="float32",
    ):
        evaluate_full_class_validation_nll(
            _model(),
            adapter=adapter,
            logical_positions=positions,
            class_id=CLASS_ID,
        )


@pytest.mark.parametrize(
    "batch_size_max",
    [
        0,
        -1,
        129,
    ],
)
def test_invalid_validation_batch_size_fails_closed(
    batch_size_max,
):
    adapter, positions = _fixture(
        4
    )

    with pytest.raises(
        ValueError
    ):
        evaluate_full_class_validation_nll(
            _model(),
            adapter=adapter,
            logical_positions=positions,
            class_id=CLASS_ID,
            batch_size_max=batch_size_max,
        )
