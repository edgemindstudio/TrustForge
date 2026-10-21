"""Regression tests for Paper 1 revision-2026 GMM initialization."""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from trustforge.models.paper01_revision_2026.gaussian_mixture_initialization import (
    INITIALIZATION_SAMPLE_CAP,
    KMEANS_PLUS_PLUS_LOCAL_TRIALS,
    MAX_SCALE_ROUNDTRIP_ULP,
    MIN_DISTINCT_CANONICAL_VECTORS,
    initialization_logical_positions,
    initialize_class_model_from_canonical_sample,
)
from trustforge.models.paper01_revision_2026.gaussian_mixture_model import (
    CANONICAL_CHANNELS,
    CANONICAL_HEIGHT,
    CANONICAL_WIDTH,
)
from trustforge.models.paper01_revision_2026.gaussian_mixture_observation import (
    CANONICAL_DIM,
    MIN_SCALE,
    MIN_VARIANCE,
    NUM_COMPONENTS,
)


def _canonical_images(
    count: int,
) -> tf.Tensor:
    """Create deterministic non-degenerate canonical test images."""

    base = np.linspace(
        0.0,
        1.0,
        CANONICAL_DIM,
        dtype=np.float32,
    )

    rows = []

    for index in range(
        count
    ):
        row = np.mod(
            base
            + np.float32(
                index
                / max(
                    count,
                    1,
                )
            ),
            np.float32(
                1.0
            ),
        )

        rows.append(
            row
        )

    matrix = np.stack(
        rows,
        axis=0,
    )

    return tf.constant(
        matrix.reshape(
            count,
            CANONICAL_HEIGHT,
            CANONICAL_WIDTH,
            CANONICAL_CHANNELS,
        ),
        dtype=tf.float32,
    )


def test_initialization_constants():
    assert INITIALIZATION_SAMPLE_CAP == 4096
    assert KMEANS_PLUS_PLUS_LOCAL_TRIALS == 3
    assert MIN_DISTINCT_CANONICAL_VECTORS == 5
    assert MAX_SCALE_ROUNDTRIP_ULP == 1
    assert NUM_COMPONENTS == 5
    assert CANONICAL_DIM == 4800


def test_positions_use_all_examples_at_or_below_cap():
    for count in (
        5,
        79,
        877,
        4096,
    ):
        positions = initialization_logical_positions(
            count
        )

        np.testing.assert_array_equal(
            positions,
            np.arange(
                count,
                dtype=np.int64,
            ),
        )


def test_positions_above_cap_follow_exact_integer_formula():
    for count in (
        4097,
        8100,
        12221,
        605255,
        619118,
    ):
        positions = initialization_logical_positions(
            count
        )

        oracle = np.array(
            [
                (
                    index
                    * (count - 1)
                )
                // (
                    INITIALIZATION_SAMPLE_CAP
                    - 1
                )
                for index in range(
                    INITIALIZATION_SAMPLE_CAP
                )
            ],
            dtype=np.int64,
        )

        np.testing.assert_array_equal(
            positions,
            oracle,
        )

        assert positions.dtype == np.int64
        assert len(
            positions
        ) == INITIALIZATION_SAMPLE_CAP
        assert int(
            positions[0]
        ) == 0
        assert int(
            positions[-1]
        ) == count - 1
        assert len(
            np.unique(
                positions
            )
        ) == INITIALIZATION_SAMPLE_CAP

        gaps = np.diff(
            positions
        )

        assert np.all(
            gaps > 0
        )

        assert int(
            gaps.max()
            - gaps.min()
        ) <= 1


def test_class_count_below_five_fails_closed():
    for count in (
        -1,
        0,
        1,
        4,
    ):
        with pytest.raises(
            ValueError
        ):
            initialization_logical_positions(
                count
            )


def test_initializer_is_seed_reproducible():
    images = _canonical_images(
        8
    )

    model_a, diagnostics_a = (
        initialize_class_model_from_canonical_sample(
            images,
            class_count=8,
            experimental_seed=42,
        )
    )

    model_b, diagnostics_b = (
        initialize_class_model_from_canonical_sample(
            images,
            class_count=8,
            experimental_seed=42,
        )
    )

    state_a = model_a.state_arrays()
    state_b = model_b.state_arrays()

    assert diagnostics_a == diagnostics_b

    for key in (
        "logits",
        "means",
        "raw_scales",
    ):
        np.testing.assert_array_equal(
            state_a[
                key
            ],
            state_b[
                key
            ],
        )


def test_different_seed_can_change_kmeans_plus_plus_centers():
    images = _canonical_images(
        16
    )

    model_a, diagnostics_a = (
        initialize_class_model_from_canonical_sample(
            images,
            class_count=16,
            experimental_seed=42,
        )
    )

    model_b, diagnostics_b = (
        initialize_class_model_from_canonical_sample(
            images,
            class_count=16,
            experimental_seed=43,
        )
    )

    assert (
        diagnostics_a.center_indices
        != diagnostics_b.center_indices
    )

    assert not np.array_equal(
        model_a.means.numpy(),
        model_b.means.numpy(),
    )


def test_initializer_uses_population_variance_ddof_zero():
    images = _canonical_images(
        7
    )

    model, diagnostics = (
        initialize_class_model_from_canonical_sample(
            images,
            class_count=7,
            experimental_seed=42,
        )
    )

    vectors = tf.reshape(
        tf.cast(
            images,
            tf.float64,
        ),
        (
            7,
            CANONICAL_DIM,
        ),
    ).numpy()

    population_variance = np.var(
        vectors,
        axis=0,
        ddof=0,
        dtype=np.float64,
    )

    sample_variance = np.var(
        vectors,
        axis=0,
        ddof=1,
        dtype=np.float64,
    )

    assert not np.array_equal(
        population_variance,
        sample_variance,
    )

    expected_variance = np.maximum(
        population_variance,
        np.float64(
            MIN_VARIANCE
        ),
    )

    expected_scale = np.sqrt(
        expected_variance,
        dtype=np.float64,
    )

    floor = np.float64(
        MIN_SCALE
    )

    expected_scale[
        expected_scale
        == floor
    ] = np.nextafter(
        floor,
        np.float64(
            np.inf
        ),
    )

    actual_scale = model.scales().numpy()[
        0
    ]

    actual_bits = actual_scale.view(
        np.uint64
    )

    expected_bits = expected_scale.view(
        np.uint64
    )

    ulp_distance = np.where(
        actual_bits >= expected_bits,
        actual_bits - expected_bits,
        expected_bits - actual_bits,
    )

    assert np.all(
        ulp_distance <= 1
    )

    assert (
        diagnostics.initialization_sample_count
        == 7
    )


def test_initializer_fails_closed_with_fewer_than_five_distinct_vectors():
    images = tf.zeros(
        (
            5,
            CANONICAL_HEIGHT,
            CANONICAL_WIDTH,
            CANONICAL_CHANNELS,
        ),
        dtype=tf.float32,
    )

    with pytest.raises(
        ValueError,
        match="at least 5 numerically distinct canonical vectors",
    ):
        initialize_class_model_from_canonical_sample(
            images,
            class_count=5,
            experimental_seed=42,
        )


def test_exact_floor_dimensions_receive_one_ulp_promotion():
    count = 5

    matrix = np.zeros(
        (
            count,
            CANONICAL_DIM,
        ),
        dtype=np.float32,
    )

    # Five numerically distinct canonical vectors with deliberately tiny
    # variance. Every dimension remains below the governed variance floor,
    # while k-means++ still has five distinct rows available.
    for index in range(
        count
    ):
        matrix[
            index,
            index,
        ] = np.float32(
            (index + 1)
            * 1.0e-4
        )

    images = tf.constant(
        matrix.reshape(
            count,
            CANONICAL_HEIGHT,
            CANONICAL_WIDTH,
            CANONICAL_CHANNELS,
        ),
        dtype=tf.float32,
    )

    model, diagnostics = (
        initialize_class_model_from_canonical_sample(
            images,
            class_count=count,
            experimental_seed=42,
        )
    )

    scales = model.scales().numpy()

    floor = np.float64(
        MIN_SCALE
    )

    expected = np.nextafter(
        floor,
        np.float64(
            np.inf
        ),
    )

    assert np.isfinite(
        model.raw_scales.numpy()
    ).all()

    assert np.all(
        scales
        == expected
    )

    assert diagnostics.variance_floor_dimension_count == CANONICAL_DIM
    assert diagnostics.variance_floor_fraction == 1.0

    assert diagnostics.one_ulp_promoted_dimension_count == CANONICAL_DIM
    assert diagnostics.one_ulp_promoted_fraction == 1.0


def test_initializer_produces_uniform_logits_and_five_selected_rows():
    images = _canonical_images(
        12
    )

    model, diagnostics = (
        initialize_class_model_from_canonical_sample(
            images,
            class_count=12,
            experimental_seed=42,
        )
    )

    np.testing.assert_array_equal(
        model.logits.numpy(),
        np.zeros(
            (
                NUM_COMPONENTS,
            ),
            dtype=np.float64,
        ),
    )

    assert len(
        diagnostics.center_indices
    ) == NUM_COMPONENTS

    assert len(
        set(
            diagnostics.center_indices
        )
    ) == NUM_COMPONENTS

    vectors = tf.reshape(
        tf.cast(
            images,
            tf.float64,
        ),
        (
            12,
            CANONICAL_DIM,
        ),
    ).numpy()

    np.testing.assert_array_equal(
        model.means.numpy(),
        vectors[
            list(
                diagnostics.center_indices
            )
        ],
    )


def test_initializer_repeats_same_scale_vector_across_components():
    images = _canonical_images(
        9
    )

    model, _ = (
        initialize_class_model_from_canonical_sample(
            images,
            class_count=9,
            experimental_seed=42,
        )
    )

    raw_scales = model.raw_scales.numpy()

    for component in range(
        1,
        NUM_COMPONENTS,
    ):
        np.testing.assert_array_equal(
            raw_scales[
                component
            ],
            raw_scales[
                0
            ],
        )


def test_initializer_rejects_wrong_sample_count():
    images = _canonical_images(
        7
    )

    with pytest.raises(
        ValueError
    ):
        initialize_class_model_from_canonical_sample(
            images,
            class_count=8,
            experimental_seed=42,
        )


def test_initializer_rejects_invalid_seed():
    images = _canonical_images(
        5
    )

    for seed in (
        -1,
        2**32,
    ):
        with pytest.raises(
            ValueError
        ):
            initialize_class_model_from_canonical_sample(
                images,
                class_count=5,
                experimental_seed=seed,
            )


def test_initializer_output_likelihood_is_finite():
    images = _canonical_images(
        6
    )

    model, _ = (
        initialize_class_model_from_canonical_sample(
            images,
            class_count=6,
            experimental_seed=42,
        )
    )

    log_prob = model.log_prob(
        images
    )

    loss = model.nll(
        images
    )

    assert bool(
        tf.reduce_all(
            tf.math.is_finite(
                log_prob
            )
        ).numpy()
    )

    assert bool(
        tf.math.is_finite(
            loss
        ).numpy()
    )

@pytest.mark.parametrize(
    "class_count",
    [
        5.0,
        5.5,
        "5",
    ],
)
def test_initialization_class_count_requires_exact_integer(
    class_count,
):
    with pytest.raises(
        TypeError,
        match="exact integer",
    ):
        initialization_logical_positions(
            class_count
        )


def test_initialization_class_count_accepts_numpy_integer():
    positions = initialization_logical_positions(
        np.int64(
            5
        )
    )

    np.testing.assert_array_equal(
        positions,
        np.arange(
            5,
            dtype=np.int64,
        ),
    )


def test_initialization_class_count_rejects_outside_int64_domain():
    with pytest.raises(
        ValueError,
        match="int64 logical-position domain",
    ):
        initialization_logical_positions(
            np.iinfo(
                np.int64
            ).max
            + 1
        )


def test_large_count_position_formula_uses_exact_integer_semantics():
    class_count = np.iinfo(
        np.int64
    ).max

    positions = initialization_logical_positions(
        class_count
    )

    assert positions.dtype == np.int64
    assert positions.shape == (
        INITIALIZATION_SAMPLE_CAP,
    )

    assert int(
        positions[
            0
        ]
    ) == 0

    assert int(
        positions[
            -1
        ]
    ) == (
        class_count
        - 1
    )

    expected_middle = (
        2048
        * (
            class_count
            - 1
        )
    ) // (
        INITIALIZATION_SAMPLE_CAP
        - 1
    )

    assert int(
        positions[
            2048
        ]
    ) == expected_middle

    gaps = np.diff(
        positions.astype(
            object
        )
    )

    assert all(
        gap > 0
        for gap in gaps
    )

    assert (
        max(
            gaps
        )
        - min(
            gaps
        )
    ) <= 1


@pytest.mark.parametrize(
    "experimental_seed",
    [
        42.0,
        42.5,
        "42",
    ],
)
def test_initialization_seed_requires_exact_integer(
    experimental_seed,
):
    images = np.zeros(
        (
            5,
            40,
            40,
            3,
        ),
        dtype=np.float32,
    )

    for index in range(
        5
    ):
        images[
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

    with pytest.raises(
        TypeError,
        match="exact integer",
    ):
        initialize_class_model_from_canonical_sample(
            images,
            class_count=5,
            experimental_seed=experimental_seed,
        )


def test_initialization_seed_accepts_numpy_integer():
    images = np.zeros(
        (
            5,
            40,
            40,
            3,
        ),
        dtype=np.float32,
    )

    for index in range(
        5
    ):
        images[
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

    model, diagnostics = (
        initialize_class_model_from_canonical_sample(
            images,
            class_count=5,
            experimental_seed=np.int64(
                42
            ),
        )
    )

    assert diagnostics.experimental_seed == 42
    assert model.state_arrays()[
        "means"
    ].shape == (
        5,
        4800,
    )
