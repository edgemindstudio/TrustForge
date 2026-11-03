"""Governed initialization for the Paper 1 revision-2026 GMM.

This module implements only class-model initialization.

Dataset adapters remain responsible for mapping class-specific logical
positions to canonical ``40 x 40 x 3`` float32 images.

Historical Gaussian-mixture implementations remain unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
import operator

import numpy as np
import tensorflow as tf
from sklearn.cluster import kmeans_plusplus

from trustforge.models.paper01_revision_2026.gaussian_mixture_model import (
    CensoredDiagonalGaussianMixture,
    canonical_images_to_vectors,
)
from trustforge.models.paper01_revision_2026.gaussian_mixture_observation import (
    CANONICAL_DIM,
    MIN_SCALE,
    MIN_VARIANCE,
    NUM_COMPONENTS,
    inverse_softplus_positive,
    transformed_scale,
)


INITIALIZATION_SAMPLE_CAP = 4096
KMEANS_PLUS_PLUS_LOCAL_TRIALS = 3
MIN_DISTINCT_CANONICAL_VECTORS = NUM_COMPONENTS
MAX_SCALE_ROUNDTRIP_ULP = 1


@dataclass(frozen=True)
class GMMInitializationDiagnostics:
    """Auditable facts about one class-model initialization."""

    class_count: int
    initialization_sample_count: int
    experimental_seed: int
    center_indices: tuple[int, ...]
    variance_floor_dimension_count: int
    variance_floor_fraction: float
    one_ulp_promoted_dimension_count: int
    one_ulp_promoted_fraction: float


def initialization_logical_positions(
    class_count: int,
) -> np.ndarray:
    """Return governed initialization positions for one ordered class.

    For ``N <= 4096``, every class-specific logical position is returned.

    For ``N > 4096``:

        floor(i * (N - 1) / 4095), i = 0,...,4095

    is evaluated using integer arithmetic only.
    """

    if isinstance(
        class_count,
        bool,
    ):
        raise TypeError(
            "class_count must be an integer, not bool"
        )

    try:
        class_count = operator.index(
            class_count
        )
    except TypeError as exc:
        raise TypeError(
            "class_count must be an exact integer"
        ) from exc

    if class_count > np.iinfo(
        np.int64
    ).max:
        raise ValueError(
            "class_count exceeds the governed int64 logical-position domain"
        )

    if class_count < NUM_COMPONENTS:
        raise ValueError(
            "revision-2026 GMM requires at least "
            f"{NUM_COMPONENTS} training examples per class; "
            f"got {class_count}"
        )

    if class_count <= INITIALIZATION_SAMPLE_CAP:
        return np.arange(
            class_count,
            dtype=np.int64,
        )

    # Evaluate the governed formula with arbitrary-precision Python integer
    # arithmetic before materializing the final int64 logical-position array.
    #
    # This avoids fixed-width multiplication overflow while preserving:
    #
    # floor(i * (N - 1) / 4095), i = 0,...,4095
    positions = np.asarray(
        [
            (
                i
                * (
                    class_count
                    - 1
                )
            )
            // (
                INITIALIZATION_SAMPLE_CAP
                - 1
            )
            for i in range(
                INITIALIZATION_SAMPLE_CAP
            )
        ],
        dtype=np.int64,
    )

    if positions.shape != (
        INITIALIZATION_SAMPLE_CAP,
    ):
        raise RuntimeError(
            "governed initialization position shape is invalid"
        )

    if positions.dtype != np.int64:
        raise RuntimeError(
            "governed initialization positions are not int64"
        )

    if int(
        positions[0]
    ) != 0:
        raise RuntimeError(
            "governed initialization positions do not include position 0"
        )

    if int(
        positions[-1]
    ) != class_count - 1:
        raise RuntimeError(
            "governed initialization positions do not include N-1"
        )

    gaps = np.diff(
        positions
    )

    if not np.all(
        gaps > 0
    ):
        raise RuntimeError(
            "governed initialization positions are not strictly increasing"
        )

    if int(
        gaps.max()
        - gaps.min()
    ) > 1:
        raise RuntimeError(
            "governed initialization position gap spread exceeds one"
        )

    return positions


def _validate_experimental_seed(
    experimental_seed: int,
) -> int:
    """Validate an sklearn-compatible governed experimental seed."""

    if isinstance(
        experimental_seed,
        bool,
    ):
        raise TypeError(
            "experimental_seed must be an integer, not bool"
        )

    try:
        seed = operator.index(
            experimental_seed
        )
    except TypeError as exc:
        raise TypeError(
            "experimental_seed must be an exact integer"
        ) from exc

    # sklearn RandomState-compatible integer range.
    if seed < 0 or seed > np.iinfo(
        np.uint32
    ).max:
        raise ValueError(
            "experimental_seed must lie in [0, 2**32 - 1]"
        )

    return seed


def _has_minimum_distinct_vectors(
    vectors: np.ndarray,
    *,
    minimum: int,
) -> bool:
    """Return whether at least ``minimum`` numerical vectors are distinct.

    Exact numerical equality is used. Because the canonical float32 vectors
    are converted exactly to float64 before this check, distinctness is
    unchanged by the dtype conversion.

    Only up to ``minimum`` representative rows are retained, so this does not
    construct a full unique-copy of a 4096 x 4800 initialization matrix.
    """

    matrix = np.asarray(
        vectors,
        dtype=np.float64,
    )

    if matrix.ndim != 2:
        raise ValueError(
            "distinct-vector check requires a rank-2 matrix"
        )

    representatives: list[np.ndarray] = []

    for row in matrix:
        if not any(
            np.array_equal(
                row,
                representative,
            )
            for representative in representatives
        ):
            representatives.append(
                row.copy()
            )

            if len(
                representatives
            ) >= minimum:
                return True

    return False


def _positive_float64_ulp_distance(
    actual: np.ndarray,
    expected: np.ndarray,
) -> np.ndarray:
    """Return exact ULP distances for finite positive float64 arrays."""

    actual_array = np.asarray(
        actual,
        dtype=np.float64,
    )

    expected_array = np.asarray(
        expected,
        dtype=np.float64,
    )

    if actual_array.shape != expected_array.shape:
        raise ValueError(
            "ULP comparison arrays must have identical shapes"
        )

    if not np.isfinite(
        actual_array
    ).all() or not np.isfinite(
        expected_array
    ).all():
        raise ValueError(
            "ULP comparison requires finite values"
        )

    if np.any(
        actual_array <= 0.0
    ) or np.any(
        expected_array <= 0.0
    ):
        raise ValueError(
            "ULP comparison helper requires positive values"
        )

    actual_bits = actual_array.view(
        np.uint64
    )

    expected_bits = expected_array.view(
        np.uint64
    )

    return np.where(
        actual_bits >= expected_bits,
        actual_bits - expected_bits,
        expected_bits - actual_bits,
    )


def _production_initial_scales(
    sample_variance: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    int,
]:
    """Apply the governed variance floor and representable-scale correction."""

    variance = np.asarray(
        sample_variance,
        dtype=np.float64,
    )

    if variance.shape != (
        CANONICAL_DIM,
    ):
        raise ValueError(
            "sample variance must have shape (4800,)"
        )

    if not np.isfinite(
        variance
    ).all():
        raise ValueError(
            "sample variance contains non-finite values"
        )

    if np.any(
        variance < 0.0
    ):
        raise ValueError(
            "sample variance contains negative values"
        )

    floor_variance = np.float64(
        MIN_VARIANCE
    )

    floored_variance = np.maximum(
        variance,
        floor_variance,
    )

    desired_scale = np.sqrt(
        floored_variance,
        dtype=np.float64,
    )

    floor_scale = np.float64(
        MIN_SCALE
    )

    next_scale = np.nextafter(
        floor_scale,
        np.float64(
            np.inf
        ),
    )

    exact_floor = (
        desired_scale
        == floor_scale
    )

    production_scale = desired_scale.copy()

    production_scale[
        exact_floor
    ] = next_scale

    if not np.isfinite(
        production_scale
    ).all():
        raise RuntimeError(
            "production initialization scale contains non-finite values"
        )

    if not np.all(
        production_scale
        > floor_scale
    ):
        raise RuntimeError(
            "production initialization scale is not strictly above floor"
        )

    offset = (
        production_scale
        - floor_scale
    )

    raw_vector = inverse_softplus_positive(
        tf.constant(
            offset,
            dtype=tf.float64,
        )
    ).numpy()

    # Correct only dimensions that exceed the governed one-ULP
    # initialization tolerance. Already acceptable raw values remain
    # bit-for-bit unchanged.
    actual_scale = transformed_scale(
        tf.constant(raw_vector, dtype=tf.float64)
    ).numpy()

    best_distance = _positive_float64_ulp_distance(
        actual_scale,
        production_scale,
    )

    needs_repair = (
        best_distance > MAX_SCALE_ROUNDTRIP_ULP
    )

    if np.any(needs_repair):
        baseline = raw_vector.copy()
        best_raw = raw_vector.copy()
        lower = baseline.copy()
        upper = baseline.copy()

        # Fixed search radius, with deterministic negative-first
        # tie-breaking. Evaluate through the production forward
        # transformation rather than a separate approximation.
        for _ in range(8):
            lower = np.nextafter(
                lower,
                np.float64(-np.inf),
            )
            upper = np.nextafter(
                upper,
                np.float64(np.inf),
            )

            for neighbor in (lower, upper):
                proposal = baseline.copy()
                proposal[needs_repair] = neighbor[needs_repair]

                if not np.isfinite(proposal).all():
                    continue

                try:
                    proposal_scale = transformed_scale(
                        tf.constant(proposal, dtype=tf.float64)
                    ).numpy()
                except tf.errors.InvalidArgumentError:
                    # An invalid neighboring candidate is rejected;
                    # the original strict scale contract remains.
                    continue

                proposal_distance = (
                    _positive_float64_ulp_distance(
                        proposal_scale,
                        production_scale,
                    )
                )

                improved = (
                    needs_repair
                    & (proposal_distance < best_distance)
                )

                best_raw[improved] = proposal[improved]
                best_distance[improved] = proposal_distance[improved]

        if np.any(
            best_distance > MAX_SCALE_ROUNDTRIP_ULP
        ):
            raise RuntimeError(
                "bounded GMM raw-scale correction could not "
                "satisfy governed one-ULP tolerance"
            )

        raw_vector = best_raw

    if raw_vector.dtype != np.float64:
        raise RuntimeError(
            "initial raw-scale vector is not float64"
        )

    if not np.isfinite(
        raw_vector
    ).all():
        raise RuntimeError(
            "initial raw-scale vector contains non-finite values"
        )

    return (
        floored_variance,
        raw_vector,
        int(
            np.count_nonzero(
                exact_floor
            )
        ),
    )


def initialize_class_model_from_canonical_sample(
    initialization_images: tf.Tensor | np.ndarray,
    *,
    class_count: int,
    experimental_seed: int,
) -> tuple[
    CensoredDiagonalGaussianMixture,
    GMMInitializationDiagnostics,
]:
    """Initialize one governed class-specific censored diagonal GMM.

    ``initialization_images`` must be exactly the canonical images selected
    by :func:`initialization_logical_positions` for the given class.
    """

    positions = initialization_logical_positions(
        class_count
    )

    seed = _validate_experimental_seed(
        experimental_seed
    )

    images = tf.convert_to_tensor(
        initialization_images
    )

    vectors_tensor = canonical_images_to_vectors(
        images
    )

    vectors = vectors_tensor.numpy()

    expected_count = int(
        positions.size
    )

    if vectors.shape != (
        expected_count,
        CANONICAL_DIM,
    ):
        raise ValueError(
            "initialization canonical batch does not match governed "
            "sample size: "
            f"expected {(expected_count, CANONICAL_DIM)}, "
            f"got {vectors.shape}"
        )

    if vectors.dtype != np.float64:
        raise RuntimeError(
            "initialization matrix is not float64"
        )

    if not np.isfinite(
        vectors
    ).all():
        raise ValueError(
            "initialization matrix contains non-finite values"
        )

    if not _has_minimum_distinct_vectors(
        vectors,
        minimum=MIN_DISTINCT_CANONICAL_VECTORS,
    ):
        raise ValueError(
            "revision-2026 GMM initialization requires at least "
            f"{MIN_DISTINCT_CANONICAL_VECTORS} numerically distinct "
            "canonical vectors"
        )

    centers, center_indices = (
        kmeans_plusplus(
            vectors,
            n_clusters=NUM_COMPONENTS,
            random_state=seed,
            n_local_trials=(
                KMEANS_PLUS_PLUS_LOCAL_TRIALS
            ),
        )
    )

    centers = np.asarray(
        centers,
        dtype=np.float64,
    )

    center_indices = np.asarray(
        center_indices,
        dtype=np.int64,
    )

    if centers.shape != (
        NUM_COMPONENTS,
        CANONICAL_DIM,
    ):
        raise RuntimeError(
            "k-means++ returned an invalid center matrix shape"
        )

    if center_indices.shape != (
        NUM_COMPONENTS,
    ):
        raise RuntimeError(
            "k-means++ returned an invalid center-index shape"
        )

    if len(
        np.unique(
            center_indices
        )
    ) != NUM_COMPONENTS:
        raise RuntimeError(
            "k-means++ did not return five distinct center indices "
            "despite a valid distinct-vector initialization sample"
        )

    if not np.array_equal(
        centers,
        vectors[
            center_indices
        ],
    ):
        raise RuntimeError(
            "k-means++ centers do not exactly match selected sample rows"
        )

    if not np.isfinite(
        centers
    ).all():
        raise RuntimeError(
            "initial GMM means contain non-finite values"
        )

    if np.unique(
        centers,
        axis=0,
    ).shape[0] != NUM_COMPONENTS:
        raise RuntimeError(
            "k-means++ did not return five numerically distinct centers"
        )

    sample_variance = np.var(
        vectors,
        axis=0,
        ddof=0,
        dtype=np.float64,
    )

    (
        floored_variance,
        raw_scale_vector,
        promoted_count,
    ) = _production_initial_scales(
        sample_variance
    )

    variance_floor_count = int(
        np.count_nonzero(
            sample_variance
            <= np.float64(
                MIN_VARIANCE
            )
        )
    )

    logits = np.zeros(
        (
            NUM_COMPONENTS,
        ),
        dtype=np.float64,
    )

    raw_scales = np.repeat(
        raw_scale_vector[
            None,
            :,
        ],
        NUM_COMPONENTS,
        axis=0,
    )

    model = CensoredDiagonalGaussianMixture(
        logits=logits,
        means=centers,
        raw_scales=raw_scales,
    )

    scales = model.scales().numpy()

    expected_scale = np.sqrt(
        floored_variance,
        dtype=np.float64,
    )

    floor_scale = np.float64(
        MIN_SCALE
    )

    expected_scale[
        expected_scale
        == floor_scale
    ] = np.nextafter(
        floor_scale,
        np.float64(
            np.inf
        ),
    )

    scale_ulp_distance = _positive_float64_ulp_distance(
        scales[
            0
        ],
        expected_scale,
    )

    if np.any(
        scale_ulp_distance
        > MAX_SCALE_ROUNDTRIP_ULP
    ):
        raise RuntimeError(
            "initial scale round-trip exceeds governed one-ULP tolerance"
        )

    for component in range(
        1,
        NUM_COMPONENTS,
    ):
        np.testing.assert_array_equal(
            scales[
                component
            ],
            scales[
                0
            ],
        )

    diagnostics = GMMInitializationDiagnostics(
        class_count=int(
            class_count
        ),
        initialization_sample_count=expected_count,
        experimental_seed=seed,
        center_indices=tuple(
            int(index)
            for index in center_indices.tolist()
        ),
        variance_floor_dimension_count=(
            variance_floor_count
        ),
        variance_floor_fraction=(
            variance_floor_count
            / CANONICAL_DIM
        ),
        one_ulp_promoted_dimension_count=(
            promoted_count
        ),
        one_ulp_promoted_fraction=(
            promoted_count
            / CANONICAL_DIM
        ),
    )

    return (
        model,
        diagnostics,
    )


__all__ = [
    "GMMInitializationDiagnostics",
    "INITIALIZATION_SAMPLE_CAP",
    "KMEANS_PLUS_PLUS_LOCAL_TRIALS",
    "MAX_SCALE_ROUNDTRIP_ULP",
    "MIN_DISTINCT_CANONICAL_VECTORS",
    "initialization_logical_positions",
    "initialize_class_model_from_canonical_sample",
]
