"""Tests for governed Paper 1 revision-2026 GMM sampling."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from trustforge.models.paper01_revision_2026.gaussian_mixture_checkpoint import (
    persist_gmm_checkpoint,
)
from trustforge.models.paper01_revision_2026.gaussian_mixture_initialization import (
    GMMInitializationDiagnostics,
)
from trustforge.models.paper01_revision_2026.gaussian_mixture_model import (
    CensoredDiagonalGaussianMixture,
)
from trustforge.models.paper01_revision_2026.gaussian_mixture_observation import (
    CANONICAL_DIM,
    NUM_COMPONENTS,
)
from trustforge.models.paper01_revision_2026.gaussian_mixture_sampling import (
    PRIMARY_SYNTHETIC_SAMPLES_PER_CLASS,
    SAMPLING_PURPOSE,
    sample_gmm_checkpoint,
    sampling_identity_sha256,
)
from trustforge.models.paper01_revision_2026.gaussian_mixture_training import (
    GMMClassTrainingResult,
    GMMValidationCheckpoint,
)
from trustforge.models.paper01_revision_2026.gaussian_mixture_training_budget import (
    training_budget,
)


DATASET_ID = "unit_test_dataset"
CLASS_ID = 3
SEED = 42


def _training_result() -> GMMClassTrainingResult:
    model = CensoredDiagonalGaussianMixture(
        logits=np.zeros(
            NUM_COMPONENTS,
            dtype=np.float64,
        ),
        means=np.stack(
            [
                np.full(
                    CANONICAL_DIM,
                    -0.25,
                    dtype=np.float64,
                ),
                np.full(
                    CANONICAL_DIM,
                    0.10,
                    dtype=np.float64,
                ),
                np.full(
                    CANONICAL_DIM,
                    0.50,
                    dtype=np.float64,
                ),
                np.full(
                    CANONICAL_DIM,
                    0.90,
                    dtype=np.float64,
                ),
                np.full(
                    CANONICAL_DIM,
                    1.25,
                    dtype=np.float64,
                ),
            ],
            axis=0,
        ),
        raw_scales=np.full(
            (
                NUM_COMPONENTS,
                CANONICAL_DIM,
            ),
            -2.0,
            dtype=np.float64,
        ),
    )

    budget = training_budget(
        5
    )

    history = tuple(
        GMMValidationCheckpoint(
            step=step,
            validation_nll=nll,
        )
        for step, nll in zip(
            budget.validation_steps,
            (
                10.0,
                9.0,
                8.0,
                7.0,
                6.0,
                5.0,
            ),
        )
    )

    diagnostics = GMMInitializationDiagnostics(
        class_count=5,
        initialization_sample_count=5,
        experimental_seed=SEED,
        center_indices=(
            0,
            1,
            2,
            3,
            4,
        ),
        variance_floor_dimension_count=0,
        variance_floor_fraction=0.0,
        one_ulp_promoted_dimension_count=0,
        one_ulp_promoted_fraction=0.0,
    )

    return GMMClassTrainingResult(
        model=model,
        initialization_diagnostics=diagnostics,
        budget=budget,
        validation_history=history,
        selected_step=100,
        selected_validation_nll=5.0,
        optimizer_steps_completed=100,
    )


def _persist(
    directory: Path,
    *,
    dataset_id: str = DATASET_ID,
    class_id: int = CLASS_ID,
    experimental_seed: int = SEED,
) -> None:
    persist_gmm_checkpoint(
        directory,
        dataset_id=dataset_id,
        class_id=class_id,
        experimental_seed=experimental_seed,
        training_result=_training_result(),
        validation_class_count=2,
    )


def test_primary_budget_constant():
    assert PRIMARY_SYNTHETIC_SAMPLES_PER_CLASS == 2000


def test_sampling_purpose_constant():
    assert SAMPLING_PURPOSE == (
        "paper01_revision2026_gmm_sampling"
    )


def test_identity_digest_is_reproducible():
    a = sampling_identity_sha256(
        dataset_id=DATASET_ID,
        class_id=CLASS_ID,
        experimental_seed=SEED,
    )

    b = sampling_identity_sha256(
        dataset_id=DATASET_ID,
        class_id=CLASS_ID,
        experimental_seed=SEED,
    )

    assert a == b
    assert len(a) == 64


def test_identity_digest_changes_with_each_identity_field():
    base = sampling_identity_sha256(
        dataset_id=DATASET_ID,
        class_id=CLASS_ID,
        experimental_seed=SEED,
    )

    assert sampling_identity_sha256(
        dataset_id="other_dataset",
        class_id=CLASS_ID,
        experimental_seed=SEED,
    ) != base

    assert sampling_identity_sha256(
        dataset_id=DATASET_ID,
        class_id=CLASS_ID + 1,
        experimental_seed=SEED,
    ) != base

    assert sampling_identity_sha256(
        dataset_id=DATASET_ID,
        class_id=CLASS_ID,
        experimental_seed=SEED + 1,
    ) != base


@pytest.mark.parametrize(
    "bad_count",
    [
        0,
        -1,
        True,
        1.0,
        "1",
    ],
)
def test_sample_count_must_be_exact_positive_integer(
    tmp_path: Path,
    bad_count,
):
    _persist(
        tmp_path
    )

    with pytest.raises(
        (
            TypeError,
            ValueError,
        )
    ):
        sample_gmm_checkpoint(
            tmp_path,
            dataset_id=DATASET_ID,
            class_id=CLASS_ID,
            experimental_seed=SEED,
            sample_count=bad_count,
        )


def test_fixed_identity_and_checkpoint_are_bitwise_reproducible(
    tmp_path: Path,
):
    _persist(
        tmp_path
    )

    first = sample_gmm_checkpoint(
        tmp_path,
        dataset_id=DATASET_ID,
        class_id=CLASS_ID,
        experimental_seed=SEED,
        sample_count=8,
    )

    second = sample_gmm_checkpoint(
        tmp_path,
        dataset_id=DATASET_ID,
        class_id=CLASS_ID,
        experimental_seed=SEED,
        sample_count=8,
    )

    assert (
        first.sampling_identity_sha256
        == second.sampling_identity_sha256
    )

    assert np.array_equal(
        first.images,
        second.images,
    )


def test_sampling_output_contract(
    tmp_path: Path,
):
    _persist(
        tmp_path
    )

    result = sample_gmm_checkpoint(
        tmp_path,
        dataset_id=DATASET_ID,
        class_id=CLASS_ID,
        experimental_seed=SEED,
        sample_count=11,
    )

    assert result.images.shape == (
        11,
        40,
        40,
        3,
    )

    assert result.images.dtype == np.float32

    assert np.isfinite(
        result.images
    ).all()

    assert np.all(
        result.images >= 0.0
    )

    assert np.all(
        result.images <= 1.0
    )


def test_censored_model_generates_legitimate_endpoints(
    tmp_path: Path,
):
    _persist(
        tmp_path
    )

    result = sample_gmm_checkpoint(
        tmp_path,
        dataset_id=DATASET_ID,
        class_id=CLASS_ID,
        experimental_seed=SEED,
        sample_count=32,
    )

    assert np.any(
        result.images == 0.0
    )

    assert np.any(
        result.images == 1.0
    )


def test_changed_seed_changes_samples(
    tmp_path: Path,
):
    directory_a = (
        tmp_path
        / "seed42"
    )

    directory_b = (
        tmp_path
        / "seed43"
    )

    directory_a.mkdir()
    directory_b.mkdir()

    _persist(
        directory_a,
        experimental_seed=42,
    )

    _persist(
        directory_b,
        experimental_seed=43,
    )

    a = sample_gmm_checkpoint(
        directory_a,
        dataset_id=DATASET_ID,
        class_id=CLASS_ID,
        experimental_seed=42,
        sample_count=8,
    )

    b = sample_gmm_checkpoint(
        directory_b,
        dataset_id=DATASET_ID,
        class_id=CLASS_ID,
        experimental_seed=43,
        sample_count=8,
    )

    assert (
        a.sampling_identity_sha256
        != b.sampling_identity_sha256
    )

    assert not np.array_equal(
        a.images,
        b.images,
    )


def test_wrong_identity_fails_closed(
    tmp_path: Path,
):
    _persist(
        tmp_path
    )

    with pytest.raises(
        ValueError,
        match="dataset identity",
    ):
        sample_gmm_checkpoint(
            tmp_path,
            dataset_id="wrong_dataset",
            class_id=CLASS_ID,
            experimental_seed=SEED,
            sample_count=2,
        )


def test_missing_checkpoint_fails_before_sampling(
    tmp_path: Path,
):
    with pytest.raises(
        FileNotFoundError,
        match="metadata",
    ):
        sample_gmm_checkpoint(
            tmp_path,
            dataset_id=DATASET_ID,
            class_id=CLASS_ID,
            experimental_seed=SEED,
            sample_count=2,
        )


def test_tampered_checkpoint_fails_closed(
    tmp_path: Path,
):
    _persist(
        tmp_path
    )

    state_path = (
        tmp_path
        / "model_state.npz"
    )

    with state_path.open(
        "ab"
    ) as handle:
        handle.write(
            b"tamper"
        )

    with pytest.raises(
        ValueError,
        match="SHA-256 mismatch",
    ):
        sample_gmm_checkpoint(
            tmp_path,
            dataset_id=DATASET_ID,
            class_id=CLASS_ID,
            experimental_seed=SEED,
            sample_count=2,
        )
