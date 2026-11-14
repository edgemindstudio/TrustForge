"""Tests for Paper 1 revision-2026 governed GMM checkpoints."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from trustforge.models.paper01_revision_2026.gaussian_mixture_checkpoint import (
    METADATA_FILENAME,
    SCHEMA_TYPE,
    SCHEMA_VERSION,
    STATE_FILENAME,
    STATE_KEYS,
    load_gmm_checkpoint,
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
from trustforge.models.paper01_revision_2026.gaussian_mixture_training import (
    GMMClassTrainingResult,
    GMMValidationCheckpoint,
)
from trustforge.models.paper01_revision_2026.gaussian_mixture_training_budget import (
    training_budget,
)
from trustforge.provenance import (
    canonical_json_bytes,
    sha256_file,
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
                8.0,
                7.0,
                6.0,
                5.0,
                4.0,
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
        selected_validation_nll=4.0,
        optimizer_steps_completed=100,
    )


def _execution_evidence() -> dict:
    return {
        "schema": "trustforge.paper01.gmm.execution-binding.v1",
        "source_freeze_sha256": "a" * 64,
        "expected_source_commit": "b" * 40,
        "historical_git_baseline": "c" * 40,
        "implementation_parent_revision": "d" * 40,
        "source_files_sha256": {
            "scripts/train_paper01_gmm_revision_production.py": "e" * 64,
        },
        "governed_experimental_seeds": list(range(42, 49)),
        "dataset_adapter": "CICMalDroidRevision2026",
        "dataset_source_paths": {"cic_source_csv": "/source.csv"},
    }


def test_legacy_checkpoint_without_execution_evidence_loads(
    tmp_path: Path,
):
    persist_gmm_checkpoint(
        tmp_path,
        dataset_id=DATASET_ID,
        class_id=CLASS_ID,
        experimental_seed=SEED,
        training_result=_training_result(),
        validation_class_count=2,
    )
    loaded = load_gmm_checkpoint(
        tmp_path,
        requested_dataset_id=DATASET_ID,
        requested_class_id=CLASS_ID,
        requested_experimental_seed=SEED,
    )
    assert "execution_evidence" not in loaded.metadata


def test_checkpoint_records_execution_evidence(
    tmp_path: Path,
):
    evidence = _execution_evidence()
    persisted = persist_gmm_checkpoint(
        tmp_path,
        dataset_id=DATASET_ID,
        class_id=CLASS_ID,
        experimental_seed=SEED,
        training_result=_training_result(),
        validation_class_count=2,
        execution_evidence=evidence,
    )
    assert persisted.metadata["execution_evidence"] == evidence
    loaded = load_gmm_checkpoint(
        tmp_path,
        requested_dataset_id=DATASET_ID,
        requested_class_id=CLASS_ID,
        requested_experimental_seed=SEED,
    )
    assert loaded.metadata["execution_evidence"] == evidence


def test_invalid_execution_evidence_rejected(
    tmp_path: Path,
):
    evidence = _execution_evidence()
    evidence["governed_experimental_seeds"] = [41, 42]
    with pytest.raises(ValueError, match="seed policy"):
        persist_gmm_checkpoint(
            tmp_path,
            dataset_id=DATASET_ID,
            class_id=CLASS_ID,
            experimental_seed=SEED,
            training_result=_training_result(),
            validation_class_count=2,
            execution_evidence=evidence,
        )
    assert not (tmp_path / STATE_FILENAME).exists()
    assert not (tmp_path / METADATA_FILENAME).exists()


def test_checkpoint_roundtrip(
    tmp_path: Path,
):
    result = _training_result()

    persisted = persist_gmm_checkpoint(
        tmp_path,
        dataset_id=DATASET_ID,
        class_id=CLASS_ID,
        experimental_seed=SEED,
        training_result=result,
        validation_class_count=2,
    )

    assert persisted.state_path == (
        tmp_path
        / STATE_FILENAME
    )

    assert persisted.metadata_path == (
        tmp_path
        / METADATA_FILENAME
    )

    assert persisted.state_path.is_file()
    assert persisted.metadata_path.is_file()

    metadata = persisted.metadata

    assert metadata[
        "schema_version"
    ] == SCHEMA_VERSION

    assert metadata[
        "schema_type"
    ] == SCHEMA_TYPE

    assert metadata[
        "dataset_id"
    ] == DATASET_ID

    assert metadata[
        "class_id"
    ] == CLASS_ID

    assert metadata[
        "experimental_seed"
    ] == SEED

    assert metadata[
        "selected_step"
    ] == 100

    assert metadata[
        "selected_validation_nll"
    ] == 4.0

    assert metadata[
        "state_sha256"
    ] == sha256_file(
        persisted.state_path
    )

    expected_state = result.model.state_arrays()
    actual_state = persisted.model.state_arrays()

    for key in STATE_KEYS:
        np.testing.assert_array_equal(
            actual_state[
                key
            ],
            expected_state[
                key
            ],
        )


def test_state_npz_has_exact_keys_shapes_and_dtypes(
    tmp_path: Path,
):
    persist_gmm_checkpoint(
        tmp_path,
        dataset_id=DATASET_ID,
        class_id=CLASS_ID,
        experimental_seed=SEED,
        training_result=_training_result(),
        validation_class_count=2,
    )

    with np.load(
        tmp_path
        / STATE_FILENAME,
        allow_pickle=False,
    ) as archive:
        assert set(
            archive.files
        ) == set(
            STATE_KEYS
        )

        assert archive[
            "logits"
        ].shape == (
            5,
        )

        assert archive[
            "means"
        ].shape == (
            5,
            4800,
        )

        assert archive[
            "raw_scales"
        ].shape == (
            5,
            4800,
        )

        for key in STATE_KEYS:
            assert archive[
                key
            ].dtype == np.float64


def test_metadata_is_exact_canonical_json_plus_one_newline(
    tmp_path: Path,
):
    persisted = persist_gmm_checkpoint(
        tmp_path,
        dataset_id=DATASET_ID,
        class_id=CLASS_ID,
        experimental_seed=SEED,
        training_result=_training_result(),
        validation_class_count=2,
    )

    raw = persisted.metadata_path.read_bytes()

    assert raw.endswith(
        b"\n"
    )

    assert not raw.endswith(
        b"\n\n"
    )

    assert raw == (
        canonical_json_bytes(
            persisted.metadata
        )
        + b"\n"
    )


def test_checkpoint_refuses_state_overwrite(
    tmp_path: Path,
):
    (
        tmp_path
        / STATE_FILENAME
    ).write_bytes(
        b"existing"
    )

    with pytest.raises(
        FileExistsError,
        match="refusing to overwrite",
    ):
        persist_gmm_checkpoint(
            tmp_path,
            dataset_id=DATASET_ID,
            class_id=CLASS_ID,
            experimental_seed=SEED,
            training_result=_training_result(),
            validation_class_count=2,
        )


def test_checkpoint_refuses_metadata_overwrite(
    tmp_path: Path,
):
    (
        tmp_path
        / METADATA_FILENAME
    ).write_text(
        "{}\n",
        encoding="utf-8",
    )

    with pytest.raises(
        FileExistsError,
        match="refusing to overwrite",
    ):
        persist_gmm_checkpoint(
            tmp_path,
            dataset_id=DATASET_ID,
            class_id=CLASS_ID,
            experimental_seed=SEED,
            training_result=_training_result(),
            validation_class_count=2,
        )


def test_identity_mismatch_fails_closed(
    tmp_path: Path,
):
    persist_gmm_checkpoint(
        tmp_path,
        dataset_id=DATASET_ID,
        class_id=CLASS_ID,
        experimental_seed=SEED,
        training_result=_training_result(),
        validation_class_count=2,
    )

    with pytest.raises(
        ValueError,
        match="dataset identity",
    ):
        load_gmm_checkpoint(
            tmp_path,
            requested_dataset_id="other_dataset",
            requested_class_id=CLASS_ID,
            requested_experimental_seed=SEED,
        )

    with pytest.raises(
        ValueError,
        match="class identity",
    ):
        load_gmm_checkpoint(
            tmp_path,
            requested_dataset_id=DATASET_ID,
            requested_class_id=CLASS_ID + 1,
            requested_experimental_seed=SEED,
        )

    with pytest.raises(
        ValueError,
        match="experimental seed",
    ):
        load_gmm_checkpoint(
            tmp_path,
            requested_dataset_id=DATASET_ID,
            requested_class_id=CLASS_ID,
            requested_experimental_seed=SEED + 1,
        )


def test_state_hash_tamper_fails_closed(
    tmp_path: Path,
):
    persisted = persist_gmm_checkpoint(
        tmp_path,
        dataset_id=DATASET_ID,
        class_id=CLASS_ID,
        experimental_seed=SEED,
        training_result=_training_result(),
        validation_class_count=2,
    )

    with persisted.state_path.open(
        "ab"
    ) as handle:
        handle.write(
            b"tamper"
        )

    with pytest.raises(
        ValueError,
        match="SHA-256 mismatch",
    ):
        load_gmm_checkpoint(
            tmp_path,
            requested_dataset_id=DATASET_ID,
            requested_class_id=CLASS_ID,
            requested_experimental_seed=SEED,
        )


def test_noncanonical_metadata_fails_closed(
    tmp_path: Path,
):
    persisted = persist_gmm_checkpoint(
        tmp_path,
        dataset_id=DATASET_ID,
        class_id=CLASS_ID,
        experimental_seed=SEED,
        training_result=_training_result(),
        validation_class_count=2,
    )

    metadata = persisted.metadata

    persisted.metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="canonical JSON",
    ):
        load_gmm_checkpoint(
            tmp_path
        )


def test_metadata_selected_checkpoint_tamper_fails_closed(
    tmp_path: Path,
):
    persisted = persist_gmm_checkpoint(
        tmp_path,
        dataset_id=DATASET_ID,
        class_id=CLASS_ID,
        experimental_seed=SEED,
        training_result=_training_result(),
        validation_class_count=2,
    )

    metadata = dict(
        persisted.metadata
    )

    metadata[
        "selected_step"
    ] = 80

    persisted.metadata_path.write_bytes(
        canonical_json_bytes(
            metadata
        )
        + b"\n"
    )

    with pytest.raises(
        ValueError,
        match="selected checkpoint",
    ):
        load_gmm_checkpoint(
            tmp_path
        )


def test_invalid_training_result_selection_is_rejected_before_publish(
    tmp_path: Path,
):
    result = _training_result()

    invalid = replace(
        result,
        selected_step=80,
        selected_validation_nll=5.0,
    )

    with pytest.raises(
        ValueError,
        match="selected checkpoint",
    ):
        persist_gmm_checkpoint(
            tmp_path,
            dataset_id=DATASET_ID,
            class_id=CLASS_ID,
            experimental_seed=SEED,
            training_result=invalid,
            validation_class_count=2,
        )

    assert not (
        tmp_path
        / STATE_FILENAME
    ).exists()

    assert not (
        tmp_path
        / METADATA_FILENAME
    ).exists()


def test_missing_metadata_is_not_valid_checkpoint(
    tmp_path: Path,
):
    np.savez(
        tmp_path
        / STATE_FILENAME,
        logits=np.zeros(
            5,
            dtype=np.float64,
        ),
        means=np.zeros(
            (
                5,
                4800,
            ),
            dtype=np.float64,
        ),
        raw_scales=np.zeros(
            (
                5,
                4800,
            ),
            dtype=np.float64,
        ),
    )

    with pytest.raises(
        FileNotFoundError,
        match="metadata",
    ):
        load_gmm_checkpoint(
            tmp_path
        )


def test_metadata_requires_exact_schema_keys(
    tmp_path: Path,
):
    persisted = persist_gmm_checkpoint(
        tmp_path,
        dataset_id=DATASET_ID,
        class_id=CLASS_ID,
        experimental_seed=SEED,
        training_result=_training_result(),
        validation_class_count=2,
    )

    metadata = dict(
        persisted.metadata
    )

    metadata[
        "unexpected"
    ] = 1

    persisted.metadata_path.write_bytes(
        canonical_json_bytes(
            metadata
        )
        + b"\n"
    )

    with pytest.raises(
        ValueError,
        match="exactly match governed schema",
    ):
        load_gmm_checkpoint(
            tmp_path
        )
