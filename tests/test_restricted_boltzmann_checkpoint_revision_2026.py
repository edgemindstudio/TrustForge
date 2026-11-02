"""Focused tests for governed revision-2026 RBM checkpoint persistence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from trustforge.models.paper01_revision_2026 import (
    restricted_boltzmann_checkpoint as checkpoint_module,
)
from trustforge.models.paper01_revision_2026.restricted_boltzmann_checkpoint import (
    CHECKPOINT_SELECTION_RULE,
    METADATA_FILENAME,
    STATE_FILENAME,
    STATE_KEYS,
    load_rbm_checkpoint,
    persist_rbm_checkpoint,
)
from trustforge.models.paper01_revision_2026.restricted_boltzmann_training import (
    train_shared_conditional_rbm,
)
from trustforge.provenance import (
    canonical_json_bytes,
    sha256_file,
)


DATASET_ID = "tiny-rbm-checkpoint-test"
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


def _persist(
    root: Path,
    training_result,
):
    return persist_rbm_checkpoint(
        root,
        dataset_id=DATASET_ID,
        experimental_seed=EXPERIMENTAL_SEED,
        class_mapping=CLASS_MAPPING,
        training_result=training_result,
    )


def _load(
    root: Path,
):
    return load_rbm_checkpoint(
        root,
        requested_dataset_id=DATASET_ID,
        requested_experimental_seed=EXPERIMENTAL_SEED,
        requested_class_mapping=CLASS_MAPPING,
    )


def _read_metadata(
    root: Path,
):
    return json.loads(
        (
            root
            / METADATA_FILENAME
        ).read_text(
            encoding="utf-8"
        )
    )


def _write_canonical_metadata(
    root: Path,
    metadata,
):
    (
        root
        / METADATA_FILENAME
    ).write_bytes(
        canonical_json_bytes(
            metadata
        )
        + b"\n"
    )


def _rewrite_state(
    root: Path,
    state,
):
    with (
        root
        / STATE_FILENAME
    ).open(
        "wb"
    ) as handle:
        np.savez(
            handle,
            **state,
        )


def _refresh_state_hash(
    root: Path,
):
    metadata = _read_metadata(
        root
    )

    metadata[
        "state_sha256"
    ] = sha256_file(
        root
        / STATE_FILENAME
    )

    _write_canonical_metadata(
        root,
        metadata,
    )


def test_roundtrip_preserves_exact_state(
    tmp_path,
    training_result,
):
    persisted = _persist(
        tmp_path,
        training_result,
    )

    loaded = _load(
        tmp_path
    )

    assert persisted.state_path.name == STATE_FILENAME
    assert persisted.metadata_path.name == METADATA_FILENAME

    original = training_result.model.state_dict()
    restored = loaded.model.state_dict()

    assert tuple(
        original
    ) == STATE_KEYS

    for key in STATE_KEYS:
        np.testing.assert_array_equal(
            original[
                key
            ],
            restored[
                key
            ],
        )


def test_state_npz_contains_exact_governed_keys(
    tmp_path,
    training_result,
):
    _persist(
        tmp_path,
        training_result,
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


def test_state_arrays_are_exact_float64(
    tmp_path,
    training_result,
):
    _persist(
        tmp_path,
        training_result,
    )

    with np.load(
        tmp_path
        / STATE_FILENAME,
        allow_pickle=False,
    ) as archive:
        for key in STATE_KEYS:
            assert archive[
                key
            ].dtype == np.float64


def test_state_shapes_are_governed(
    tmp_path,
    training_result,
):
    _persist(
        tmp_path,
        training_result,
    )

    with np.load(
        tmp_path
        / STATE_FILENAME,
        allow_pickle=False,
    ) as archive:
        assert archive["W"].shape == (
            4800,
            128,
        )
        assert archive["a"].shape == (
            4800,
        )
        assert archive["b"].shape == (
            128,
        )
        assert archive["U"].shape == (
            2,
            128,
        )


def test_class_mapping_is_identity_bound(
    tmp_path,
    training_result,
):
    _persist(
        tmp_path,
        training_result,
    )

    with pytest.raises(
        ValueError,
        match="class mapping",
    ):
        load_rbm_checkpoint(
            tmp_path,
            requested_dataset_id=DATASET_ID,
            requested_experimental_seed=EXPERIMENTAL_SEED,
            requested_class_mapping=(
                "class_1",
                "class_0",
            ),
        )


def test_dataset_identity_mismatch_fails_closed(
    tmp_path,
    training_result,
):
    _persist(
        tmp_path,
        training_result,
    )

    with pytest.raises(
        ValueError,
        match="dataset",
    ):
        load_rbm_checkpoint(
            tmp_path,
            requested_dataset_id="different-dataset",
            requested_experimental_seed=EXPERIMENTAL_SEED,
            requested_class_mapping=CLASS_MAPPING,
        )


def test_seed_identity_mismatch_fails_closed(
    tmp_path,
    training_result,
):
    _persist(
        tmp_path,
        training_result,
    )

    with pytest.raises(
        ValueError,
        match="seed",
    ):
        load_rbm_checkpoint(
            tmp_path,
            requested_dataset_id=DATASET_ID,
            requested_experimental_seed=43,
            requested_class_mapping=CLASS_MAPPING,
        )


def test_state_sha256_is_bound_and_verified(
    tmp_path,
    training_result,
):
    _persist(
        tmp_path,
        training_result,
    )

    metadata = _read_metadata(
        tmp_path
    )

    assert metadata[
        "state_sha256"
    ] == sha256_file(
        tmp_path
        / STATE_FILENAME
    )

    with (
        tmp_path
        / STATE_FILENAME
    ).open(
        "ab"
    ) as handle:
        handle.write(
            b"tamper"
        )

    with pytest.raises(
        ValueError,
        match="SHA-256",
    ):
        _load(
            tmp_path
        )


def test_object_state_is_rejected_with_pickle_disabled(
    tmp_path,
    training_result,
):
    _persist(
        tmp_path,
        training_result,
    )

    state = training_result.model.state_dict()

    state[
        "a"
    ] = np.array(
        [
            object()
        ],
        dtype=object,
    )

    _rewrite_state(
        tmp_path,
        state,
    )

    _refresh_state_hash(
        tmp_path
    )

    with pytest.raises(
        ValueError,
    ):
        _load(
            tmp_path
        )


def test_malformed_state_key_set_fails_closed(
    tmp_path,
    training_result,
):
    _persist(
        tmp_path,
        training_result,
    )

    state = training_result.model.state_dict()

    state.pop(
        "b"
    )

    _rewrite_state(
        tmp_path,
        state,
    )

    _refresh_state_hash(
        tmp_path
    )

    with pytest.raises(
        ValueError,
        match="unexpected or missing keys",
    ):
        _load(
            tmp_path
        )


def test_wrong_state_dtype_fails_closed(
    tmp_path,
    training_result,
):
    _persist(
        tmp_path,
        training_result,
    )

    state = training_result.model.state_dict()

    state[
        "a"
    ] = state[
        "a"
    ].astype(
        np.float32
    )

    _rewrite_state(
        tmp_path,
        state,
    )

    _refresh_state_hash(
        tmp_path
    )

    with pytest.raises(
        TypeError,
        match="float64",
    ):
        _load(
            tmp_path
        )


def test_nonfinite_state_fails_closed(
    tmp_path,
    training_result,
):
    _persist(
        tmp_path,
        training_result,
    )

    state = training_result.model.state_dict()

    state[
        "a"
    ] = state[
        "a"
    ].copy()

    state[
        "a"
    ][0] = np.nan

    _rewrite_state(
        tmp_path,
        state,
    )

    _refresh_state_hash(
        tmp_path
    )

    with pytest.raises(
        ValueError,
        match="non-finite",
    ):
        _load(
            tmp_path
        )


def test_malformed_metadata_fails_closed(
    tmp_path,
    training_result,
):
    _persist(
        tmp_path,
        training_result,
    )

    (
        tmp_path
        / METADATA_FILENAME
    ).write_text(
        "{not-json}\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="metadata JSON",
    ):
        _load(
            tmp_path
        )


@pytest.mark.parametrize(
    "missing_name",
    [
        STATE_FILENAME,
        METADATA_FILENAME,
    ],
)
def test_incomplete_checkpoint_pair_fails_closed(
    tmp_path,
    training_result,
    missing_name,
):
    _persist(
        tmp_path,
        training_result,
    )

    (
        tmp_path
        / missing_name
    ).unlink()

    with pytest.raises(
        FileNotFoundError,
    ):
        _load(
            tmp_path
        )


def test_authoritative_overwrite_is_refused(
    tmp_path,
    training_result,
):
    _persist(
        tmp_path,
        training_result,
    )

    with pytest.raises(
        FileExistsError,
    ):
        _persist(
            tmp_path,
            training_result,
        )


def test_selected_step_is_exact_final_governed_update(
    tmp_path,
    training_result,
):
    _persist(
        tmp_path,
        training_result,
    )

    metadata = _read_metadata(
        tmp_path
    )

    assert metadata[
        "selected_step"
    ] == metadata[
        "maximum_updates"
    ]

    assert metadata[
        "checkpoint_selection_rule"
    ] == CHECKPOINT_SELECTION_RULE


def test_reconstruction_mse_does_not_select_checkpoint(
    tmp_path,
    training_result,
):
    _persist(
        tmp_path,
        training_result,
    )

    metadata = _read_metadata(
        tmp_path
    )

    history = metadata[
        "validation_history"
    ]

    minimum_mse_step = min(
        history,
        key=lambda item: item[
            "reconstruction_mse"
        ],
    )[
        "step"
    ]

    assert metadata[
        "selected_step"
    ] == metadata[
        "maximum_updates"
    ]

    assert metadata[
        "checkpoint_selection_rule"
    ] == "final_governed_update"

    assert all(
        "selected" not in record
        for record in history
    )

    assert minimum_mse_step in metadata[
        "validation_schedule"
    ]


def test_validation_history_is_complete_diagnostic_provenance(
    tmp_path,
    training_result,
):
    _persist(
        tmp_path,
        training_result,
    )

    metadata = _read_metadata(
        tmp_path
    )

    assert metadata[
        "validation_schedule"
    ] == [
        0,
        1,
        2,
    ]

    assert [
        record[
            "step"
        ]
        for record in metadata[
            "validation_history"
        ]
    ] == [
        0,
        1,
        2,
    ]

    assert all(
        record[
            "finite_state"
        ]
        is True
        for record in metadata[
            "validation_history"
        ]
    )


def test_metadata_with_nonfinal_selected_step_fails_closed(
    tmp_path,
    training_result,
):
    _persist(
        tmp_path,
        training_result,
    )

    metadata = _read_metadata(
        tmp_path
    )

    metadata[
        "selected_step"
    ] = 1

    _write_canonical_metadata(
        tmp_path,
        metadata,
    )

    with pytest.raises(
        ValueError,
        match="final governed update",
    ):
        _load(
            tmp_path
        )


def test_metadata_with_best_checkpoint_rule_fails_closed(
    tmp_path,
    training_result,
):
    _persist(
        tmp_path,
        training_result,
    )

    metadata = _read_metadata(
        tmp_path
    )

    metadata[
        "checkpoint_selection_rule"
    ] = "best_validation_mse"

    _write_canonical_metadata(
        tmp_path,
        metadata,
    )

    with pytest.raises(
        ValueError,
        match="selection rule",
    ):
        _load(
            tmp_path
        )


def test_noncanonical_metadata_bytes_fail_closed(
    tmp_path,
    training_result,
):
    _persist(
        tmp_path,
        training_result,
    )

    metadata = _read_metadata(
        tmp_path
    )

    (
        tmp_path
        / METADATA_FILENAME
    ).write_text(
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
        match="canonical",
    ):
        _load(
            tmp_path
        )


def test_historical_rbm_best_file_is_not_a_revision_checkpoint(
    tmp_path,
):
    historical = (
        tmp_path
        / "RBM_best.weights.h5"
    )

    historical.write_bytes(
        b"historical-only"
    )

    with pytest.raises(
        FileNotFoundError,
    ):
        load_rbm_checkpoint(
            tmp_path,
            requested_dataset_id=DATASET_ID,
            requested_experimental_seed=EXPERIMENTAL_SEED,
            requested_class_mapping=CLASS_MAPPING,
        )


def test_publication_order_is_state_before_metadata(
    tmp_path,
    training_result,
    monkeypatch,
):
    observed = []

    original = checkpoint_module._link_no_overwrite

    def recording_link(
        temporary,
        destination,
    ):
        observed.append(
            destination.name
        )

        return original(
            temporary,
            destination,
        )

    monkeypatch.setattr(
        checkpoint_module,
        "_link_no_overwrite",
        recording_link,
    )

    _persist(
        tmp_path,
        training_result,
    )

    assert observed == [
        STATE_FILENAME,
        METADATA_FILENAME,
    ]


def test_postpublication_roundtrip_validation_occurs(
    tmp_path,
    training_result,
    monkeypatch,
):
    observed = []

    original = checkpoint_module.load_rbm_checkpoint

    def recording_load(
        *args,
        **kwargs,
    ):
        observed.append(
            True
        )

        return original(
            *args,
            **kwargs,
        )

    monkeypatch.setattr(
        checkpoint_module,
        "load_rbm_checkpoint",
        recording_load,
    )

    result = checkpoint_module.persist_rbm_checkpoint(
        tmp_path,
        dataset_id=DATASET_ID,
        experimental_seed=EXPERIMENTAL_SEED,
        class_mapping=CLASS_MAPPING,
        training_result=training_result,
    )

    assert observed == [
        True
    ]

    assert result.metadata[
        "dataset_id"
    ] == DATASET_ID


def test_class_mapping_length_must_match_U_rows(
    tmp_path,
    training_result,
):
    with pytest.raises(
        ValueError,
        match="length",
    ):
        persist_rbm_checkpoint(
            tmp_path,
            dataset_id=DATASET_ID,
            experimental_seed=EXPERIMENTAL_SEED,
            class_mapping=(
                "only_one_class",
            ),
            training_result=training_result,
        )


def test_duplicate_class_labels_are_rejected(
    tmp_path,
    training_result,
):
    with pytest.raises(
        ValueError,
        match="duplicate",
    ):
        persist_rbm_checkpoint(
            tmp_path,
            dataset_id=DATASET_ID,
            experimental_seed=EXPERIMENTAL_SEED,
            class_mapping=(
                "same",
                "same",
            ),
            training_result=training_result,
        )
