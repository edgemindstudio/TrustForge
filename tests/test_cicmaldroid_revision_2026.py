"""Regression tests for the governed Paper 1 revision-2026 CIC adapter."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from trustforge.data.paper01_revision_2026.cicmaldroid import (
    CICMalDroidRevision2026,
)


CIC_COUNTS = (
    1253,
    2100,
    3904,
    2546,
    1795,
)


@pytest.fixture(scope="module")
def cic_source(
    tmp_path_factory,
) -> tuple[Path, Path]:
    """Create a contract-shaped synthetic CIC unsplit corpus."""
    root = tmp_path_factory.mktemp(
        "cicmaldroid_revision_2026"
    )

    source_csv = root / "cic.csv"
    feature_columns_path = root / "feature_columns.json"

    feature_columns = [
        f"feature_{index:03d}"
        for index in range(139)
    ]

    feature_columns_path.write_text(
        json.dumps(
            feature_columns,
            indent=2,
        ),
        encoding="utf-8",
    )

    labels = np.concatenate(
        [
            np.full(
                count,
                class_id,
                dtype=np.int32,
            )
            for class_id, count in enumerate(
                CIC_COUNTS,
                start=1,
            )
        ]
    )

    rows = len(labels)

    row_ids = np.arange(
        rows,
        dtype=np.float32,
    )[:, None]

    feature_ids = np.arange(
        139,
        dtype=np.float32,
    )[None, :]

    # Deterministic finite values with variation in every feature.
    features = (
        (
            row_ids
            * (
                feature_ids + 1.0
            )
        )
        % (
            97.0
            + (
                feature_ids % 17.0
            )
        )
    ).astype(
        np.float32,
    )

    frame = pd.DataFrame(
        features,
        columns=feature_columns,
    )

    frame["Class"] = labels

    frame.to_csv(
        source_csv,
        index=False,
    )

    return (
        source_csv,
        feature_columns_path,
    )


@pytest.fixture()
def dataset(
    cic_source,
) -> CICMalDroidRevision2026:
    source_csv, feature_columns_path = cic_source

    return CICMalDroidRevision2026(
        source_csv=source_csv,
        feature_columns_path=feature_columns_path,
        validate_hashes=False,
    )


def test_cic_revision_case3_partition_counts(dataset):
    assert dataset.split_size("train") == 8117
    assert dataset.split_size("validation") == 1158
    assert dataset.split_size("test") == 2323

    expected = {
        "train": np.array(
            [877, 1470, 2732, 1782, 1256],
            dtype=np.int64,
        ),
        "validation": np.array(
            [125, 210, 390, 254, 179],
            dtype=np.int64,
        ),
        "test": np.array(
            [251, 420, 782, 510, 360],
            dtype=np.int64,
        ),
    }

    for split, expected_counts in expected.items():
        labels = dataset.raw_labels(
            split
        )

        values, counts = np.unique(
            labels,
            return_counts=True,
        )

        assert np.array_equal(
            values,
            np.arange(
                1,
                6,
                dtype=np.int32,
            ),
        )

        assert np.array_equal(
            counts,
            expected_counts,
        )


def test_cic_revision_partition_is_disjoint_and_complete(dataset):
    train = dataset.partitions.train
    validation = dataset.partitions.validation
    test = dataset.partitions.test

    assert np.intersect1d(
        train,
        validation,
    ).size == 0

    assert np.intersect1d(
        train,
        test,
    ).size == 0

    assert np.intersect1d(
        validation,
        test,
    ).size == 0

    combined = np.concatenate(
        [
            train,
            validation,
            test,
        ]
    )

    assert np.array_equal(
        np.sort(combined),
        np.arange(
            11598,
            dtype=np.int64,
        ),
    )


def test_cic_revision_partition_is_deterministic(cic_source):
    source_csv, feature_columns_path = cic_source

    a = CICMalDroidRevision2026(
        source_csv,
        feature_columns_path,
        partition_seed=42,
        validate_hashes=False,
    )

    b = CICMalDroidRevision2026(
        source_csv,
        feature_columns_path,
        partition_seed=42,
        validate_hashes=False,
    )

    assert np.array_equal(
        a.partitions.train,
        b.partitions.train,
    )

    assert np.array_equal(
        a.partitions.validation,
        b.partitions.validation,
    )

    assert np.array_equal(
        a.partitions.test,
        b.partitions.test,
    )


def test_cic_revision_maps_source_labels_to_zero_based(dataset):
    for split in (
        "train",
        "validation",
        "test",
    ):
        raw = dataset.raw_labels(
            split
        )

        model = dataset.labels(
            split
        )

        assert np.array_equal(
            model,
            raw - 1,
        )

        assert np.array_equal(
            np.unique(model),
            np.arange(
                5,
                dtype=np.int32,
            ),
        )


def test_cic_revision_scaler_is_fit_on_revision_train_only(dataset):
    revised_train = dataset.arrays.features[
        dataset.partitions.train
    ]

    independent = StandardScaler()

    independent.fit(
        revised_train
    )

    assert np.array_equal(
        dataset.scaler.mean_,
        independent.mean_,
    )

    assert np.array_equal(
        dataset.scaler.var_,
        independent.var_,
    )

    assert np.array_equal(
        dataset.scaler.scale_,
        independent.scale_,
    )

    assert dataset.scaler.n_features_in_ == 139


def test_cic_revision_source_representation_matches_verified_semantics(
    dataset,
):
    positions = np.array(
        [0, 1, 2, 3],
        dtype=np.int64,
    )

    source_indices = dataset.partitions.train[
        positions
    ]

    raw = np.asarray(
        dataset.arrays.features[
            source_indices
        ],
        dtype=np.float32,
    )

    transformed = dataset.scaler.transform(
        raw
    )

    assert transformed.dtype == np.float32

    expected = np.tanh(
        transformed
    )

    assert expected.dtype == np.float32

    expected = np.pad(
        expected,
        (
            (0, 0),
            (0, 5),
        ),
        mode="constant",
        constant_values=0.0,
    )

    expected = expected.reshape(
        -1,
        12,
        12,
        1,
    )

    observed = dataset._source_batch(
        "train",
        positions,
    )

    assert np.array_equal(
        observed,
        expected,
    )

    flattened = observed.reshape(
        len(observed),
        144,
    )

    assert np.all(
        flattened[:, 139:] == 0.0
    )


def test_cic_revision_canonical_batch_contract(dataset):
    positions = np.array(
        [0, 1, 2, 3],
        dtype=np.int64,
    )

    images, labels = dataset.canonical_batch(
        "train",
        positions,
    )

    assert images.shape == (4, 40, 40, 3)
    assert images.dtype == np.float32
    assert np.isfinite(images).all()
    assert images.min() >= 0.0
    assert images.max() <= 1.0

    assert labels.shape == (4, 5)
    assert labels.dtype == np.float32

    assert np.array_equal(
        labels.sum(
            axis=1
        ),
        np.ones(
            4,
            dtype=np.float32,
        ),
    )

    # CIC is a scalar feature-grid representation and must therefore
    # remain exactly replicated across canonical RGB channels.
    assert np.array_equal(
        images[..., 0],
        images[..., 1],
    )

    assert np.array_equal(
        images[..., 1],
        images[..., 2],
    )


def test_cic_revision_rejects_out_of_bounds_batch_position(dataset):
    invalid = np.array(
        [
            dataset.split_size(
                "train"
            )
        ],
        dtype=np.int64,
    )

    with pytest.raises(
        IndexError,
    ):
        dataset.canonical_batch(
            "train",
            invalid,
        )
