"""Regression tests for the governed Paper 1 revision-2026 MaleX adapter."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from PIL import Image
import pytest

from trustforge.data.paper01_revision_2026.malex import (
    MALEX_CONFLICT_IDENTITIES,
    MaleXRevision2026,
)


def _sha(
    value: int,
) -> bytes:
    return (
        f"{value:064x}"
    ).encode(
        "ascii"
    )


@pytest.fixture()
def malex_fixture(
    tmp_path: Path,
):
    """Create a tiny contiguous-HDF5 MaleX-style corpus."""

    label_archive = (
        tmp_path
        / "MaleXv1-fixture.hdf5"
    )

    conflict = MALEX_CONFLICT_IDENTITIES[
        0
    ].encode(
        "ascii"
    )

    benign_ids = np.asarray(
        [
            _sha(1),
            _sha(2),
            conflict,
            _sha(3),
            _sha(4),
            _sha(5),
            _sha(6),
            _sha(7),
            _sha(8),
            _sha(9),
        ],
        dtype="S64",
    )

    malware_ids = np.asarray(
        [
            _sha(101),
            _sha(102),
            _sha(103),
            conflict,
            _sha(104),
            _sha(105),
            _sha(106),
            _sha(107),
            _sha(108),
            _sha(109),
            _sha(110),
            _sha(111),
            _sha(112),
            _sha(113),
            _sha(114),
            _sha(115),
            _sha(116),
            _sha(117),
            _sha(118),
            _sha(119),
        ],
        dtype="S64",
    )

    with h5py.File(
        label_archive,
        "w",
    ) as h5:
        h5.create_dataset(
            "ben_sha256s",
            data=benign_ids,
        )

        h5.create_dataset(
            "mal_sha256s",
            data=malware_ids,
        )

    benign_root = (
        tmp_path
        / "benign"
    )

    malware_root = (
        tmp_path
        / "malware"
    )

    benign_root.mkdir()
    malware_root.mkdir()

    for class_index, (
        ids,
        root,
    ) in enumerate(
        (
            (
                benign_ids,
                benign_root,
            ),
            (
                malware_ids,
                malware_root,
            ),
        )
    ):
        for sample_index, identity in enumerate(
            ids
        ):
            text = bytes(
                identity
            ).decode(
                "ascii"
            )

            # One grayscale source image per identity.
            y = np.arange(
                256,
                dtype=np.uint16,
            )[:, None]

            x = np.arange(
                256,
                dtype=np.uint16,
            )[None, :]

            array = (
                x
                + y
                + sample_index
                + 17 * class_index
            ) % 256

            Image.fromarray(
                array.astype(
                    np.uint8
                ),
                mode="L",
            ).save(
                root
                / f"{text}_img_256.png",
                format="PNG",
            )

    dataset = MaleXRevision2026(
        label_archive=label_archive,
        benign_byteplot_root=benign_root,
        malware_byteplot_root=malware_root,
        validate_hashes=False,
        validate_production_contract=False,
    )

    return (
        dataset,
        benign_ids,
        malware_ids,
        conflict,
        benign_root,
        malware_root,
    )


def test_malex_revision_uses_direct_identity_memmaps(
    malex_fixture,
):
    dataset, *_ = malex_fixture

    assert isinstance(
        dataset._benign_source,
        np.memmap,
    )

    assert isinstance(
        dataset._malware_source,
        np.memmap,
    )

    assert dataset._benign_source.dtype == np.dtype(
        "S64"
    )

    assert dataset._malware_source.dtype == np.dtype(
        "S64"
    )


def test_malex_revision_excludes_cross_class_conflict(
    malex_fixture,
):
    (
        dataset,
        benign_ids,
        malware_ids,
        conflict,
        _,
        _,
    ) = malex_fixture

    expected_benign = int(
        np.sum(
            benign_ids != conflict
        )
    )

    expected_malware = int(
        np.sum(
            malware_ids != conflict
        )
    )

    assert dataset._benign_positions.shape == (
        expected_benign,
    )

    assert dataset._malware_positions.shape == (
        expected_malware,
    )

    all_indices = np.arange(
        dataset._logical_labels.shape[0],
        dtype=np.int64,
    )

    observed = dataset._identities_for_corpus_indices(
        all_indices
    )

    assert conflict not in set(
        observed.tolist()
    )


def test_malex_revision_case3_splits_cover_post_exclusion_universe(
    malex_fixture,
):
    dataset, *_ = malex_fixture

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

    union = np.concatenate(
        (
            train,
            validation,
            test,
        )
    )

    expected = np.arange(
        dataset._logical_labels.shape[0],
        dtype=np.int64,
    )

    assert np.array_equal(
        np.sort(
            union
        ),
        expected,
    )


def test_malex_revision_split_is_deterministic(
    tmp_path: Path,
    malex_fixture,
):
    (
        first,
        _,
        _,
        _,
        benign_root,
        malware_root,
    ) = malex_fixture

    second = MaleXRevision2026(
        label_archive=first.label_archive,
        benign_byteplot_root=benign_root,
        malware_byteplot_root=malware_root,
        validate_hashes=False,
        validate_production_contract=False,
    )

    assert np.array_equal(
        first.partitions.train,
        second.partitions.train,
    )

    assert np.array_equal(
        first.partitions.validation,
        second.partitions.validation,
    )

    assert np.array_equal(
        first.partitions.test,
        second.partitions.test,
    )


def test_malex_revision_labels_are_batch_local_and_binary(
    malex_fixture,
):
    dataset, *_ = malex_fixture

    split = "train"

    positions = np.arange(
        min(
            4,
            dataset.split_size(
                split
            ),
        ),
        dtype=np.int64,
    )

    labels = dataset.labels(
        split,
        positions,
    )

    strings = dataset.string_labels(
        split,
        positions,
    )

    one_hot = dataset.one_hot_labels(
        split,
        positions,
    )

    assert labels.dtype == np.int32

    assert set(
        labels.tolist()
    ).issubset(
        {
            0,
            1,
        }
    )

    assert set(
        strings
    ).issubset(
        {
            "benign",
            "malware",
        }
    )

    assert one_hot.shape == (
        positions.shape[0],
        2,
    )

    assert one_hot.dtype == np.float32

    assert np.allclose(
        one_hot.sum(
            axis=1
        ),
        1.0,
    )

    assert np.array_equal(
        np.argmax(
            one_hot,
            axis=1,
        ).astype(
            np.int32
        ),
        labels,
    )


def test_malex_revision_source_paths_match_class_roots(
    malex_fixture,
):
    dataset, *_ = malex_fixture

    for split in (
        "train",
        "validation",
        "test",
    ):
        size = dataset.split_size(
            split
        )

        if size == 0:
            continue

        positions = np.arange(
            min(
                3,
                size,
            ),
            dtype=np.int64,
        )

        labels = dataset.labels(
            split,
            positions,
        )

        paths = dataset.source_paths(
            split,
            positions,
        )

        for class_id, path in zip(
            labels.tolist(),
            paths,
        ):
            expected_root = (
                dataset.benign_byteplot_root
                if class_id == 0
                else dataset.malware_byteplot_root
            )

            assert path.parent == expected_root

            assert path.is_file()


def test_malex_revision_source_batch_is_grayscale_uint8(
    malex_fixture,
):
    dataset, *_ = malex_fixture

    positions = np.arange(
        min(
            2,
            dataset.split_size(
                "train"
            ),
        ),
        dtype=np.int64,
    )

    source = dataset._source_batch(
        "train",
        positions,
    )

    assert source.shape == (
        positions.shape[0],
        256,
        256,
        1,
    )

    assert source.dtype == np.uint8


def test_malex_revision_canonical_batch_contract(
    malex_fixture,
):
    dataset, *_ = malex_fixture

    positions = np.arange(
        min(
            2,
            dataset.split_size(
                "train"
            ),
        ),
        dtype=np.int64,
    )

    images, labels = dataset.canonical_batch(
        "train",
        positions,
    )

    assert images.shape == (
        positions.shape[0],
        40,
        40,
        3,
    )

    assert images.dtype == np.float32

    assert np.isfinite(
        images
    ).all()

    assert images.min() >= 0.0

    assert images.max() <= 1.0

    assert np.array_equal(
        images[..., 0],
        images[..., 1],
    )

    assert np.array_equal(
        images[..., 1],
        images[..., 2],
    )

    assert labels.shape == (
        positions.shape[0],
        2,
    )

    assert labels.dtype == np.float32


def test_malex_revision_rejects_out_of_bounds_position(
    malex_fixture,
):
    dataset, *_ = malex_fixture

    invalid = np.asarray(
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


def test_malex_revision_rejects_unknown_split(
    malex_fixture,
):
    dataset, *_ = malex_fixture

    with pytest.raises(
        ValueError,
        match="unsupported MaleX logical split",
    ):
        dataset.identities(
            "unknown",
            np.asarray(
                [0],
                dtype=np.int64,
            ),
        )
