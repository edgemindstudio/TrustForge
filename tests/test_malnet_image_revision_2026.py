"""Regression tests for the governed Paper 1 revision-2026 MalNet adapter."""

from __future__ import annotations

import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
import pytest

from trustforge.data.paper01_revision_2026.malnet_image import (
    MalNetImageRevision2026,
)
from trustforge.data.paper01_revision_2026.malnet_preparation import (
    prepare_malnet_index,
)


PROVIDER_MEMBERS = {
    "train": "split_info/type/1.0/train.txt",
    "validation": "split_info/type/1.0/val.txt",
    "test": "split_info/type/1.0/test.txt",
}


@pytest.fixture()
def malnet_fixture(
    tmp_path: Path,
):
    """Create a tiny provider-style MalNet corpus and preparation index."""

    rows = {
        "train": [
            "adware/family_a/AAA",
            "benign/benign/BBB",
            "trojan/family_t/CCC",
        ],
        "validation": [
            "benign/benign/DDD",
            "adware/family_b/EEE",
        ],
        "test": [
            "trojan/family_t/FFF",
            "adware/family_a/GGG",
            "benign/benign/HHH",
        ],
    }

    archive = (
        tmp_path
        / "split_info.zip"
    )

    with zipfile.ZipFile(
        archive,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as zf:
        for split, member in PROVIDER_MEMBERS.items():
            content = "\n".join(
                rows[split]
            ) + "\n"

            zf.writestr(
                member,
                content.encode("utf-8"),
            )

    image_root = (
        tmp_path
        / "malnet-images"
    )

    image_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_entries = []

    for split_rows in rows.values():
        all_entries.extend(
            split_rows
        )

    for sample_index, entry in enumerate(
        all_entries
    ):
        path = (
            image_root
            / f"{entry}.png"
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Genuine RGB fixture:
        # each channel follows a different deterministic pattern.
        y = np.arange(
            256,
            dtype=np.uint16,
        )[:, None]

        x = np.arange(
            256,
            dtype=np.uint16,
        )[None, :]

        red = (
            x
            + sample_index
        ) % 256

        green = (
            y
            + 2 * sample_index
        ) % 256

        blue = (
            x
            + y
            + 3 * sample_index
        ) % 256

        rgb = np.stack(
            [
                np.broadcast_to(
                    red,
                    (256, 256),
                ),
                np.broadcast_to(
                    green,
                    (256, 256),
                ),
                blue,
            ],
            axis=-1,
        ).astype(
            np.uint8
        )

        Image.fromarray(
            rgb,
            mode="RGB",
        ).save(
            path,
            format="PNG",
        )

    preparation_dir = (
        tmp_path
        / "prepared"
    )

    expected_counts = {
        split: len(values)
        for split, values in rows.items()
    }

    prepare_malnet_index(
        archive,
        preparation_dir,
        expected_archive_sha256=None,
        expected_counts=expected_counts,
        provider_members=PROVIDER_MEMBERS,
    )

    mapping_path = (
        tmp_path
        / "mapping.json"
    )

    class_to_id = {
        "adware": 0,
        "benign": 1,
        "trojan": 2,
    }

    id_to_class = {
        str(class_id): label
        for label, class_id in class_to_id.items()
    }

    mapping_path.write_text(
        json.dumps(
            {
                "dataset": "MalNet-Image",
                "paper": "Paper 1 revision 2026",
                "task": "type-level classification",
                "num_classes": 3,
                "class_to_id": class_to_id,
                "id_to_class": id_to_class,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = MalNetImageRevision2026(
        image_root=image_root,
        preparation_dir=preparation_dir,
        mapping_path=mapping_path,
        validate_hashes=False,
        validate_production_contract=False,
    )

    return (
        dataset,
        rows,
        image_root,
    )


def test_malnet_revision_preserves_provider_split_counts(
    malnet_fixture,
):
    dataset, rows, _ = malnet_fixture

    for split, expected in rows.items():
        assert dataset.split_size(
            split
        ) == len(expected)


def test_malnet_revision_prepared_artifacts_are_memory_mapped(
    malnet_fixture,
):
    dataset, _, _ = malnet_fixture

    for split in (
        "train",
        "validation",
        "test",
    ):
        prepared = dataset.splits[
            split
        ]

        assert isinstance(
            prepared.text_bytes,
            np.memmap,
        )

        assert isinstance(
            prepared.offsets,
            np.memmap,
        )

        assert prepared.offsets.dtype == np.int64


def test_malnet_revision_random_access_preserves_provider_order(
    malnet_fixture,
):
    dataset, rows, _ = malnet_fixture

    positions = np.array(
        [2, 0, 1],
        dtype=np.int64,
    )

    observed = dataset.entries(
        "train",
        positions,
    )

    expected = (
        rows["train"][2],
        rows["train"][0],
        rows["train"][1],
    )

    assert observed == expected


def test_malnet_revision_maps_provider_labels_to_governed_ids(
    malnet_fixture,
):
    dataset, _, _ = malnet_fixture

    positions = np.array(
        [0, 1, 2],
        dtype=np.int64,
    )

    strings = dataset.string_labels(
        "train",
        positions,
    )

    assert strings == (
        "adware",
        "benign",
        "trojan",
    )

    ids = dataset.labels(
        "train",
        positions,
    )

    assert np.array_equal(
        ids,
        np.array(
            [0, 1, 2],
            dtype=np.int32,
        ),
    )

    one_hot = dataset.one_hot_labels(
        "train",
        positions,
    )

    assert one_hot.shape == (
        3,
        3,
    )

    assert one_hot.dtype == np.float32

    assert np.array_equal(
        one_hot.sum(
            axis=1
        ),
        np.ones(
            3,
            dtype=np.float32,
        ),
    )


def test_malnet_revision_resolves_provider_entries_to_png_paths(
    malnet_fixture,
):
    dataset, rows, image_root = malnet_fixture

    positions = np.array(
        [0, 2],
        dtype=np.int64,
    )

    paths = dataset.source_paths(
        "test",
        positions,
    )

    assert paths == (
        image_root
        / f"{rows['test'][0]}.png",
        image_root
        / f"{rows['test'][2]}.png",
    )


def test_malnet_revision_source_batch_is_native_rgb_uint8(
    malnet_fixture,
):
    dataset, _, _ = malnet_fixture

    positions = np.array(
        [0, 1],
        dtype=np.int64,
    )

    source = dataset._source_batch(
        "train",
        positions,
    )

    assert source.shape == (
        2,
        256,
        256,
        3,
    )

    assert source.dtype == np.uint8

    # The synthetic fixture is intentionally genuine RGB,
    # not grayscale replicated into three channels.
    assert not np.array_equal(
        source[..., 0],
        source[..., 1],
    )

    assert not np.array_equal(
        source[..., 1],
        source[..., 2],
    )


def test_malnet_revision_canonical_batch_contract(
    malnet_fixture,
):
    dataset, _, _ = malnet_fixture

    positions = np.array(
        [0, 1],
        dtype=np.int64,
    )

    images, labels = dataset.canonical_batch(
        "train",
        positions,
    )

    assert images.shape == (
        2,
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

    assert labels.shape == (
        2,
        3,
    )

    assert labels.dtype == np.float32

    # MalNet RGB channels must be preserved as distinct channels.
    assert not np.array_equal(
        images[..., 0],
        images[..., 1],
    )

    assert not np.array_equal(
        images[..., 1],
        images[..., 2],
    )


def test_malnet_revision_rejects_out_of_bounds_batch_position(
    malnet_fixture,
):
    dataset, _, _ = malnet_fixture

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


def test_malnet_revision_rejects_unknown_split(
    malnet_fixture,
):
    dataset, _, _ = malnet_fixture

    with pytest.raises(
        ValueError,
        match="unsupported MalNet logical split",
    ):
        dataset.entries(
            "unknown",
            np.array(
                [0],
                dtype=np.int64,
            ),
        )
