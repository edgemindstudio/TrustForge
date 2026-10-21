"""Regression tests for the governed MalNet preparation-index layer."""

from __future__ import annotations

import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from trustforge.data.paper01_revision_2026.malnet_preparation import (
    prepare_malnet_index,
)


PROVIDER_MEMBERS = {
    "train": "split_info/type/1.0/train.txt",
    "validation": "split_info/type/1.0/val.txt",
    "test": "split_info/type/1.0/test.txt",
}


def _write_fixture_archive(
    path: Path,
) -> tuple[Path, dict[str, list[str]]]:
    """Create a tiny provider-style MalNet split archive."""

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

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with zipfile.ZipFile(
        path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as zf:
        for split, member in PROVIDER_MEMBERS.items():
            # Exercise normalization of provider line terminators and
            # preservation of non-empty logical entries.
            content = "\r\n".join(
                rows[split]
            ) + "\r\n"

            zf.writestr(
                member,
                content.encode("utf-8"),
            )

    return path, rows


def _prepare_fixture(
    tmp_path: Path,
):
    archive, rows = _write_fixture_archive(
        tmp_path / "split_info.zip"
    )

    output_dir = (
        tmp_path
        / "prepared"
    )

    expected_counts = {
        split: len(values)
        for split, values in rows.items()
    }

    manifest = prepare_malnet_index(
        archive,
        output_dir,
        expected_archive_sha256=None,
        expected_counts=expected_counts,
        provider_members=PROVIDER_MEMBERS,
    )

    return (
        archive,
        output_dir,
        manifest,
        rows,
    )


def test_malnet_preparation_preserves_provider_order(tmp_path):
    _, output_dir, _, rows = _prepare_fixture(
        tmp_path
    )

    for split, expected in rows.items():
        observed = (
            output_dir
            / f"{split}.txt"
        ).read_text(
            encoding="utf-8"
        ).splitlines()

        assert observed == expected


def test_malnet_preparation_offsets_reconstruct_exact_entries(tmp_path):
    _, output_dir, _, rows = _prepare_fixture(
        tmp_path
    )

    for split, expected in rows.items():
        text_path = (
            output_dir
            / f"{split}.txt"
        )

        offsets = np.load(
            output_dir
            / f"{split}_offsets.npy",
            allow_pickle=False,
            mmap_mode="r",
        )

        assert isinstance(
            offsets,
            np.memmap,
        )

        assert offsets.dtype == np.int64

        assert offsets.shape == (
            len(expected) + 1,
        )

        assert int(
            offsets[0]
        ) == 0

        assert int(
            offsets[-1]
        ) == text_path.stat().st_size

        assert np.all(
            np.diff(
                offsets
            ) > 0
        )

        with text_path.open(
            "rb"
        ) as handle:
            reconstructed = []

            for index in range(
                len(expected)
            ):
                start = int(
                    offsets[index]
                )
                stop = int(
                    offsets[index + 1]
                )

                handle.seek(
                    start
                )

                raw = handle.read(
                    stop - start
                )

                assert raw.endswith(
                    b"\n"
                )

                reconstructed.append(
                    raw[:-1].decode(
                        "utf-8"
                    )
                )

        assert reconstructed == expected


def test_malnet_preparation_manifest_contract(tmp_path):
    archive, output_dir, manifest_path, rows = _prepare_fixture(
        tmp_path
    )

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    assert manifest["dataset"] == "MalNet-Image"

    assert manifest[
        "task"
    ] == "type-level classification"

    assert Path(
        manifest[
            "provider_archive"
        ]
    ) == archive.resolve()

    for split, expected in rows.items():
        record = manifest[
            "splits"
        ][split]

        assert record[
            "provider_member"
        ] == PROVIDER_MEMBERS[split]

        assert record[
            "provider_rows"
        ] == len(expected)

        assert record[
            "prepared_rows"
        ] == len(expected)

        assert record[
            "offsets_dtype"
        ] == "int64"

        assert record[
            "offsets_length"
        ] == len(expected) + 1

        assert record[
            "first_offset"
        ] == 0

        assert record[
            "terminal_offset"
        ] == (
            output_dir
            / f"{split}.txt"
        ).stat().st_size

        assert len(
            record[
                "prepared_text_sha256"
            ]
        ) == 64

        assert len(
            record[
                "offsets_sha256"
            ]
        ) == 64


def test_malnet_preparation_rejects_wrong_expected_count(tmp_path):
    archive, _ = _write_fixture_archive(
        tmp_path
        / "split_info.zip"
    )

    with pytest.raises(
        ValueError,
        match="unexpected MalNet train row count",
    ):
        prepare_malnet_index(
            archive,
            tmp_path / "prepared",
            expected_archive_sha256=None,
            expected_counts={
                "train": 999,
                "validation": 2,
                "test": 3,
            },
            provider_members=PROVIDER_MEMBERS,
        )


def test_malnet_preparation_rejects_malformed_entry(tmp_path):
    archive = (
        tmp_path
        / "bad_split_info.zip"
    )

    with zipfile.ZipFile(
        archive,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as zf:
        zf.writestr(
            PROVIDER_MEMBERS["train"],
            b"adware/family_a/AAA\n"
            b"malformed-entry\n",
        )

        zf.writestr(
            PROVIDER_MEMBERS["validation"],
            b"benign/benign/BBB\n",
        )

        zf.writestr(
            PROVIDER_MEMBERS["test"],
            b"trojan/family_t/CCC\n",
        )

    with pytest.raises(
        ValueError,
        match="Malformed MalNet train row",
    ):
        prepare_malnet_index(
            archive,
            tmp_path / "prepared",
            expected_archive_sha256=None,
            expected_counts=None,
            provider_members=PROVIDER_MEMBERS,
        )


def test_malnet_preparation_refuses_overwrite(tmp_path):
    archive, output_dir, _, rows = _prepare_fixture(
        tmp_path
    )

    expected_counts = {
        split: len(values)
        for split, values in rows.items()
    }

    with pytest.raises(
        FileExistsError,
    ):
        prepare_malnet_index(
            archive,
            output_dir,
            expected_archive_sha256=None,
            expected_counts=expected_counts,
            provider_members=PROVIDER_MEMBERS,
        )
