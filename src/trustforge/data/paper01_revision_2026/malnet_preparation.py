"""MalNet-Image preparation indexer for Paper 1 revision 2026.

The official provider split archive remains the scientific authority.

This module creates derived random-access preparation artifacts:

- normalized UTF-8 split text preserving provider order;
- int64 byte-offset arrays;
- compact integrity manifest.

The derived index never redefines provider membership, ordering, or labels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Mapping
import zipfile

import numpy as np


MALNET_SPLIT_ARCHIVE_SHA256 = (
    "74e3615aa848c27dc06bb9ea2ae53e415f4b842fdce8dc45ba50849db8851492"
)

MALNET_PROVIDER_MEMBERS = {
    "train": "split_info/type/1.0/train.txt",
    "validation": "split_info/type/1.0/val.txt",
    "test": "split_info/type/1.0/test.txt",
}

MALNET_PROVIDER_COUNTS = {
    "train": 883_416,
    "validation": 126_202,
    "test": 252_406,
}


@dataclass(frozen=True)
class MalNetPreparedSplitRecord:
    """Integrity metadata for one prepared MalNet provider split."""

    split: str
    provider_member: str
    provider_rows: int
    prepared_rows: int
    prepared_text_file: str
    prepared_text_bytes: int
    prepared_text_sha256: str
    offsets_file: str
    offsets_sha256: str
    offsets_dtype: str
    offsets_length: int
    first_offset: int
    terminal_offset: int


def _sha256_file(
    path: Path,
    *,
    chunk_size: int = 1024 * 1024,
) -> str:
    digest = sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def _validate_logical_entry(
    line: bytes,
    *,
    split: str,
    row_number: int,
) -> None:
    try:
        text = line.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"MalNet {split} row {row_number} is not valid UTF-8"
        ) from exc

    parts = text.split("/")

    if len(parts) != 3:
        raise ValueError(
            f"Malformed MalNet {split} row {row_number}: {text!r}"
        )

    if not all(parts):
        raise ValueError(
            f"Empty MalNet path component in {split} row "
            f"{row_number}: {text!r}"
        )


def _prepare_one_split(
    zf: zipfile.ZipFile,
    *,
    split: str,
    member: str,
    output_dir: Path,
    expected_rows: int | None,
) -> MalNetPreparedSplitRecord:
    text_path = output_dir / f"{split}.txt"
    offsets_path = output_dir / f"{split}_offsets.npy"

    if text_path.exists():
        raise FileExistsError(
            f"refusing to overwrite prepared MalNet text: {text_path}"
        )

    if offsets_path.exists():
        raise FileExistsError(
            f"refusing to overwrite prepared MalNet offsets: {offsets_path}"
        )

    offsets = [0]
    rows = 0

    with zf.open(member, "r") as source, text_path.open("wb") as target:
        for raw_line in source:
            logical = raw_line.rstrip(b"\r\n")

            if not logical:
                continue

            rows += 1

            _validate_logical_entry(
                logical,
                split=split,
                row_number=rows,
            )

            target.write(logical)
            target.write(b"\n")

            offsets.append(
                target.tell()
            )

    if expected_rows is not None and rows != expected_rows:
        raise ValueError(
            f"unexpected MalNet {split} row count: "
            f"{rows} != {expected_rows}"
        )

    offsets_array = np.asarray(
        offsets,
        dtype=np.int64,
    )

    if offsets_array.shape != (rows + 1,):
        raise RuntimeError(
            f"unexpected MalNet {split} offset shape: "
            f"{offsets_array.shape}"
        )

    if offsets_array[0] != 0:
        raise RuntimeError(
            f"MalNet {split} offsets do not begin at zero"
        )

    if np.any(
        np.diff(offsets_array) <= 0
    ):
        raise RuntimeError(
            f"MalNet {split} offsets are not strictly increasing"
        )

    text_size = text_path.stat().st_size

    if int(offsets_array[-1]) != text_size:
        raise RuntimeError(
            f"MalNet {split} terminal offset does not equal text size"
        )

    np.save(
        offsets_path,
        offsets_array,
        allow_pickle=False,
    )

    mapped = np.load(
        offsets_path,
        allow_pickle=False,
        mmap_mode="r",
    )

    if mapped.dtype != np.int64:
        raise RuntimeError(
            f"MalNet {split} prepared offsets are not int64"
        )

    if mapped.shape != (rows + 1,):
        raise RuntimeError(
            f"MalNet {split} mapped offset shape mismatch"
        )

    # Verify exact prepared-line reconstruction for every logical position.
    with text_path.open("rb") as prepared:
        for index in range(rows):
            start = int(mapped[index])
            stop = int(mapped[index + 1])

            prepared.seek(start)

            raw = prepared.read(
                stop - start
            )

            if not raw.endswith(b"\n"):
                raise RuntimeError(
                    f"MalNet {split} prepared row {index} "
                    "does not end with governed newline"
                )

            logical = raw[:-1]

            _validate_logical_entry(
                logical,
                split=split,
                row_number=index + 1,
            )

    return MalNetPreparedSplitRecord(
        split=split,
        provider_member=member,
        provider_rows=rows,
        prepared_rows=rows,
        prepared_text_file=text_path.name,
        prepared_text_bytes=text_size,
        prepared_text_sha256=_sha256_file(
            text_path
        ),
        offsets_file=offsets_path.name,
        offsets_sha256=_sha256_file(
            offsets_path
        ),
        offsets_dtype=str(mapped.dtype),
        offsets_length=int(
            mapped.shape[0]
        ),
        first_offset=int(
            mapped[0]
        ),
        terminal_offset=int(
            mapped[-1]
        ),
    )


def prepare_malnet_index(
    archive_path: Path | str,
    output_dir: Path | str,
    *,
    expected_archive_sha256: str | None = MALNET_SPLIT_ARCHIVE_SHA256,
    expected_counts: Mapping[str, int] | None = MALNET_PROVIDER_COUNTS,
    provider_members: Mapping[str, str] = MALNET_PROVIDER_MEMBERS,
) -> Path:
    """Create governed random-access MalNet split preparation artifacts.

    Returns the generated manifest path.

    Passing ``expected_archive_sha256=None`` and custom ``expected_counts`` is
    intended for self-contained regression fixtures only.
    """

    archive = Path(
        archive_path
    ).expanduser().resolve()

    destination = Path(
        output_dir
    ).expanduser().resolve()

    if not archive.is_file():
        raise FileNotFoundError(
            f"missing MalNet provider split archive: {archive}"
        )

    if expected_archive_sha256 is not None:
        observed_archive_hash = _sha256_file(
            archive
        )

        if observed_archive_hash != expected_archive_sha256:
            raise ValueError(
                "unexpected MalNet provider archive SHA-256: "
                f"{observed_archive_hash}"
            )
    else:
        observed_archive_hash = _sha256_file(
            archive
        )

    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_path = destination / "manifest.json"

    if manifest_path.exists():
        raise FileExistsError(
            f"refusing to overwrite MalNet preparation manifest: "
            f"{manifest_path}"
        )

    records = []

    with zipfile.ZipFile(
        archive,
        "r",
    ) as zf:
        names = set(
            zf.namelist()
        )

        for split, member in provider_members.items():
            if member not in names:
                raise FileNotFoundError(
                    f"missing MalNet provider member: {member}"
                )

            expected_rows = (
                None
                if expected_counts is None
                else int(
                    expected_counts[split]
                )
            )

            records.append(
                _prepare_one_split(
                    zf,
                    split=split,
                    member=member,
                    output_dir=destination,
                    expected_rows=expected_rows,
                )
            )

    manifest = {
        "dataset": "MalNet-Image",
        "paper": "Paper 1 revision 2026",
        "task": "type-level classification",
        "authority": (
            "Derived preparation infrastructure only; "
            "provider split archive remains authoritative."
        ),
        "provider_archive": str(
            archive
        ),
        "provider_archive_sha256": observed_archive_hash,
        "splits": {
            record.split: asdict(
                record
            )
            for record in records
        },
    }

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=False,
        )
        + "\n",
        encoding="utf-8",
    )

    return manifest_path


__all__ = [
    "MALNET_PROVIDER_COUNTS",
    "MALNET_PROVIDER_MEMBERS",
    "MALNET_SPLIT_ARCHIVE_SHA256",
    "MalNetPreparedSplitRecord",
    "prepare_malnet_index",
]
