"""Revision-2026 MalNet-Image data adapter.

The adapter preserves the official provider type-level 1.0 partitions exactly
while using the governed preparation-index layer for scalable random access.

Key properties:

- official provider train / validation / test membership is preserved;
- prepared split text and int64 offsets are memory-mapped;
- only requested split entries are decoded;
- provider string type labels remain authoritative scientific metadata;
- TrustForge numerical class IDs come only from the governed mapping artifact;
- source PNGs are decoded as RGB;
- canonicalization converts uint8 [0,255] RGB to float32 [0,1] and resizes
  256 x 256 x 3 -> 40 x 40 x 3;
- no complete path universe or canonical image corpus is materialized.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image
import tensorflow as tf

from common.canonicalization import (
    CANONICAL_SHAPE,
    SourceNumericRange,
    canonicalize_images,
)


MALNET_NUM_CLASSES = 47
MALNET_SOURCE_SHAPE = (256, 256, 3)
MALNET_CANONICAL_SHAPE = CANONICAL_SHAPE

MALNET_MAPPING_SHA256 = (
    "84406057e188a35a98ec7237275bf1bce22df037d627c40d2ded5d735020b253"
)

MALNET_PROVIDER_ARCHIVE_SHA256 = (
    "74e3615aa848c27dc06bb9ea2ae53e415f4b842fdce8dc45ba50849db8851492"
)

MALNET_EXPECTED_SPLIT_COUNTS = {
    "train": 883_416,
    "validation": 126_202,
    "test": 252_406,
}

SplitName = Literal[
    "train",
    "validation",
    "test",
]


@dataclass(frozen=True)
class MalNetPreparedSplit:
    """Memory-mapped preparation artifacts for one provider split."""

    text_path: Path
    text_bytes: np.memmap
    offsets: np.memmap
    rows: int


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


class MalNetImageRevision2026:
    """Governed revision-2026 MalNet-Image dataset view."""

    def __init__(
        self,
        image_root: Path | str,
        preparation_dir: Path | str,
        mapping_path: Path | str,
        *,
        validate_hashes: bool = True,
        validate_production_contract: bool = True,
    ) -> None:
        self.image_root = Path(
            image_root
        ).expanduser().resolve()

        self.preparation_dir = Path(
            preparation_dir
        ).expanduser().resolve()

        self.mapping_path = Path(
            mapping_path
        ).expanduser().resolve()

        self.validate_hashes = bool(
            validate_hashes
        )

        self.validate_production_contract = bool(
            validate_production_contract
        )

        self._validate_required_paths()

        self.mapping = self._load_mapping()

        self.class_to_id = dict(
            self.mapping["class_to_id"]
        )

        self.id_to_class = {
            int(class_id): label
            for class_id, label in self.mapping[
                "id_to_class"
            ].items()
        }

        self.num_classes = int(
            self.mapping["num_classes"]
        )

        self._validate_mapping_contract()

        self.manifest = self._load_manifest()

        self.splits = {
            split: self._load_prepared_split(
                split
            )
            for split in (
                "train",
                "validation",
                "test",
            )
        }

        self._validate_preparation_contract()

    def _validate_required_paths(self) -> None:
        if not self.image_root.is_dir():
            raise FileNotFoundError(
                f"missing MalNet image root: {self.image_root}"
            )

        if not self.preparation_dir.is_dir():
            raise FileNotFoundError(
                "missing MalNet preparation directory: "
                f"{self.preparation_dir}"
            )

        if not self.mapping_path.is_file():
            raise FileNotFoundError(
                f"missing MalNet mapping artifact: {self.mapping_path}"
            )

        manifest = (
            self.preparation_dir
            / "manifest.json"
        )

        if not manifest.is_file():
            raise FileNotFoundError(
                f"missing MalNet preparation manifest: {manifest}"
            )

    def _load_mapping(self) -> dict:
        if self.validate_hashes:
            observed = _sha256_file(
                self.mapping_path
            )

            if observed != MALNET_MAPPING_SHA256:
                raise ValueError(
                    "unexpected MalNet mapping SHA-256: "
                    f"{observed}"
                )

        obj = json.loads(
            self.mapping_path.read_text(
                encoding="utf-8"
            )
        )

        if not isinstance(obj, dict):
            raise TypeError(
                "MalNet mapping artifact must be a JSON object"
            )

        return obj

    def _validate_mapping_contract(self) -> None:
        required = {
            "num_classes",
            "class_to_id",
            "id_to_class",
        }

        missing = (
            required
            - set(
                self.mapping
            )
        )

        if missing:
            raise ValueError(
                "MalNet mapping artifact missing fields: "
                f"{sorted(missing)}"
            )

        if not isinstance(
            self.class_to_id,
            dict,
        ):
            raise TypeError(
                "MalNet class_to_id must be a mapping"
            )

        if not isinstance(
            self.id_to_class,
            dict,
        ):
            raise TypeError(
                "MalNet id_to_class must be a mapping"
            )

        if len(
            self.class_to_id
        ) != self.num_classes:
            raise ValueError(
                "MalNet class_to_id size does not match num_classes"
            )

        if len(
            self.id_to_class
        ) != self.num_classes:
            raise ValueError(
                "MalNet id_to_class size does not match num_classes"
            )

        expected_ids = list(
            range(
                self.num_classes
            )
        )

        observed_ids = sorted(
            int(value)
            for value in self.class_to_id.values()
        )

        if observed_ids != expected_ids:
            raise ValueError(
                "MalNet class IDs are not contiguous zero-based IDs"
            )

        for label, class_id in self.class_to_id.items():
            if not isinstance(
                label,
                str,
            ):
                raise TypeError(
                    "MalNet class labels must be strings"
                )

            if self.id_to_class[
                int(class_id)
            ] != label:
                raise ValueError(
                    "MalNet mapping is not bijective"
                )

        if self.validate_production_contract:
            if self.num_classes != MALNET_NUM_CLASSES:
                raise ValueError(
                    "unexpected production MalNet class count: "
                    f"{self.num_classes}"
                )

            labels = list(
                self.class_to_id.keys()
            )

            if labels != sorted(
                labels
            ):
                raise ValueError(
                    "production MalNet mapping is not lexicographic"
                )

    def _load_manifest(self) -> dict:
        path = (
            self.preparation_dir
            / "manifest.json"
        )

        obj = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        if not isinstance(
            obj,
            dict,
        ):
            raise TypeError(
                "MalNet preparation manifest must be a JSON object"
            )

        if obj.get(
            "dataset"
        ) != "MalNet-Image":
            raise ValueError(
                "unexpected MalNet preparation dataset identity"
            )

        splits = obj.get(
            "splits"
        )

        if not isinstance(
            splits,
            dict,
        ):
            raise TypeError(
                "MalNet preparation manifest splits must be a mapping"
            )

        return obj

    def _load_prepared_split(
        self,
        split: SplitName,
    ) -> MalNetPreparedSplit:
        try:
            record = self.manifest[
                "splits"
            ][split]
        except KeyError as exc:
            raise ValueError(
                f"MalNet manifest missing split: {split}"
            ) from exc

        text_path = (
            self.preparation_dir
            / record[
                "prepared_text_file"
            ]
        )

        offsets_path = (
            self.preparation_dir
            / record[
                "offsets_file"
            ]
        )

        if not text_path.is_file():
            raise FileNotFoundError(
                f"missing prepared MalNet text: {text_path}"
            )

        if not offsets_path.is_file():
            raise FileNotFoundError(
                f"missing prepared MalNet offsets: {offsets_path}"
            )

        if self.validate_hashes:
            text_hash = _sha256_file(
                text_path
            )

            if text_hash != record[
                "prepared_text_sha256"
            ]:
                raise ValueError(
                    f"MalNet {split} prepared-text SHA-256 mismatch"
                )

            offsets_hash = _sha256_file(
                offsets_path
            )

            if offsets_hash != record[
                "offsets_sha256"
            ]:
                raise ValueError(
                    f"MalNet {split} offsets SHA-256 mismatch"
                )

        text_bytes = np.memmap(
            text_path,
            dtype=np.uint8,
            mode="r",
        )

        offsets = np.load(
            offsets_path,
            allow_pickle=False,
            mmap_mode="r",
        )

        if not isinstance(
            offsets,
            np.memmap,
        ):
            raise RuntimeError(
                f"MalNet {split} offsets are not memory-mapped"
            )

        if offsets.dtype != np.int64:
            raise TypeError(
                f"MalNet {split} offsets are not int64"
            )

        rows = int(
            record[
                "prepared_rows"
            ]
        )

        if offsets.shape != (
            rows + 1,
        ):
            raise ValueError(
                f"MalNet {split} offset shape mismatch: "
                f"{offsets.shape}"
            )

        if int(
            offsets[0]
        ) != 0:
            raise ValueError(
                f"MalNet {split} offsets do not begin at zero"
            )

        if int(
            offsets[-1]
        ) != int(
            text_bytes.shape[0]
        ):
            raise ValueError(
                f"MalNet {split} terminal offset mismatch"
            )

        return MalNetPreparedSplit(
            text_path=text_path,
            text_bytes=text_bytes,
            offsets=offsets,
            rows=rows,
        )

    def _validate_preparation_contract(self) -> None:
        if self.validate_production_contract:
            archive_hash = self.manifest.get(
                "provider_archive_sha256"
            )

            if archive_hash != MALNET_PROVIDER_ARCHIVE_SHA256:
                raise ValueError(
                    "unexpected production MalNet provider archive hash"
                )

        for split in (
            "train",
            "validation",
            "test",
        ):
            prepared = self.splits[
                split
            ]

            record = self.manifest[
                "splits"
            ][split]

            if int(
                record[
                    "provider_rows"
                ]
            ) != prepared.rows:
                raise ValueError(
                    f"MalNet {split} provider/prepared row mismatch"
                )

            if self.validate_production_contract:
                expected = MALNET_EXPECTED_SPLIT_COUNTS[
                    split
                ]

                if prepared.rows != expected:
                    raise ValueError(
                        f"unexpected production MalNet {split} "
                        f"row count: {prepared.rows}"
                    )

    def _prepared_split(
        self,
        split: SplitName,
    ) -> MalNetPreparedSplit:
        try:
            return self.splits[
                split
            ]
        except KeyError as exc:
            raise ValueError(
                f"unsupported MalNet logical split: {split!r}"
            ) from exc

    def entries(
        self,
        split: SplitName,
        logical_positions: np.ndarray,
    ) -> tuple[str, ...]:
        """Return provider entries for requested logical positions only."""

        prepared = self._prepared_split(
            split
        )

        positions = np.asarray(
            logical_positions,
            dtype=np.int64,
        )

        if positions.ndim != 1:
            raise ValueError(
                "logical_positions must be one-dimensional"
            )

        if np.any(
            positions < 0
        ) or np.any(
            positions >= prepared.rows
        ):
            raise IndexError(
                "logical batch position outside split bounds"
            )

        entries = []

        for position in positions:
            index = int(
                position
            )

            start = int(
                prepared.offsets[
                    index
                ]
            )

            stop = int(
                prepared.offsets[
                    index + 1
                ]
            )

            raw = bytes(
                prepared.text_bytes[
                    start:stop
                ]
            )

            if not raw.endswith(
                b"\n"
            ):
                raise RuntimeError(
                    f"MalNet {split} prepared entry {index} "
                    "does not end with governed newline"
                )

            entry = raw[:-1].decode(
                "utf-8"
            )

            parts = entry.split(
                "/"
            )

            if len(
                parts
            ) != 3 or not all(
                parts
            ):
                raise RuntimeError(
                    f"Malformed prepared MalNet entry: {entry!r}"
                )

            entries.append(
                entry
            )

        return tuple(
            entries
        )

    def string_labels(
        self,
        split: SplitName,
        logical_positions: np.ndarray,
    ) -> tuple[str, ...]:
        """Return provider type strings for requested positions only."""

        positions = np.asarray(
            logical_positions,
            dtype=np.int64,
        )

        return tuple(
            entry.split(
                "/",
                1,
            )[0]
            for entry in self.entries(
                split,
                positions,
            )
        )

    def labels(
        self,
        split: SplitName,
        logical_positions: np.ndarray,
    ) -> np.ndarray:
        """Return governed model IDs for requested positions only."""

        string_labels = self.string_labels(
            split,
            logical_positions,
        )

        try:
            ids = [
                self.class_to_id[
                    label
                ]
                for label in string_labels
            ]
        except KeyError as exc:
            raise ValueError(
                f"MalNet provider label missing from governed mapping: "
                f"{exc.args[0]!r}"
            ) from exc

        return np.asarray(
            ids,
            dtype=np.int32,
        )

    def one_hot_labels(
        self,
        split: SplitName,
        logical_positions: np.ndarray,
    ) -> np.ndarray:
        """Return float32 one-hot labels for requested positions only."""

        ids = self.labels(
            split,
            logical_positions,
        )

        return tf.one_hot(
            ids,
            depth=self.num_classes,
            dtype=tf.float32,
        ).numpy()

    def source_paths(
        self,
        split: SplitName,
        logical_positions: np.ndarray,
    ) -> tuple[Path, ...]:
        """Resolve requested provider entries to source PNG paths."""

        return tuple(
            self.image_root
            / f"{entry}.png"
            for entry in self.entries(
                split,
                logical_positions,
            )
        )

    def _source_batch(
        self,
        split: SplitName,
        logical_positions: np.ndarray,
    ) -> np.ndarray:
        """Decode one requested MalNet RGB uint8 batch."""

        paths = self.source_paths(
            split,
            logical_positions,
        )

        images = []

        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(
                    f"missing MalNet source image: {path}"
                )

            with Image.open(
                path
            ) as image:
                if image.format != "PNG":
                    raise ValueError(
                        f"MalNet source image is not PNG: {path}"
                    )

                if image.mode != "RGB":
                    raise ValueError(
                        f"MalNet source image is not RGB: "
                        f"{path} mode={image.mode!r}"
                    )

                array = np.asarray(
                    image,
                    dtype=np.uint8,
                )

            if array.shape != MALNET_SOURCE_SHAPE:
                raise ValueError(
                    f"unexpected MalNet source image shape: "
                    f"{path} shape={array.shape}"
                )

            images.append(
                array
            )

        if not images:
            return np.empty(
                (
                    0,
                    *MALNET_SOURCE_SHAPE,
                ),
                dtype=np.uint8,
            )

        return np.stack(
            images,
            axis=0,
        )

    def canonical_batch(
        self,
        split: SplitName,
        logical_positions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return canonical image and one-hot-label batch."""

        positions = np.asarray(
            logical_positions,
            dtype=np.int64,
        )

        source = self._source_batch(
            split,
            positions,
        )

        images = canonicalize_images(
            source,
            source_range=SourceNumericRange.UINT8_255,
        )

        labels = self.one_hot_labels(
            split,
            positions,
        )

        if tuple(
            images.shape[1:]
        ) != MALNET_CANONICAL_SHAPE:
            raise RuntimeError(
                "MalNet canonical batch has unexpected shape: "
                f"{images.shape}"
            )

        if images.dtype != np.float32:
            raise RuntimeError(
                "MalNet canonical batch is not float32"
            )

        return (
            images,
            labels.astype(
                np.float32,
                copy=False,
            ),
        )

    def split_size(
        self,
        split: SplitName,
    ) -> int:
        """Return provider-defined sample count for one split."""

        return int(
            self._prepared_split(
                split
            ).rows
        )


__all__ = [
    "MALNET_CANONICAL_SHAPE",
    "MALNET_EXPECTED_SPLIT_COUNTS",
    "MALNET_MAPPING_SHA256",
    "MALNET_NUM_CLASSES",
    "MALNET_PROVIDER_ARCHIVE_SHA256",
    "MALNET_SOURCE_SHAPE",
    "MalNetImageRevision2026",
    "MalNetPreparedSplit",
]
