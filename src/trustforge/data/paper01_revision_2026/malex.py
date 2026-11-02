"""Revision-2026 MaleX data adapter.

The adapter is bound to the validated MaleX v1 identity/label HDF5 artifact and
the selected 256 x 256 grayscale byteplot representation.

Key properties:

- authoritative HDF5 SHA-256 identities remain the source of sample identity;
- the exact HDF5 artifact is hash-validated in production mode;
- contiguous S64 HDF5 identity datasets are accessed through direct read-only
  memory maps using offsets obtained from h5py;
- all 19 cross-class SHA-256 conflicts are excluded from both classes before
  partitioning;
- the governed logical corpus preserves HDF5 benign order followed by HDF5
  malware order after exclusion;
- Case-3 70/10/20 stratified partitioning uses seed 42;
- only requested PNG files are decoded;
- canonicalization converts uint8 grayscale byteplots to 40 x 40 x 3
  float32 [0,1];
- raw source artifacts are never modified.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
from PIL import Image
import tensorflow as tf

from common.canonicalization import (
    CANONICAL_SHAPE,
    SourceNumericRange,
    canonicalize_images,
)
from common.partitioning import (
    DEFAULT_PARTITION_SEED,
    split_unsplit_corpus,
)


MALEX_NUM_CLASSES = 2
MALEX_SOURCE_SHAPE = (256, 256, 1)
MALEX_CANONICAL_SHAPE = CANONICAL_SHAPE

MALEX_LABEL_ARCHIVE_SHA256 = (
    "0ef4285cc14d1edbf6f937f783bc9a3b896bff70e762907cf556d41413f9f0ff"
)

MALEX_LABEL_ARCHIVE_SIZE = 253_462_319

MALEX_BENIGN_SOURCE_COUNT = 179_725
MALEX_MALWARE_SOURCE_COUNT = 864_669

MALEX_POST_EXCLUSION_BENIGN = 179_706
MALEX_POST_EXCLUSION_MALWARE = 864_650
MALEX_POST_EXCLUSION_TOTAL = 1_044_356

MALEX_EXPECTED_SPLIT_COUNTS = {
    "train": 731_049,
    "validation": 104_435,
    "test": 208_872,
}

MALEX_EXPECTED_CLASS_COUNTS = {
    "train": {
        0: 125_794,
        1: 605_255,
    },
    "validation": {
        0: 17_970,
        1: 86_465,
    },
    "test": {
        0: 35_942,
        1: 172_930,
    },
}

MALEX_EXPECTED_INDEX_SHA256 = {
    "train": (
        "8c31521b7c2151d2c14bd3a9a3025ca5881bf241e27aaadc998279a9def3b32e"
    ),
    "validation": (
        "e0b6d6d63060f2b6bb54f244aa11ba3b5fd4e8455e2536a49e355bff1841d9ac"
    ),
    "test": (
        "c60e118991f4db59a30dd312c9c906a3e9fc8234b0a3fd743009b65097c4d8a7"
    ),
}

MALEX_EXPECTED_MEMBERSHIP_SHA256 = {
    "train": (
        "4c051c7e5ac388ab12e017c1f34eb1b93e650cded71de08abcad923bda7ebcf2"
    ),
    "validation": (
        "5fcc103c2657b811b1a85b30d40370db3a6459da9cb4f36bb9e9dec24867b91b"
    ),
    "test": (
        "f974c0c533ec23ef3e71b8037d22329bc5fcbc63e69d925662e8436b2429663c"
    ),
}

MALEX_EXPECTED_BENIGN_SEQUENCE_SHA256 = (
    "30b7e6d4cead5263bb983356f6be77348b2ac48a2878b934ebfe68623b0f8424"
)

MALEX_EXPECTED_MALWARE_SEQUENCE_SHA256 = (
    "e20eff6ceaaaa3d87bcf2b185fa6ddd80615570e3677d887199dcf08754059e9"
)

MALEX_CONFLICT_IDENTITIES = (
    "294dea313b606a035305e9827c6d2cb389f567a49c0a4cf9c9399504c68f5b12",
    "2c3ddd638363b9da5491165d33b922de6cf30430b8a7187de1e5994e8524354f",
    "366db53139dcdff142d57d58aede52274e7325bdbbd77cff8b700d8c1ad5c13f",
    "3840424265b7ffdcf0c283ccbaed2fae2aff9850a3b69a920ffe1b7443d0d4b3",
    "42c88fca4ce525603cbea74a28cccee8fbcaa7c7b76037425b749f078cab6d27",
    "907e284c206c53f69aa1a0b365b93bb555c18ae8ed78af031fe85bfc0a4ef844",
    "93cb5bfed8cac47c5fa3ecf29bfe9c4de8cf238101f4891422037f87892715f9",
    "b49ea8a816a80757394f3a907cfa9ff72977635ddfc18c2c7a36d5bc4f83bbe0",
    "c5d17f98bc7865c7dd4f1572d8814a8af088982c24e688c710d513a2ee734cb5",
    "c60f6d6ea9690cc3404b7c1e6ae1f114e161245b9d541db0dd85016e1adb54dc",
    "c8bc0247ec39802219c995d0553e98ab7125390a7130741b46c2694c6d75616f",
    "d4c20fa8d43e7b592a95d37c386a4335f128cee06674d1393686fdeeb399543a",
    "d9ebc53ba716dc7470dac38145085a5f3e9a04930412715941082bd6edb012af",
    "e92aeb64d3e59a015e6c758b566708f6164797dabdaeaa8a1bb5f1f1f14a41e6",
    "eaf1508d00e73a1d8e0214e2b4a96dc85aeb9bc6883a5894882551714bf758ca",
    "eb61606562a3e5da03f0f2699a5a54fbb4b10c7da23a0480ee6a16e8fc92d7e4",
    "f23c452188e2af286d1479f74756ea1792c291efd4e640553de029c8c5f139d5",
    "f67ca74eba81e0c4f780a053ae5e3766097991f29256344210588c7454bc1216",
    "f87077f8b738c62ce06de362d00444dc162b68245270b5cc45c7550e8195ba62",
)

SplitName = Literal[
    "train",
    "validation",
    "test",
]


@dataclass(frozen=True)
class MaleXPartitionIndices:
    """Governed logical-corpus indices for MaleX revision-2026 splits."""

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def _sha256_file(
    path: Path,
    *,
    chunk_size: int = 1024 * 1024,
) -> str:
    digest = sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(
                chunk_size
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def _index_sha256(
    values: np.ndarray,
) -> str:
    array = np.asarray(
        values,
        dtype="<i8",
    )

    return sha256(
        array.tobytes(
            order="C"
        )
    ).hexdigest()


def _membership_sha256(
    identities: np.ndarray,
    labels: np.ndarray,
) -> str:
    digest = sha256()

    for identity, label in zip(
        identities,
        labels,
    ):
        digest.update(
            str(
                int(
                    label
                )
            ).encode(
                "ascii"
            )
        )

        digest.update(
            b"\t"
        )

        digest.update(
            bytes(
                identity
            )
        )

        digest.update(
            b"\n"
        )

    return digest.hexdigest()


class MaleXRevision2026:
    """Governed revision-2026 MaleX byteplot dataset view."""

    def __init__(
        self,
        label_archive: Path | str,
        benign_byteplot_root: Path | str,
        malware_byteplot_root: Path | str,
        *,
        validate_hashes: bool = True,
        validate_production_contract: bool = True,
    ) -> None:
        self.label_archive = Path(
            label_archive
        ).expanduser().resolve()

        self.benign_byteplot_root = Path(
            benign_byteplot_root
        ).expanduser().resolve()

        self.malware_byteplot_root = Path(
            malware_byteplot_root
        ).expanduser().resolve()

        self.validate_hashes = bool(
            validate_hashes
        )

        self.validate_production_contract = bool(
            validate_production_contract
        )

        self.num_classes = MALEX_NUM_CLASSES

        self.class_to_id = {
            "benign": 0,
            "malware": 1,
        }

        self.id_to_class = {
            0: "benign",
            1: "malware",
        }

        self._validate_required_paths()

        (
            self._benign_source,
            self._malware_source,
        ) = self._open_identity_memmaps()

        self._validate_source_sequences()

        (
            self._benign_positions,
            self._malware_positions,
        ) = self._derive_post_exclusion_positions()

        self._logical_labels = np.concatenate(
            (
                np.zeros(
                    self._benign_positions.shape[0],
                    dtype=np.int8,
                ),
                np.ones(
                    self._malware_positions.shape[0],
                    dtype=np.int8,
                ),
            )
        )

        self.partitions = self._derive_partitions()

        self._validate_partition_contract()

    def _validate_required_paths(
        self,
    ) -> None:
        if not self.label_archive.is_file():
            raise FileNotFoundError(
                f"missing MaleX label archive: {self.label_archive}"
            )

        if not self.benign_byteplot_root.is_dir():
            raise FileNotFoundError(
                "missing MaleX benign byteplot root: "
                f"{self.benign_byteplot_root}"
            )

        if not self.malware_byteplot_root.is_dir():
            raise FileNotFoundError(
                "missing MaleX malware byteplot root: "
                f"{self.malware_byteplot_root}"
            )

        if self.validate_production_contract:
            size = self.label_archive.stat().st_size

            if size != MALEX_LABEL_ARCHIVE_SIZE:
                raise ValueError(
                    "unexpected production MaleX label archive size: "
                    f"{size}"
                )

        if self.validate_hashes:
            observed = _sha256_file(
                self.label_archive
            )

            if (
                self.validate_production_contract
                and observed != MALEX_LABEL_ARCHIVE_SHA256
            ):
                raise ValueError(
                    "unexpected production MaleX label archive SHA-256: "
                    f"{observed}"
                )

    def _open_identity_memmaps(
        self,
    ) -> tuple[np.memmap, np.memmap]:
        with h5py.File(
            self.label_archive,
            "r",
        ) as h5:
            required = (
                "ben_sha256s",
                "mal_sha256s",
            )

            for name in required:
                if name not in h5:
                    raise ValueError(
                        f"MaleX label archive missing dataset: {name}"
                    )

            benign_ds = h5[
                "ben_sha256s"
            ]

            malware_ds = h5[
                "mal_sha256s"
            ]

            for name, ds in (
                (
                    "ben_sha256s",
                    benign_ds,
                ),
                (
                    "mal_sha256s",
                    malware_ds,
                ),
            ):
                if ds.ndim != 1:
                    raise ValueError(
                        f"MaleX {name} must be one-dimensional"
                    )

                if ds.dtype != np.dtype(
                    "S64"
                ):
                    raise TypeError(
                        f"MaleX {name} must use fixed-width S64 identities"
                    )

                if ds.chunks is not None:
                    raise ValueError(
                        f"MaleX {name} is not contiguous"
                    )

                if ds.compression is not None:
                    raise ValueError(
                        f"MaleX {name} is unexpectedly compressed"
                    )

            benign_offset = benign_ds.id.get_offset()
            malware_offset = malware_ds.id.get_offset()

            if benign_offset is None:
                raise RuntimeError(
                    "MaleX benign identity dataset has no direct file offset"
                )

            if malware_offset is None:
                raise RuntimeError(
                    "MaleX malware identity dataset has no direct file offset"
                )

            benign_shape = benign_ds.shape
            malware_shape = malware_ds.shape

        benign = np.memmap(
            self.label_archive,
            dtype="S64",
            mode="r",
            offset=int(
                benign_offset
            ),
            shape=benign_shape,
            order="C",
        )

        malware = np.memmap(
            self.label_archive,
            dtype="S64",
            mode="r",
            offset=int(
                malware_offset
            ),
            shape=malware_shape,
            order="C",
        )

        if self.validate_production_contract:
            if benign.shape != (
                MALEX_BENIGN_SOURCE_COUNT,
            ):
                raise ValueError(
                    "unexpected production MaleX benign identity count: "
                    f"{benign.shape}"
                )

            if malware.shape != (
                MALEX_MALWARE_SOURCE_COUNT,
            ):
                raise ValueError(
                    "unexpected production MaleX malware identity count: "
                    f"{malware.shape}"
                )

        return (
            benign,
            malware,
        )

    def _validate_source_sequences(self) -> None:
        """Verify original ordered HDF5 identity sequences.

        Production-only: SHA-256 over each S64 identity followed
        by a newline, in authoritative HDF5 source order.
        Runs before cross-class identity exclusions.
        """
        if not (
            self.validate_hashes
            and self.validate_production_contract
        ):
            return

        sequences = (
            (
                "benign",
                self._benign_source,
                MALEX_BENIGN_SOURCE_COUNT,
                MALEX_EXPECTED_BENIGN_SEQUENCE_SHA256,
            ),
            (
                "malware",
                self._malware_source,
                MALEX_MALWARE_SOURCE_COUNT,
                MALEX_EXPECTED_MALWARE_SEQUENCE_SHA256,
            ),
        )

        for name, identities, expected_count, expected_hash in sequences:
            if identities.shape != (expected_count,):
                raise ValueError(
                    f"MaleX {name} source sequence count mismatch"
                )

            if identities.dtype != np.dtype("S64"):
                raise TypeError(
                    f"MaleX {name} source sequence dtype mismatch"
                )

            digest = sha256()

            for start in range(0, expected_count, 8192):
                batch = identities[start:start + 8192]

                for identity in batch:
                    raw = bytes(identity)

                    if len(raw) != 64:
                        raise ValueError(
                            f"MaleX {name} source identity length mismatch"
                        )

                    digest.update(raw)
                    digest.update(b"\n")

            if digest.hexdigest() != expected_hash:
                raise ValueError(
                    f"MaleX {name} source sequence SHA-256 mismatch"
                )

    def _derive_post_exclusion_positions(
        self,
    ) -> tuple[np.ndarray, np.ndarray]:
        conflicts = np.asarray(
            [
                value.encode(
                    "ascii"
                )
                for value in MALEX_CONFLICT_IDENTITIES
            ],
            dtype="S64",
        )

        benign_mask = ~np.isin(
            self._benign_source,
            conflicts,
        )

        malware_mask = ~np.isin(
            self._malware_source,
            conflicts,
        )

        benign_positions = np.flatnonzero(
            benign_mask
        ).astype(
            np.int64,
            copy=False,
        )

        malware_positions = np.flatnonzero(
            malware_mask
        ).astype(
            np.int64,
            copy=False,
        )

        if self.validate_production_contract:
            if benign_positions.shape != (
                MALEX_POST_EXCLUSION_BENIGN,
            ):
                raise ValueError(
                    "unexpected MaleX post-exclusion benign count"
                )

            if malware_positions.shape != (
                MALEX_POST_EXCLUSION_MALWARE,
            ):
                raise ValueError(
                    "unexpected MaleX post-exclusion malware count"
                )

        return (
            benign_positions,
            malware_positions,
        )

    def _derive_partitions(
        self,
    ) -> MaleXPartitionIndices:
        logical_indices = np.arange(
            self._logical_labels.shape[0],
            dtype=np.int64,
        )

        partitions = split_unsplit_corpus(
            logical_indices,
            self._logical_labels,
            seed=DEFAULT_PARTITION_SEED,
        )

        return MaleXPartitionIndices(
            train=np.asarray(
                partitions.x_train,
                dtype=np.int64,
            ),
            validation=np.asarray(
                partitions.x_val,
                dtype=np.int64,
            ),
            test=np.asarray(
                partitions.x_test,
                dtype=np.int64,
            ),
        )

    def _split_indices(
        self,
        split: SplitName,
    ) -> np.ndarray:
        if split == "train":
            return self.partitions.train

        if split == "validation":
            return self.partitions.validation

        if split == "test":
            return self.partitions.test

        raise ValueError(
            f"unsupported MaleX logical split: {split!r}"
        )

    def _identities_for_corpus_indices(
        self,
        corpus_indices: np.ndarray,
    ) -> np.ndarray:
        indices = np.asarray(
            corpus_indices,
            dtype=np.int64,
        )

        benign_count = self._benign_positions.shape[
            0
        ]

        if np.any(
            indices < 0
        ) or np.any(
            indices >= self._logical_labels.shape[0]
        ):
            raise IndexError(
                "MaleX corpus index outside governed universe"
            )

        result = np.empty(
            indices.shape,
            dtype="S64",
        )

        benign_mask = (
            indices < benign_count
        )

        if np.any(
            benign_mask
        ):
            benign_logical = indices[
                benign_mask
            ]

            benign_source_positions = self._benign_positions[
                benign_logical
            ]

            result[
                benign_mask
            ] = self._benign_source[
                benign_source_positions
            ]

        malware_mask = ~benign_mask

        if np.any(
            malware_mask
        ):
            malware_logical = (
                indices[
                    malware_mask
                ]
                - benign_count
            )

            malware_source_positions = self._malware_positions[
                malware_logical
            ]

            result[
                malware_mask
            ] = self._malware_source[
                malware_source_positions
            ]

        return result

    def _validate_partition_contract(
        self,
    ) -> None:
        if not self.validate_production_contract:
            return

        for split in (
            "train",
            "validation",
            "test",
        ):
            indices = self._split_indices(
                split
            )

            expected_total = MALEX_EXPECTED_SPLIT_COUNTS[
                split
            ]

            if indices.shape != (
                expected_total,
            ):
                raise ValueError(
                    f"unexpected production MaleX {split} split size: "
                    f"{indices.shape}"
                )

            observed_index_hash = _index_sha256(
                indices
            )

            if (
                observed_index_hash
                != MALEX_EXPECTED_INDEX_SHA256[
                    split
                ]
            ):
                raise ValueError(
                    f"MaleX {split} logical-index fingerprint mismatch"
                )

            labels = self._logical_labels[
                indices
            ]

            for class_id in (
                0,
                1,
            ):
                observed_count = int(
                    np.sum(
                        labels == class_id
                    )
                )

                expected_count = MALEX_EXPECTED_CLASS_COUNTS[
                    split
                ][
                    class_id
                ]

                if observed_count != expected_count:
                    raise ValueError(
                        f"MaleX {split} class {class_id} count mismatch: "
                        f"{observed_count} != {expected_count}"
                    )

            identities = self._identities_for_corpus_indices(
                indices
            )

            observed_membership_hash = _membership_sha256(
                identities,
                labels,
            )

            if (
                observed_membership_hash
                != MALEX_EXPECTED_MEMBERSHIP_SHA256[
                    split
                ]
            ):
                raise ValueError(
                    f"MaleX {split} membership fingerprint mismatch"
                )

    def corpus_indices(
        self,
        split: SplitName,
        logical_positions: np.ndarray,
    ) -> np.ndarray:
        """Map requested split-local positions to governed corpus indices."""

        split_indices = self._split_indices(
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
            positions >= split_indices.shape[0]
        ):
            raise IndexError(
                "logical batch position outside split bounds"
            )

        return split_indices[
            positions
        ]

    def identities(
        self,
        split: SplitName,
        logical_positions: np.ndarray,
    ) -> tuple[str, ...]:
        """Return requested SHA-256 sample identities only."""

        corpus_indices = self.corpus_indices(
            split,
            logical_positions,
        )

        values = self._identities_for_corpus_indices(
            corpus_indices
        )

        return tuple(
            bytes(
                value
            ).decode(
                "ascii"
            )
            for value in values
        )

    def labels(
        self,
        split: SplitName,
        logical_positions: np.ndarray,
    ) -> np.ndarray:
        """Return governed binary model IDs for requested positions only."""

        corpus_indices = self.corpus_indices(
            split,
            logical_positions,
        )

        return np.asarray(
            self._logical_labels[
                corpus_indices
            ],
            dtype=np.int32,
        )

    def string_labels(
        self,
        split: SplitName,
        logical_positions: np.ndarray,
    ) -> tuple[str, ...]:
        """Return authoritative semantic labels for requested positions."""

        ids = self.labels(
            split,
            logical_positions,
        )

        return tuple(
            self.id_to_class[
                int(
                    class_id
                )
            ]
            for class_id in ids
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
        """Resolve requested MaleX identities to byteplot PNG paths."""

        ids = self.labels(
            split,
            logical_positions,
        )

        identities = self.identities(
            split,
            logical_positions,
        )

        paths = []

        for identity, class_id in zip(
            identities,
            ids,
        ):
            root = (
                self.benign_byteplot_root
                if int(
                    class_id
                ) == 0
                else self.malware_byteplot_root
            )

            paths.append(
                root
                / f"{identity}_img_256.png"
            )

        return tuple(
            paths
        )

    def _source_batch(
        self,
        split: SplitName,
        logical_positions: np.ndarray,
    ) -> np.ndarray:
        """Decode one requested MaleX grayscale uint8 batch."""

        paths = self.source_paths(
            split,
            logical_positions,
        )

        images = []

        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(
                    f"missing MaleX source image: {path}"
                )

            with Image.open(
                path
            ) as image:
                if image.format != "PNG":
                    raise ValueError(
                        f"MaleX source image is not PNG: {path}"
                    )

                if image.mode != "L":
                    raise ValueError(
                        f"MaleX source image is not grayscale L: "
                        f"{path} mode={image.mode!r}"
                    )

                array = np.asarray(
                    image,
                    dtype=np.uint8,
                )

            if array.shape != (
                256,
                256,
            ):
                raise ValueError(
                    f"unexpected MaleX source image shape: "
                    f"{path} shape={array.shape}"
                )

            images.append(
                array[
                    ...,
                    None,
                ]
            )

        if not images:
            return np.empty(
                (
                    0,
                    *MALEX_SOURCE_SHAPE,
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
        ) != MALEX_CANONICAL_SHAPE:
            raise RuntimeError(
                "MaleX canonical batch has unexpected shape: "
                f"{images.shape}"
            )

        if images.dtype != np.float32:
            raise RuntimeError(
                "MaleX canonical batch is not float32"
            )

        if images.shape[0] > 0:
            if not np.array_equal(
                images[
                    ...,
                    0,
                ],
                images[
                    ...,
                    1,
                ],
            ):
                raise RuntimeError(
                    "MaleX canonical grayscale channels differ"
                )

            if not np.array_equal(
                images[
                    ...,
                    1,
                ],
                images[
                    ...,
                    2,
                ],
            ):
                raise RuntimeError(
                    "MaleX canonical grayscale channels differ"
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
        """Return governed sample count for one logical split."""

        return int(
            self._split_indices(
                split
            ).shape[0]
        )


__all__ = [
    "MALEX_CANONICAL_SHAPE",
    "MALEX_CONFLICT_IDENTITIES",
    "MALEX_EXPECTED_CLASS_COUNTS",
    "MALEX_EXPECTED_INDEX_SHA256",
    "MALEX_EXPECTED_MEMBERSHIP_SHA256",
    "MALEX_EXPECTED_SPLIT_COUNTS",
    "MALEX_LABEL_ARCHIVE_SHA256",
    "MALEX_NUM_CLASSES",
    "MALEX_SOURCE_SHAPE",
    "MaleXPartitionIndices",
    "MaleXRevision2026",
]
