"""Revision-2026 CICMalDroid2020 data adapter.

The adapter reconstructs the verified CIC representation semantics from the
authoritative unsplit source while applying the governed revision-2026 Case-3
partition policy.

Key properties:

- authoritative source is the unsplit 11,598-row CSV;
- feature order is explicit and contract-bound;
- train / validation / test are derived 70 / 10 / 20 with partition seed 42;
- partition seed is independent of experimental/model seeds;
- StandardScaler is fit on the revised training partition only;
- sklearn float32 transform semantics are preserved;
- tanh is applied before five zero-padding features are appended;
- 144 values are reshaped row-major to 12 x 12 x 1;
- canonicalization converts [-1, 1] to [0, 1], resizes to 40 x 40, and
  replicates grayscale to RGB;
- no full transformed or canonical dataset is materialized.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

from common.canonicalization import (
    CANONICAL_SHAPE,
    SourceNumericRange,
    canonicalize_images,
)
from common.partitioning import (
    DEFAULT_CASE3_RATIOS,
    DEFAULT_PARTITION_SEED,
    split_unsplit_corpus,
)


CIC_NUM_CLASSES = 5
CIC_SOURCE_ROWS = 11_598
CIC_RAW_FEATURES = 139
CIC_PADDED_FEATURES = 144
CIC_SOURCE_IMAGE_SHAPE = (12, 12, 1)
CIC_CANONICAL_SHAPE = CANONICAL_SHAPE

CIC_SOURCE_SHA256 = (
    "3415babdd76a2b77c400a9ec3a8a6057705d5a57a9e91753191ecc2946df7c00"
)

CIC_FEATURE_COLUMNS_SHA256 = (
    "7a6045fe10bfe27c1faee98d729de470401b7cf733ccccc107499f11dea8fc8f"
)

SplitName = Literal["train", "validation", "test"]


@dataclass(frozen=True)
class CICSourceArrays:
    """Authoritative unsplit CIC feature matrix and source labels."""

    features: np.ndarray
    raw_labels: np.ndarray


@dataclass(frozen=True)
class CICPartitionIndices:
    """Revision-2026 logical split indices into the unsplit source."""

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
            chunk = handle.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


class CICMalDroidRevision2026:
    """Governed revision-2026 CICMalDroid2020 dataset view."""

    def __init__(
        self,
        source_csv: Path | str,
        feature_columns_path: Path | str,
        *,
        partition_seed: int = DEFAULT_PARTITION_SEED,
        train_fraction: float = DEFAULT_CASE3_RATIOS[0],
        validation_fraction: float = DEFAULT_CASE3_RATIOS[1],
        test_fraction: float = DEFAULT_CASE3_RATIOS[2],
        validate_hashes: bool = True,
    ) -> None:
        self.source_csv = Path(
            source_csv
        ).expanduser().resolve()

        self.feature_columns_path = Path(
            feature_columns_path
        ).expanduser().resolve()

        self.partition_seed = int(partition_seed)
        self.train_fraction = float(train_fraction)
        self.validation_fraction = float(validation_fraction)
        self.test_fraction = float(test_fraction)
        self.validate_hashes = bool(validate_hashes)

        self._validate_source_files()

        self.feature_columns = self._load_feature_columns()

        self.arrays = self._load_source_arrays()
        self._validate_source_contract()

        self.partitions = self._derive_partition_indices()

        self.scaler = self._fit_revision_scaler()

    def _validate_source_files(self) -> None:
        if not self.source_csv.is_file():
            raise FileNotFoundError(
                f"missing CIC source CSV: {self.source_csv}"
            )

        if not self.feature_columns_path.is_file():
            raise FileNotFoundError(
                "missing CIC feature-column contract: "
                f"{self.feature_columns_path}"
            )

        if self.validate_hashes:
            source_hash = _sha256_file(
                self.source_csv
            )

            if source_hash != CIC_SOURCE_SHA256:
                raise ValueError(
                    "unexpected CIC authoritative-source SHA-256: "
                    f"{source_hash}"
                )

            feature_hash = _sha256_file(
                self.feature_columns_path
            )

            if feature_hash != CIC_FEATURE_COLUMNS_SHA256:
                raise ValueError(
                    "unexpected CIC feature-column SHA-256: "
                    f"{feature_hash}"
                )

    def _load_feature_columns(self) -> tuple[str, ...]:
        obj = json.loads(
            self.feature_columns_path.read_text(
                encoding="utf-8"
            )
        )

        if not isinstance(obj, list):
            raise TypeError(
                "CIC feature-column contract must be a JSON list"
            )

        if len(obj) != CIC_RAW_FEATURES:
            raise ValueError(
                "unexpected CIC feature-column count: "
                f"{len(obj)}"
            )

        if not all(
            isinstance(item, str)
            for item in obj
        ):
            raise TypeError(
                "CIC feature-column contract contains non-string values"
            )

        if len(set(obj)) != len(obj):
            raise ValueError(
                "CIC feature-column contract contains duplicates"
            )

        return tuple(obj)

    def _load_source_arrays(self) -> CICSourceArrays:
        usecols = [
            *self.feature_columns,
            "Class",
        ]

        frame = pd.read_csv(
            self.source_csv,
            usecols=usecols,
        )

        features = frame.loc[
            :,
            self.feature_columns,
        ].to_numpy(
            dtype=np.float32,
            copy=True,
        )

        raw_labels_float = frame[
            "Class"
        ].to_numpy(
            dtype=np.float64,
            copy=True,
        )

        if not np.all(
            np.isfinite(raw_labels_float)
        ):
            raise ValueError(
                "CIC source contains non-finite labels"
            )

        rounded = np.rint(
            raw_labels_float
        )

        if not np.array_equal(
            raw_labels_float,
            rounded,
        ):
            raise ValueError(
                "CIC source contains non-integer-valued labels"
            )

        raw_labels = rounded.astype(
            np.int32,
        )

        return CICSourceArrays(
            features=features,
            raw_labels=raw_labels,
        )

    def _validate_source_contract(self) -> None:
        arrays = self.arrays

        if arrays.features.shape != (
            CIC_SOURCE_ROWS,
            CIC_RAW_FEATURES,
        ):
            raise ValueError(
                "unexpected CIC feature-matrix shape: "
                f"{arrays.features.shape}"
            )

        if arrays.features.dtype != np.float32:
            raise TypeError(
                "CIC source feature matrix is not float32"
            )

        if arrays.raw_labels.shape != (
            CIC_SOURCE_ROWS,
        ):
            raise ValueError(
                "unexpected CIC label shape: "
                f"{arrays.raw_labels.shape}"
            )

        if arrays.raw_labels.dtype != np.int32:
            raise TypeError(
                "CIC source labels are not int32"
            )

        if not np.all(
            np.isfinite(arrays.features)
        ):
            raise ValueError(
                "CIC source contains non-finite feature values"
            )

        expected_classes = np.arange(
            1,
            CIC_NUM_CLASSES + 1,
            dtype=np.int32,
        )

        observed_classes = np.unique(
            arrays.raw_labels
        )

        if not np.array_equal(
            observed_classes,
            expected_classes,
        ):
            raise ValueError(
                "unexpected CIC source classes: "
                f"{observed_classes}"
            )

        _, counts = np.unique(
            arrays.raw_labels,
            return_counts=True,
        )

        expected_counts = np.asarray(
            [
                1253,
                2100,
                3904,
                2546,
                1795,
            ],
            dtype=np.int64,
        )

        if not np.array_equal(
            counts,
            expected_counts,
        ):
            raise ValueError(
                "unexpected CIC source class counts: "
                f"{counts}"
            )

    def _derive_partition_indices(
        self,
    ) -> CICPartitionIndices:
        source_indices = np.arange(
            CIC_SOURCE_ROWS,
            dtype=np.int64,
        )

        logical = split_unsplit_corpus(
            source_indices,
            self.arrays.raw_labels,
            train_fraction=self.train_fraction,
            val_fraction=self.validation_fraction,
            test_fraction=self.test_fraction,
            seed=self.partition_seed,
        )

        return CICPartitionIndices(
            train=np.asarray(
                logical.x_train,
                dtype=np.int64,
            ),
            validation=np.asarray(
                logical.x_val,
                dtype=np.int64,
            ),
            test=np.asarray(
                logical.x_test,
                dtype=np.int64,
            ),
        )

    def _fit_revision_scaler(
        self,
    ) -> StandardScaler:
        scaler = StandardScaler()

        train_features = self.arrays.features[
            self.partitions.train
        ]

        scaler.fit(
            train_features
        )

        if int(
            scaler.n_features_in_
        ) != CIC_RAW_FEATURES:
            raise RuntimeError(
                "CIC fitted scaler has unexpected feature count"
            )

        return scaler

    def raw_labels(
        self,
        split: SplitName,
    ) -> np.ndarray:
        """Return source class IDs in the authoritative 1..5 encoding."""
        indices = self._split_indices(
            split
        )

        return np.asarray(
            self.arrays.raw_labels[
                indices
            ],
            dtype=np.int32,
        )

    def labels(
        self,
        split: SplitName,
    ) -> np.ndarray:
        """Return zero-based integer class IDs for model consumption."""
        source_ids = self.raw_labels(
            split
        )

        model_ids = source_ids - 1

        expected = np.arange(
            CIC_NUM_CLASSES,
            dtype=np.int32,
        )

        observed = np.unique(
            model_ids
        )

        if not np.array_equal(
            observed,
            expected,
        ):
            raise RuntimeError(
                "CIC logical split does not contain expected "
                f"zero-based classes: {observed}"
            )

        return model_ids.astype(
            np.int32,
            copy=False,
        )

    def one_hot_labels(
        self,
        split: SplitName,
    ) -> np.ndarray:
        """Return float32 one-hot labels for one logical split."""
        ids = self.labels(
            split
        )

        return tf.one_hot(
            ids,
            depth=CIC_NUM_CLASSES,
            dtype=tf.float32,
        ).numpy()

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
            f"unsupported CIC logical split: {split!r}"
        )

    def _source_batch(
        self,
        split: SplitName,
        logical_positions: np.ndarray,
    ) -> np.ndarray:
        """Build one revised CIC 12 x 12 x 1 source-representation batch."""
        positions = np.asarray(
            logical_positions,
            dtype=np.int64,
        )

        if positions.ndim != 1:
            raise ValueError(
                "logical_positions must be one-dimensional"
            )

        split_indices = self._split_indices(
            split
        )

        if np.any(
            positions < 0
        ) or np.any(
            positions >= len(split_indices)
        ):
            raise IndexError(
                "logical batch position outside split bounds"
            )

        source_indices = split_indices[
            positions
        ]

        raw = np.asarray(
            self.arrays.features[
                source_indices
            ],
            dtype=np.float32,
        )

        transformed = self.scaler.transform(
            raw
        )

        if transformed.dtype != np.float32:
            raise RuntimeError(
                "CIC StandardScaler transform did not preserve float32"
            )

        bounded = np.tanh(
            transformed
        )

        if bounded.dtype != np.float32:
            raise RuntimeError(
                "CIC tanh output did not preserve float32"
            )

        padded = np.pad(
            bounded,
            (
                (0, 0),
                (
                    0,
                    CIC_PADDED_FEATURES
                    - CIC_RAW_FEATURES,
                ),
            ),
            mode="constant",
            constant_values=0.0,
        )

        source_images = padded.reshape(
            -1,
            *CIC_SOURCE_IMAGE_SHAPE,
        )

        if source_images.dtype != np.float32:
            raise RuntimeError(
                "CIC source image batch is not float32"
            )

        if tuple(
            source_images.shape[1:]
        ) != CIC_SOURCE_IMAGE_SHAPE:
            raise RuntimeError(
                "CIC source image batch has unexpected shape: "
                f"{source_images.shape}"
            )

        return source_images

    def canonical_batch(
        self,
        split: SplitName,
        logical_positions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return one canonical image/one-hot-label batch."""
        positions = np.asarray(
            logical_positions,
            dtype=np.int64,
        )

        source_images = self._source_batch(
            split,
            positions,
        )

        images = canonicalize_images(
            source_images,
            source_range=SourceNumericRange.MINUS_ONE_ONE,
        )

        labels = self.one_hot_labels(
            split
        )[positions]

        if tuple(
            images.shape[1:]
        ) != CIC_CANONICAL_SHAPE:
            raise RuntimeError(
                "CIC canonical batch has unexpected shape: "
                f"{images.shape}"
            )

        if images.dtype != np.float32:
            raise RuntimeError(
                "CIC canonical batch is not float32"
            )

        if not np.all(
            np.isfinite(images)
        ):
            raise RuntimeError(
                "CIC canonical batch contains non-finite values"
            )

        if np.any(
            images < 0.0
        ) or np.any(
            images > 1.0
        ):
            raise RuntimeError(
                "CIC canonical batch is outside [0, 1]"
            )

        return images, labels.astype(
            np.float32,
            copy=False,
        )

    def split_size(
        self,
        split: SplitName,
    ) -> int:
        """Return sample count for one logical split."""
        return int(
            len(
                self._split_indices(
                    split
                )
            )
        )


__all__ = [
    "CICMalDroidRevision2026",
    "CICPartitionIndices",
    "CICSourceArrays",
    "CIC_NUM_CLASSES",
]
