"""Revision-2026 USTC-TFC2016 data adapter.

The adapter preserves the historical source artifacts and applies only the
governed revision-2026 partitioning and canonicalization semantics.

Key properties:

- primary source arrays are memory-mapped;
- provider test is preserved;
- validation is derived from provider training (Case 2);
- partition seed is independent of experimental/model seeds;
- source NCHW images are explicitly transposed to NHWC;
- canonicalization occurs per requested batch;
- no full canonical dataset is materialized on disk or in memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import tensorflow as tf

from common.canonicalization import (
    CANONICAL_SHAPE,
    SourceNumericRange,
    canonicalize_images,
)
from common.partitioning import (
    DEFAULT_CASE2_VAL_FRACTION,
    DEFAULT_PARTITION_SEED,
    derive_validation_from_train,
)


USTC_NUM_CLASSES = 9
USTC_SOURCE_SHAPE_NCHW = (1, 40, 40)
USTC_CANONICAL_SHAPE = CANONICAL_SHAPE

SplitName = Literal["train", "validation", "test"]


@dataclass(frozen=True)
class USTCSourceArrays:
    """Memory-mapped provider-style USTC arrays."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


@dataclass(frozen=True)
class USTCPartitionIndices:
    """Revision-2026 logical split indices."""

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


class USTCRevision2026:
    """Governed revision-2026 USTC-TFC2016 dataset view."""

    def __init__(
        self,
        source_dir: Path | str,
        *,
        partition_seed: int = DEFAULT_PARTITION_SEED,
        validation_fraction: float = DEFAULT_CASE2_VAL_FRACTION,
    ) -> None:
        self.source_dir = Path(source_dir).expanduser().resolve()
        self.partition_seed = int(partition_seed)
        self.validation_fraction = float(validation_fraction)

        self._validate_source_files()

        self.arrays = self._load_source_arrays()
        self._validate_source_contract()

        self.partitions = self._derive_partition_indices()

    def _validate_source_files(self) -> None:
        required = (
            "train_data.npy",
            "train_labels.npy",
            "test_data.npy",
            "test_labels.npy",
        )

        missing = [
            name
            for name in required
            if not (self.source_dir / name).is_file()
        ]

        if missing:
            raise FileNotFoundError(
                f"missing USTC source files under {self.source_dir}: "
                f"{missing}"
            )

    def _load_source_arrays(self) -> USTCSourceArrays:
        return USTCSourceArrays(
            x_train=np.load(
                self.source_dir / "train_data.npy",
                allow_pickle=False,
                mmap_mode="r",
            ),
            y_train=np.load(
                self.source_dir / "train_labels.npy",
                allow_pickle=False,
                mmap_mode="r",
            ),
            x_test=np.load(
                self.source_dir / "test_data.npy",
                allow_pickle=False,
                mmap_mode="r",
            ),
            y_test=np.load(
                self.source_dir / "test_labels.npy",
                allow_pickle=False,
                mmap_mode="r",
            ),
        )

    def _validate_source_contract(self) -> None:
        arrays = self.arrays

        if tuple(arrays.x_train.shape[1:]) != USTC_SOURCE_SHAPE_NCHW:
            raise ValueError(
                "unexpected USTC provider training shape: "
                f"{arrays.x_train.shape}"
            )

        if tuple(arrays.x_test.shape[1:]) != USTC_SOURCE_SHAPE_NCHW:
            raise ValueError(
                "unexpected USTC provider test shape: "
                f"{arrays.x_test.shape}"
            )

        if arrays.x_train.dtype != np.float64:
            raise TypeError(
                "unexpected USTC provider training dtype: "
                f"{arrays.x_train.dtype}"
            )

        if arrays.x_test.dtype != np.float64:
            raise TypeError(
                "unexpected USTC provider test dtype: "
                f"{arrays.x_test.dtype}"
            )

        if arrays.y_train.dtype != np.int32:
            raise TypeError(
                "unexpected USTC provider training-label dtype: "
                f"{arrays.y_train.dtype}"
            )

        if arrays.y_test.dtype != np.int32:
            raise TypeError(
                "unexpected USTC provider test-label dtype: "
                f"{arrays.y_test.dtype}"
            )

        if len(arrays.x_train) != len(arrays.y_train):
            raise ValueError(
                "USTC provider training feature/label length mismatch"
            )

        if len(arrays.x_test) != len(arrays.y_test):
            raise ValueError(
                "USTC provider test feature/label length mismatch"
            )

        train_classes = np.unique(arrays.y_train)
        test_classes = np.unique(arrays.y_test)
        expected_classes = np.arange(
            USTC_NUM_CLASSES,
            dtype=np.int32,
        )

        if not np.array_equal(
            train_classes,
            expected_classes,
        ):
            raise ValueError(
                "unexpected USTC provider training classes: "
                f"{train_classes}"
            )

        if not np.array_equal(
            test_classes,
            expected_classes,
        ):
            raise ValueError(
                "unexpected USTC provider test classes: "
                f"{test_classes}"
            )

    def _derive_partition_indices(self) -> USTCPartitionIndices:
        train_indices = np.arange(
            len(self.arrays.x_train),
            dtype=np.int64,
        )

        test_indices = np.arange(
            len(self.arrays.x_test),
            dtype=np.int64,
        )

        logical = derive_validation_from_train(
            train_indices,
            self.arrays.y_train,
            test_indices,
            self.arrays.y_test,
            val_fraction=self.validation_fraction,
            seed=self.partition_seed,
        )

        return USTCPartitionIndices(
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

    def labels(
        self,
        split: SplitName,
    ) -> np.ndarray:
        """Return integer class IDs for one logical split."""
        if split == "train":
            return np.asarray(
                self.arrays.y_train[
                    self.partitions.train
                ],
                dtype=np.int32,
            )

        if split == "validation":
            return np.asarray(
                self.arrays.y_train[
                    self.partitions.validation
                ],
                dtype=np.int32,
            )

        if split == "test":
            return np.asarray(
                self.arrays.y_test[
                    self.partitions.test
                ],
                dtype=np.int32,
            )

        raise ValueError(
            f"unsupported USTC logical split: {split!r}"
        )

    def one_hot_labels(
        self,
        split: SplitName,
    ) -> np.ndarray:
        """Return float32 one-hot labels for one logical split."""
        ids = self.labels(split)

        return tf.one_hot(
            ids,
            depth=USTC_NUM_CLASSES,
            dtype=tf.float32,
        ).numpy()

    def _source_batch(
        self,
        split: SplitName,
        logical_positions: np.ndarray,
    ) -> np.ndarray:
        """Fetch one source batch using logical split positions."""
        positions = np.asarray(
            logical_positions,
            dtype=np.int64,
        )

        if positions.ndim != 1:
            raise ValueError(
                "logical_positions must be one-dimensional"
            )

        if split == "train":
            split_indices = self.partitions.train
            source = self.arrays.x_train

        elif split == "validation":
            split_indices = self.partitions.validation
            source = self.arrays.x_train

        elif split == "test":
            split_indices = self.partitions.test
            source = self.arrays.x_test

        else:
            raise ValueError(
                f"unsupported USTC logical split: {split!r}"
            )

        if np.any(positions < 0) or np.any(
            positions >= len(split_indices)
        ):
            raise IndexError(
                "logical batch position outside split bounds"
            )

        source_indices = split_indices[positions]

        batch_nchw = np.asarray(
            source[source_indices]
        )

        return np.transpose(
            batch_nchw,
            (0, 2, 3, 1),
        )

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

        source_nhwc = self._source_batch(
            split,
            positions,
        )

        images = canonicalize_images(
            source_nhwc,
            source_range=SourceNumericRange.ZERO_ONE,
        )

        labels = self.one_hot_labels(split)[positions]

        if tuple(images.shape[1:]) != USTC_CANONICAL_SHAPE:
            raise RuntimeError(
                "USTC canonical batch has unexpected shape: "
                f"{images.shape}"
            )

        if images.dtype != np.float32:
            raise RuntimeError(
                "USTC canonical batch is not float32"
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
        if split == "train":
            return int(len(self.partitions.train))

        if split == "validation":
            return int(len(self.partitions.validation))

        if split == "test":
            return int(len(self.partitions.test))

        raise ValueError(
            f"unsupported USTC logical split: {split!r}"
        )


__all__ = [
    "USTC_NUM_CLASSES",
    "USTCPartitionIndices",
    "USTCRevision2026",
    "USTCSourceArrays",
]
