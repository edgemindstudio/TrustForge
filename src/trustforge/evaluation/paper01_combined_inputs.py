"""Logical real-plus-synthetic training access without full image concatenation.

CIC AR/MAF scope. Does not authenticate provenance or select shuffle policy.
"""

from copy import deepcopy

import numpy as np

from .paper01_inputs import MAX_BATCH_SIZE, RealSplitView, validate_batch
from .paper01_cic_pool import CICPoolView


class CombinedTrainingView:
    def __init__(self, *, real, synthetic):
        if not isinstance(real, RealSplitView):
            raise TypeError("governed real split view required")
        if not isinstance(synthetic, CICPoolView):
            raise TypeError("CIC synthetic pool view required")
        real_identity = real.identity
        synthetic_identity = synthetic.identity
        if real_identity["split"] != "train":
            raise ValueError("combined training requires the real training split")
        if (
            real_identity["dataset_id"] != "cicmaldroid2020"
            or synthetic_identity["dataset_id"] != real_identity["dataset_id"]
            or real_identity["class_count"] != 5
        ):
            raise ValueError("real and synthetic dataset/class contracts differ")

        # Full streaming verification precedes availability to classifier code.
        synthetic.preflight()
        self._real = real
        self._synthetic = synthetic
        self._real_size = real.size
        self._synthetic_size = synthetic.size
        self._identity = {
            "condition": "real_plus_synthetic",
            "real": real_identity,
            "synthetic": synthetic_identity,
            "position_order": "real_then_synthetic",
        }

    @property
    def size(self):
        return self._real_size + self._synthetic_size

    @property
    def identity(self):
        return deepcopy(self._identity)

    def sample_identity(self, position):
        if type(position) is not int or not 0 <= position < self.size:
            raise ValueError("invalid combined sample position")
        if position < self._real_size:
            return {
                "source": "real", "split": "train",
                "logical_position": position,
            }
        return {
            "source": "synthetic",
            "manifest_sha256": self._identity["synthetic"]["manifest_sha256"],
            "logical_position": position - self._real_size,
        }

    def batch(self, positions):
        if (
            not isinstance(positions, np.ndarray)
            or positions.dtype != np.int64
            or positions.ndim != 1
            or not 1 <= positions.size <= MAX_BATCH_SIZE
            or np.any(positions < 0) or np.any(positions >= self.size)
            or np.unique(positions).size != positions.size
        ):
            raise ValueError("invalid bounded combined positions")
        if (
            self._real.identity != self._identity["real"]
            or self._synthetic.identity != self._identity["synthetic"]
        ):
            raise RuntimeError("input view identity changed")

        requested = positions.copy()
        images = np.empty((len(requested), 40, 40, 3), dtype=np.float32)
        labels = np.empty((len(requested), 5), dtype=np.float32)
        real_mask = requested < self._real_size
        for mask, view, offset in (
            (real_mask, self._real, 0),
            (~real_mask, self._synthetic, self._real_size),
        ):
            locations = np.flatnonzero(mask)
            if not len(locations):
                continue
            local = requested[locations] - offset
            observed, batch_images, batch_labels = view.batch(local)
            if not np.array_equal(observed, local):
                raise RuntimeError("input view changed requested position order")
            validate_batch(
                batch_images, batch_labels, count=len(local), class_count=5,
            )
            images[locations] = batch_images
            labels[locations] = batch_labels

        validate_batch(images, labels, count=len(requested), class_count=5)
        return requested, images, labels

    def iter_ordered(self, *, batch_size=MAX_BATCH_SIZE):
        """Complete ordered traversal; not a selected training shuffle policy."""
        if type(batch_size) is not int or not 1 <= batch_size <= MAX_BATCH_SIZE:
            raise ValueError("invalid bounded batch size")
        for start in range(0, self.size, batch_size):
            yield self.batch(np.arange(
                start, min(start + batch_size, self.size), dtype=np.int64,
            ))
