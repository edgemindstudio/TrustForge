"""Deterministic primary MS-SSIM pair selection.

Consumes labels in verified manifest order. Does not authenticate that order
or manifest identity. Caller must persist selection before scoring.
"""

import hashlib
import re
import numpy as np


def _decode_pair_ranks(ranks, count):
    """Map lexicographic unordered-pair ranks to distinct local indices."""
    starts = np.arange(count, dtype=np.int64)
    starts = starts * (2 * count - starts - 1) // 2
    first = np.searchsorted(starts, ranks, side="right") - 1
    second = first + 1 + ranks - starts[first]
    return np.column_stack((first, second))


def select_ms_ssim_pairs(*, labels, class_count, manifest_sha256):
    if type(class_count) is not int or class_count < 2:
        raise ValueError("authoritative class count >=2 required")
    if (
        type(manifest_sha256) is not str
        or re.fullmatch(r"[0-9a-f]{64}", manifest_sha256) is None
    ):
        raise ValueError("explicit manifest SHA-256 required")
    if (
        not isinstance(labels, np.ndarray)
        or labels.ndim != 1
        or labels.dtype.kind not in "iu"
        or labels.size != class_count * 2000
        or np.any(labels < 0)
        or np.any(labels >= class_count)
    ):
        raise ValueError("ordered authoritative integer labels required")
    counts = np.bincount(labels.astype(np.int64), minlength=class_count)
    if not np.all(counts == 2000):
        raise ValueError("complete 2000/class pool required")

    classes = {}
    possible = 2000 * 1999 // 2
    for class_id in range(class_count):
        positions = np.flatnonzero(labels == class_id)
        ranks = np.random.default_rng(42 + class_id).choice(
            possible, size=200, replace=False,
        )
        local = _decode_pair_ranks(ranks, 2000)
        classes[str(class_id)] = {
            "sample_count": 2000,
            "possible_pair_count": possible,
            "pair_count": 200,
            "seed": 42 + class_id,
            "local_pair_indices": local.tolist(),
            "manifest_position_pairs": positions[local].tolist(),
        }
    normalized = labels.astype("<i8", copy=False)
    label_digest = hashlib.sha256(
        b"paper01-ordered-labels-int64le-v1"
        + len(normalized).to_bytes(8, "big")
        + normalized.tobytes(order="C")
    ).hexdigest()
    return {
        "schema": "trustforge.paper01.ms-ssim-pairs.v1",
        "metric": "ms_ssim_3scale_40x40_v1",
        "manifest_sha256": manifest_sha256,
        "ordered_labels_sha256": label_digest,
        "class_count": class_count,
        "sampling": "uniform_unordered_pairs_without_replacement",
        "pair_rank_order": "lexicographic_first_then_second",
        "classes": classes,
    }
