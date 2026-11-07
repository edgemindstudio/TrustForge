"""Accepted class-balanced fidelity reference selection.

Consumes labels in governed split/manifest order. Does not authenticate that
order, split identity or manifest identity. No images/features are inspected.
Real-reference persistence before family comparison is caller-owned.
"""

import hashlib
import json
import re

import numpy as np


HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _digest(value):
    if type(value) is not str or HEX64.fullmatch(value) is None:
        raise ValueError("explicit SHA-256 identity required")
    return value


def _canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def _labels(value, classes):
    if (
        not isinstance(value, np.ndarray)
        or value.ndim != 1
        or value.size == 0
        or value.dtype.kind not in "iu"
    ):
        raise ValueError("ordered integer label vector required")
    if np.any(value < 0) or np.any(value >= classes):
        raise ValueError("labels outside authoritative class universe")
    counts = np.bincount(value.astype(np.int64), minlength=classes)
    if np.any(counts == 0):
        raise ValueError("missing authoritative class")
    return counts


def _label_digest(labels):
    # Normalize integer storage while preserving ordered label identities.
    values = labels.astype("<i8", copy=False)
    return hashlib.sha256(
        b"paper01-ordered-labels-int64le-v1"
        + len(values).to_bytes(8, "big")
        + values.tobytes(order="C")
    ).hexdigest()


def select_balanced_references(
    *, real_validation_labels, synthetic_labels, class_count,
    real_validation_identity_sha256, synthetic_manifest_sha256,
):
    """Return reusable real references and a matched synthetic selection."""
    if type(class_count) is not int or class_count < 2:
        raise ValueError("authoritative class count must be an integer >=2")
    real_identity = _digest(real_validation_identity_sha256)
    synthetic_identity = _digest(synthetic_manifest_sha256)
    real_counts = _labels(real_validation_labels, class_count)
    synthetic_counts = _labels(synthetic_labels, class_count)
    if not np.all(synthetic_counts == 2000):
        raise ValueError("primary fidelity requires exactly 2000 synthetic/class")
    m = min(200, int(real_counts.min()))
    if m < 2:
        raise ValueError("at least two validation samples/class required")

    real_positions = []
    synthetic_positions = []
    for class_id in range(class_count):
        real_candidates = np.flatnonzero(real_validation_labels == class_id)
        synthetic_candidates = np.flatnonzero(synthetic_labels == class_id)
        # Separate generators prevent synthetic ordering/counts from advancing
        # the real-reference stream.
        real_rng = np.random.default_rng(42 + class_id)
        synthetic_rng = np.random.default_rng(42 + class_id)
        real_positions.append(
            real_rng.choice(real_candidates, size=m, replace=False).tolist()
        )
        synthetic_positions.append(
            synthetic_rng.choice(synthetic_candidates, size=m, replace=False).tolist()
        )

    common = {
        "class_count": class_count,
        "samples_per_class": m,
        "selection": "uniform_without_replacement",
        "seed_policy": "separate_default_rng_42_plus_class_id",
        "output_order": "class_id_then_rng_draw_order",
    }
    real = {
        "schema": "trustforge.paper01.real-fidelity-reference.v1",
        **common,
        "split": "validation",
        "declared_split_identity_sha256": real_identity,
        "ordered_labels_sha256": _label_digest(real_validation_labels),
        "available_counts": real_counts.tolist(),
        "positions_by_class": real_positions,
    }
    synthetic = {
        "schema": "trustforge.paper01.synthetic-fidelity-reference.v1",
        **common,
        "manifest_sha256": synthetic_identity,
        "ordered_labels_sha256": _label_digest(synthetic_labels),
        "available_counts": synthetic_counts.tolist(),
        "positions_by_class": synthetic_positions,
    }
    return {
        "real_reference": real,
        "real_reference_sha256": hashlib.sha256(_canonical(real)).hexdigest(),
        "synthetic_reference": synthetic,
        "synthetic_reference_sha256": hashlib.sha256(_canonical(synthetic)).hexdigest(),
    }
