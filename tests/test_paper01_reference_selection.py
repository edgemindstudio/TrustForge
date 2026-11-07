"""Reference-selection invariants; no image or feature extraction."""

import numpy as np
import pytest

from trustforge.evaluation.paper01_reference_selection import (
    select_balanced_references,
)


def select(real=None, synthetic=None, **changes):
    arguments = {
        "real_validation_labels": (
            np.repeat(np.arange(3, dtype=np.int32), [7, 3, 5])
            if real is None else real
        ),
        "synthetic_labels": (
            np.repeat(np.arange(3, dtype=np.int32), 2000)
            if synthetic is None else synthetic
        ),
        "class_count": 3,
        "real_validation_identity_sha256": "a" * 64,
        "synthetic_manifest_sha256": "b" * 64,
    }
    arguments.update(changes)
    return select_balanced_references(**arguments)


def test_balanced_unique_in_class_references_and_replay():
    real = np.repeat(np.arange(3, dtype=np.int32), [7, 3, 5])
    synthetic = np.repeat(np.arange(3, dtype=np.int32), 2000)
    result = select(real, synthetic)
    assert result == select(real, synthetic)
    for key, labels in (
        ("real_reference", real), ("synthetic_reference", synthetic),
    ):
        reference = result[key]
        assert reference["samples_per_class"] == 3
        for cls, positions in enumerate(reference["positions_by_class"]):
            assert len(positions) == len(set(positions)) == 3
            assert np.all(labels[positions] == cls)


def test_real_reference_independent_of_synthetic_order_and_identity():
    synthetic = np.repeat(np.arange(3, dtype=np.int32), 2000)
    reordered = np.random.default_rng(17).permutation(synthetic)
    first = select(synthetic=synthetic)
    second = select(
        synthetic=reordered, synthetic_manifest_sha256="c" * 64,
    )
    assert first["real_reference"] == second["real_reference"]
    assert first["real_reference_sha256"] == second["real_reference_sha256"]
    assert first["synthetic_reference_sha256"] != second["synthetic_reference_sha256"]


def test_reference_cap_is_200_per_class():
    result = select(real=np.repeat(np.arange(3, dtype=np.int32), 201))
    assert result["real_reference"]["samples_per_class"] == 200
    assert all(
        len(row) == 200
        for row in result["real_reference"]["positions_by_class"]
    )


def test_split_identity_changes_real_reference_identity():
    first = select()
    second = select(real_validation_identity_sha256="d" * 64)
    assert first["real_reference_sha256"] != second["real_reference_sha256"]
    assert (
        first["real_reference"]["positions_by_class"]
        == second["real_reference"]["positions_by_class"]
    )


@pytest.mark.parametrize("fault", [
    "missing_class", "singleton_class", "out_of_range",
    "floating_labels", "incomplete_synthetic", "invalid_identity",
])
def test_invalid_primary_reference_inputs(fault):
    changes = {}
    if fault == "missing_class":
        changes["real_validation_labels"] = np.array([0, 0, 1, 1], dtype=np.int32)
    elif fault == "singleton_class":
        changes["real_validation_labels"] = np.array([0, 0, 1, 1, 2], dtype=np.int32)
    elif fault == "out_of_range":
        changes["real_validation_labels"] = np.array([0, 1, 2, 3], dtype=np.int32)
    elif fault == "floating_labels":
        changes["real_validation_labels"] = np.array([0, 1, 2], dtype=np.float32)
    elif fault == "incomplete_synthetic":
        changes["synthetic_labels"] = np.repeat(np.arange(3, dtype=np.int32), 1999)
    else:
        changes["real_validation_identity_sha256"] = "not-a-digest"
    with pytest.raises(ValueError):
        select(**changes)
