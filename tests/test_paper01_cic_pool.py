"""Synthetic fixtures; full coverage uses hard links to one tiny batch file."""

import hashlib
import json
import os

import numpy as np
import pytest

from trustforge.evaluation.paper01_cic_pool import CICPoolView
from trustforge.models.paper01_revision_2026.ar_maf_execution_inputs import (
    synthesis_batch_seed,
)


@pytest.fixture
def pool(tmp_path):
    first = tmp_path / "c0_b0.npy"
    np.save(first, np.zeros((8, 40, 40, 3), dtype=np.float32))
    digest = hashlib.sha256(first.read_bytes()).hexdigest()
    records = []
    for cls in range(5):
        for batch in range(250):
            name = f"c{cls}_b{batch}.npy"
            if (cls, batch) != (0, 0):
                os.link(first, tmp_path / name)
            records.append({
                "class_id": cls, "batch_index": batch,
                "filename": name, "sha256": digest,
                "sampling_seed": synthesis_batch_seed(
                    dataset_id="cicmaldroid2020", family="maf",
                    experimental_seed=42, class_id=cls, batch_index=batch,
                ),
            })
    document = {
        "schema_version": 1, "dataset_id": "cicmaldroid2020",
        "family": "maf", "experimental_seed": 42,
        "recipe_sha256": "a" * 64, "checkpoint_sha256": "b" * 64,
        "batches": records,
    }

    def open_view():
        path = tmp_path / "manifest.json"
        raw = json.dumps(document).encode()
        path.write_bytes(raw)
        return CICPoolView(
            path, manifest_sha256=hashlib.sha256(raw).hexdigest(),
            family="maf", experimental_seed=42,
            recipe_sha256="a" * 64, checkpoint_sha256="b" * 64,
        )
    return tmp_path, document, open_view


def test_complete_preflight_and_position_order(pool):
    _, _, open_view = pool
    view = open_view()
    result = view.preflight()
    assert result["verified_batches"] == 1250
    assert result["verified_images"] == 10000
    positions = np.array([9999, 0, 2000, 7], dtype=np.int64)
    observed, images, labels = view.batch(positions)
    np.testing.assert_array_equal(observed, positions)
    np.testing.assert_array_equal(labels.argmax(axis=1), [4, 0, 1, 0])
    assert images.shape == (4, 40, 40, 3)


def test_manifest_construction_is_lazy(pool, monkeypatch):
    _, _, open_view = pool

    def forbidden(*args, **kwargs):
        raise AssertionError("image decoding during construction")

    monkeypatch.setattr(np, "load", forbidden)
    view = open_view()
    assert view.size == 10000
    with pytest.raises(RuntimeError, match="preflight"):
        view.batch(np.array([0], dtype=np.int64))


@pytest.mark.parametrize("fault", ["missing", "duplicate", "seed", "filename"])
def test_manifest_coverage_and_identity_rejection(pool, fault):
    _, document, open_view = pool
    records = document["batches"]
    if fault == "missing":
        records.pop()
    elif fault == "duplicate":
        records[-1] = dict(records[0])
    elif fault == "seed":
        records[0]["sampling_seed"] += 1
    else:
        records[0]["filename"] = "../outside.npy"
    with pytest.raises(ValueError):
        open_view()


def test_missing_file_prevents_preflight(pool):
    root, _, open_view = pool
    view = open_view()
    (root / "c4_b249.npy").unlink()
    with pytest.raises(FileNotFoundError):
        view.preflight()
    with pytest.raises(RuntimeError, match="preflight"):
        view.batch(np.array([0], dtype=np.int64))


def test_file_changed_after_preflight_is_rejected(pool):
    root, _, open_view = pool
    view = open_view()
    view.preflight()
    # Replace one directory entry without modifying other hard-linked fixtures.
    path = root / "c0_b0.npy"
    path.unlink()
    np.save(path, np.ones((8, 40, 40, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="digest mismatch"):
        view.batch(np.array([0], dtype=np.int64))
