"""Tests for governed GMM class-pool publication."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
import pytest

from trustforge.models.paper01_revision_2026.gaussian_mixture_class_pool import (
    BATCH_COUNT,
    BATCH_SIZE,
    SAMPLING_PURPOSE,
    SAMPLE_COUNT,
    publish_gmm_class_pool,
    verify_gmm_class_pool,
)
from trustforge.provenance import canonical_json_bytes


DATASET_ID = "cicmaldroid2020"
CLASS_ID = 0
SEED = 42
TRAIN_COMMIT = "b" * 40
FREEZE_SHA256 = "c" * 64
SYNTHESIS_COMMIT = "d" * 40
STATE_SHA256 = "e" * 64
SAMPLING_IDENTITY = "f" * 64
PURPOSE = SAMPLING_PURPOSE


def _images():
    values = np.linspace(
        0.0, 1.0,
        num=SAMPLE_COUNT * 40 * 40 * 3,
        dtype=np.float32,
    )
    return values.reshape((SAMPLE_COUNT, 40, 40, 3))


def _result(**overrides):
    values = {
        "images": _images(),
        "dataset_id": DATASET_ID,
        "class_id": CLASS_ID,
        "experimental_seed": SEED,
        "sampling_identity_sha256": SAMPLING_IDENTITY,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _metadata_raw(**overrides):
    metadata = {
        "dataset_id": DATASET_ID,
        "class_id": CLASS_ID,
        "experimental_seed": SEED,
        "state_sha256": STATE_SHA256,
        "execution_evidence": {
            "schema": "trustforge.paper01.gmm.execution-binding.v1",
            "expected_source_commit": TRAIN_COMMIT,
            "source_freeze_sha256": FREEZE_SHA256,
            "source_files_sha256": {
                "scripts/train_paper01_gmm_revision_production.py": "a" * 64,
            },
            "historical_git_baseline": "1" * 40,
            "implementation_parent_revision": "2" * 40,
            "governed_experimental_seeds": list(range(42, 49)),
            "dataset_adapter": "CICMalDroidRevision2026",
            "dataset_source_paths": {"cic_source_csv": "/source.csv"},
        },
    }
    metadata.update(overrides)
    return canonical_json_bytes(metadata)


def _publish(tmp_path, *, result=None, metadata_raw=None, **overrides):
    destination = tmp_path / "pool"
    arguments = {
        "sampling_result": _result() if result is None else result,
        "expected_sampling_identity_sha256": SAMPLING_IDENTITY,
        "sampling_purpose": PURPOSE,
        "checkpoint_metadata_raw": (
            _metadata_raw() if metadata_raw is None else metadata_raw
        ),
        "checkpoint_state_sha256": STATE_SHA256,
        "expected_training_source_commit": TRAIN_COMMIT,
        "expected_training_source_freeze_sha256": FREEZE_SHA256,
        "synthesis_source_commit": SYNTHESIS_COMMIT,
    }
    arguments.update(overrides)
    return destination, publish_gmm_class_pool(destination, **arguments)


def test_publishes_exactly_2000_images_in_250_ordered_batches(tmp_path):
    destination, report = _publish(tmp_path)
    assert report["sample_count"] == 2000
    assert report["batch_count"] == 250
    assert report["class_id"] == 0
    assert report["experimental_seed"] == 42
    assert report["sampling_identity_sha256"] == SAMPLING_IDENTITY
    assert report["training_source_commit"] == TRAIN_COMMIT
    assert report["training_source_freeze_sha256"] == FREEZE_SHA256
    assert report["synthesis_source_commit"] == SYNTHESIS_COMMIT

    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    assert manifest["schema"] == "trustforge.paper01.gmm-class-pool.v1"
    assert manifest["sample_count"] == SAMPLE_COUNT
    assert len(manifest["batches"]) == BATCH_COUNT
    images = _images()
    for index in (0, 1, 249):
        record = manifest["batches"][index]
        assert record["batch_index"] == index
        assert record["start_sample"] == index * BATCH_SIZE
        assert record["stop_sample"] == (index + 1) * BATCH_SIZE
        raw = (destination / record["filename"]).read_bytes()
        assert record["sha256"] == hashlib.sha256(
            raw
        ).hexdigest()
        batch = np.load(
            io.BytesIO(raw), allow_pickle=False,
        )
        assert np.array_equal(
            batch, images[index * BATCH_SIZE:(index + 1) * BATCH_SIZE],
        )


def test_identical_result_is_reproducibly_published(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    arguments = {
        "sampling_result": _result(),
        "expected_sampling_identity_sha256": SAMPLING_IDENTITY,
        "sampling_purpose": PURPOSE,
        "checkpoint_metadata_raw": _metadata_raw(),
        "checkpoint_state_sha256": STATE_SHA256,
        "expected_training_source_commit": TRAIN_COMMIT,
        "expected_training_source_freeze_sha256": FREEZE_SHA256,
        "synthesis_source_commit": SYNTHESIS_COMMIT,
    }
    first_report = publish_gmm_class_pool(first, **arguments)
    second_report = publish_gmm_class_pool(second, **arguments)
    assert first_report["manifest_sha256"] == second_report["manifest_sha256"]
    assert (first / "manifest.json").read_bytes() == (
        second / "manifest.json"
    ).read_bytes()


@pytest.mark.parametrize(
    "kwargs",
    (
        {"expected_sampling_identity_sha256": "0" * 64},
        {"sampling_result": _result(dataset_id="malex")},
        {"sampling_result": _result(class_id=1)},
        {"sampling_result": _result(experimental_seed=43)},
        {"expected_training_source_commit": "3" * 40},
        {"expected_training_source_freeze_sha256": "4" * 64},
        {"sampling_purpose": "paper01_revision2026_gmm_resampling"},
        {"sampling_purpose": " paper01_revision2026_gmm_sampling "},
    ),
)
def test_scientific_identity_mismatch_is_rejected(tmp_path, kwargs):
    with pytest.raises(ValueError):
        _publish(tmp_path, **kwargs)


@pytest.mark.parametrize(
    "images,error",
    (
        (np.zeros((1999, 40, 40, 3), dtype=np.float32), ValueError),
        (np.zeros((2000, 40, 40, 3), dtype=np.float64), TypeError),
        (np.full((2000, 40, 40, 3), np.nan, dtype=np.float32), ValueError),
        (np.full((2000, 40, 40, 3), 1.1, dtype=np.float32), ValueError),
    ),
)
def test_corrupted_sampling_output_is_rejected(tmp_path, images, error):
    with pytest.raises(error):
        _publish(tmp_path, result=_result(images=images))


def test_missing_or_corrupted_batch_is_rejected(tmp_path):
    destination, _ = _publish(tmp_path)
    batch = destination / "class0-b100.npy"
    batch.unlink()
    with pytest.raises(ValueError, match="regular file"):
        verify_gmm_class_pool(destination / "manifest.json")


def test_malformed_persisted_batch_is_rejected(tmp_path):
    destination, _ = _publish(tmp_path)
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    record = manifest["batches"][100]
    bad = np.zeros((7, 40, 40, 3), dtype=np.float32)
    buffer = io.BytesIO()
    np.save(buffer, bad, allow_pickle=False)
    raw = buffer.getvalue()
    (destination / record["filename"]).write_bytes(raw)
    record["sha256"] = hashlib.sha256(raw).hexdigest()
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")
    with pytest.raises(ValueError, match="batch shape"):
        verify_gmm_class_pool(manifest_path)


def test_each_batch_is_fsynced_before_manifest_publication(
    tmp_path,
    monkeypatch,
):
    import os as os_module

    from trustforge.models.paper01_revision_2026 import (
        gaussian_mixture_class_pool as class_pool,
    )

    real_fsync = os_module.fsync
    real_link = os_module.link
    synced_inodes = []
    fsync_counts_at_link = []

    def recording_fsync(descriptor):
        real_fsync(descriptor)
        synced_inodes.append(os_module.fstat(descriptor).st_ino)

    def recording_link(source, target):
        fsync_counts_at_link.append(len(synced_inodes))
        return real_link(source, target)

    monkeypatch.setattr(class_pool.os, "fsync", recording_fsync)
    monkeypatch.setattr(class_pool.os, "link", recording_link)
    destination, _ = _publish(tmp_path)

    assert len(fsync_counts_at_link) == 1
    assert fsync_counts_at_link[0] >= BATCH_COUNT
    batch_inodes = {
        (destination / f"class0-b{index:03d}.npy").stat().st_ino
        for index in range(BATCH_COUNT)
    }
    assert len(batch_inodes) == BATCH_COUNT
    assert batch_inodes <= set(synced_inodes)


def test_modified_persisted_manifest_purpose_is_rejected(tmp_path):
    destination, _ = _publish(tmp_path)
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["sampling_purpose"] = "paper01_revision2026_gmm_resampling"
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")
    with pytest.raises(ValueError, match="sampling purpose"):
        verify_gmm_class_pool(manifest_path)


def test_governed_sampling_purpose_matches_sampler_source():
    root = Path(__file__).resolve().parents[1]
    source = (
        root
        / "src/trustforge/models/paper01_revision_2026"
        / "gaussian_mixture_sampling.py"
    ).read_text(encoding="utf-8")
    match = re.search(
        r'^SAMPLING_PURPOSE = "([^"]+)"$',
        source,
        flags=re.MULTILINE,
    )
    assert match is not None
    assert match.group(1) == SAMPLING_PURPOSE


def test_partial_publication_without_manifest_is_not_complete(tmp_path):
    destination = tmp_path / "partial"
    destination.mkdir()
    buffer = io.BytesIO()
    np.save(buffer, _images()[:8], allow_pickle=False)
    (destination / "class0-b000.npy").write_bytes(buffer.getvalue())
    with pytest.raises(ValueError, match="manifest"):
        verify_gmm_class_pool(destination / "manifest.json")


def test_existing_destination_is_never_overwritten(tmp_path):
    destination = tmp_path / "pool"
    destination.mkdir()
    with pytest.raises(FileExistsError):
        _publish(tmp_path)
    assert list(destination.iterdir()) == []

    destination.rmdir()
    _publish(tmp_path)
    with pytest.raises(FileExistsError):
        _publish(tmp_path)
