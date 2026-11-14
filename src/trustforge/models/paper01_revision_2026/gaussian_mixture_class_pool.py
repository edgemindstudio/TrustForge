"""Publish one validated GMM class sample pool from a governed checkpoint."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import re

import numpy as np

from trustforge.provenance import canonical_json_bytes


SCHEMA = "trustforge.paper01.gmm-class-pool.v1"
SAMPLING_PURPOSE = "paper01_revision2026_gmm_sampling"
BATCH_SIZE = 8
SAMPLE_COUNT = 2000
BATCH_COUNT = SAMPLE_COUNT // BATCH_SIZE
CANONICAL_SHAPE = (40, 40, 3)
POOL_SHAPE = (SAMPLE_COUNT, *CANONICAL_SHAPE)
HEX64 = re.compile(r"^[0-9a-f]{64}$")
COMMIT = re.compile(r"^[0-9a-f]{40}$")


def _digest(value, *, name):
    if type(value) is not str or HEX64.fullmatch(value) is None:
        raise ValueError(name + " must be a lowercase SHA-256")
    return value


def _commit(value, *, name):
    if type(value) is not str or COMMIT.fullmatch(value) is None:
        raise ValueError(name + " must be a 40-character commit")
    return value


def _require_regular_file(path, *, name):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(name + " must be a regular file")
    return path


def _validate_images(images):
    images = np.asarray(images)
    if images.shape != POOL_SHAPE:
        raise ValueError("GMM pool must contain exactly 2,000 canonical images")
    if images.dtype != np.float32:
        raise TypeError("GMM pool images must be float32")
    if not np.isfinite(images).all():
        raise ValueError("GMM pool images contain non-finite values")
    if np.any(images < 0.0) or np.any(images > 1.0):
        raise ValueError("GMM pool images escaped [0,1]")
    return images


def _validate_batch(batch, *, name):
    batch = np.asarray(batch)
    if batch.shape != (BATCH_SIZE, *CANONICAL_SHAPE):
        raise ValueError(name + " has invalid canonical batch shape")
    if batch.dtype != np.float32:
        raise TypeError(name + " must be float32")
    if not np.isfinite(batch).all():
        raise ValueError(name + " contains non-finite values")
    if np.any(batch < 0.0) or np.any(batch > 1.0):
        raise ValueError(name + " escaped [0,1]")
    return batch


def _checkpoint_metadata(metadata_raw, *, expected_state_sha256):
    if type(metadata_raw) is not bytes or not metadata_raw:
        raise ValueError("checkpoint metadata bytes required")
    try:
        metadata = json.loads(metadata_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid checkpoint metadata JSON") from exc
    if type(metadata) is not dict:
        raise ValueError("checkpoint metadata must be an object")
    if (
        metadata.get("dataset_id") != "cicmaldroid2020"
        or metadata.get("class_id") != 0
        or metadata.get("experimental_seed") != 42
    ):
        raise ValueError("checkpoint scientific identity mismatch")
    state_sha256 = _digest(
        metadata.get("state_sha256"), name="checkpoint state digest",
    )
    if state_sha256 != _digest(
        expected_state_sha256, name="checkpoint state digest",
    ):
        raise ValueError("checkpoint state digest mismatch")
    evidence = metadata.get("execution_evidence")
    if type(evidence) is not dict:
        raise ValueError("checkpoint training execution evidence required")
    return metadata, evidence


def _fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _verify_manifest(manifest, *, directory, manifest_sha256):
    if (
        type(manifest) is not dict
        or manifest.get("schema") != SCHEMA
        or manifest.get("dataset_id") != "cicmaldroid2020"
        or manifest.get("family") != "gmm"
        or manifest.get("class_id") != 0
        or manifest.get("experimental_seed") != 42
        or manifest.get("sample_count") != SAMPLE_COUNT
        or manifest.get("batch_size") != BATCH_SIZE
        or manifest.get("batch_count") != BATCH_COUNT
    ):
        raise ValueError("GMM class-pool manifest identity mismatch")
    if manifest.get("canonical_output") != {
        "dtype": "float32",
        "image_shape": [40, 40, 3],
        "pool_shape": [2000, 40, 40, 3],
        "range": [0.0, 1.0],
    }:
        raise ValueError("GMM class-pool canonical output mismatch")
    _digest(
        manifest.get("sampling_identity_sha256"),
        name="GMM sampling identity",
    )
    if manifest.get("sampling_purpose") != SAMPLING_PURPOSE:
        raise ValueError("GMM sampling purpose mismatch")
    checkpoint = manifest.get("checkpoint")
    training = manifest.get("training_source")
    if (
        type(checkpoint) is not dict
        or set(checkpoint) != {"metadata_file_sha256", "state_file_sha256"}
        or type(training) is not dict
        or set(training) != {
            "source_commit", "source_freeze_sha256", "source_files_sha256",
        }
    ):
        raise ValueError("GMM class-pool provenance schema mismatch")
    _digest(checkpoint["metadata_file_sha256"], name="checkpoint metadata sha256")
    _digest(checkpoint["state_file_sha256"], name="checkpoint state sha256")
    _commit(training["source_commit"], name="training source commit")
    _digest(
        training["source_freeze_sha256"],
        name="training source-freeze sha256",
    )
    if (
        not isinstance(training["source_files_sha256"], dict)
        or not training["source_files_sha256"]
    ):
        raise ValueError("training source-file evidence required")
    _commit(
        manifest.get("synthesis_source_commit"),
        name="synthesis source commit",
    )
    if manifest.get("publication_status") != "WRITTEN_VERIFIED_AND_SYNCED":
        raise ValueError("GMM class-pool publication status mismatch")
    batches = manifest.get("batches")
    if type(batches) is not list or len(batches) != BATCH_COUNT:
        raise ValueError("GMM class-pool requires 250 ordered batches")
    filenames = set()
    total = 0
    for index, record in enumerate(batches):
        if (
            type(record) is not dict
            or set(record) != {
                "batch_index", "filename", "start_sample", "stop_sample",
                "sample_count", "sha256",
            }
            or record["batch_index"] != index
            or record["start_sample"] != index * BATCH_SIZE
            or record["stop_sample"] != (index + 1) * BATCH_SIZE
            or record["sample_count"] != BATCH_SIZE
            or type(record["filename"]) is not str
            or Path(record["filename"]).name != record["filename"]
            or not record["filename"].endswith(".npy")
            or record["filename"] in filenames
        ):
            raise ValueError("invalid GMM batch record")
        _digest(record["sha256"], name="GMM batch sha256")
        filenames.add(record["filename"])
        path = _require_regular_file(
            directory / record["filename"], name="GMM batch file",
        )
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != record["sha256"]:
            raise ValueError("GMM batch digest mismatch")
        batch = np.load(io.BytesIO(raw), allow_pickle=False)
        if not isinstance(batch, np.ndarray):
            raise ValueError("GMM batch archive must be an ndarray")
        _validate_batch(batch, name="GMM batch")
        total += int(batch.shape[0])
    if total != SAMPLE_COUNT:
        raise ValueError("GMM class-pool total image count mismatch")
    return {
        "schema": SCHEMA,
        "manifest_sha256": manifest_sha256,
        "dataset_id": "cicmaldroid2020",
        "family": "gmm",
        "class_id": 0,
        "experimental_seed": 42,
        "batch_count": BATCH_COUNT,
        "sample_count": total,
        "sampling_identity_sha256": manifest["sampling_identity_sha256"],
        "checkpoint_metadata_file_sha256":
            checkpoint["metadata_file_sha256"],
        "checkpoint_state_file_sha256": checkpoint["state_file_sha256"],
        "training_source_commit": training["source_commit"],
        "training_source_freeze_sha256": training["source_freeze_sha256"],
        "synthesis_source_commit": manifest["synthesis_source_commit"],
    }


def verify_gmm_class_pool(manifest_path):
    """Independently read back a complete GMM class pool."""
    manifest_path = _require_regular_file(
        manifest_path, name="GMM class-pool manifest",
    )
    raw = manifest_path.read_bytes()
    try:
        manifest = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid GMM class-pool manifest JSON") from exc
    if raw != canonical_json_bytes(manifest) + b"\n":
        raise ValueError("noncanonical GMM class-pool manifest")
    return _verify_manifest(
        manifest,
        directory=manifest_path.parent,
        manifest_sha256=hashlib.sha256(raw).hexdigest(),
    )


def publish_gmm_class_pool(
    destination,
    *,
    sampling_result,
    expected_sampling_identity_sha256,
    sampling_purpose,
    checkpoint_metadata_raw,
    checkpoint_state_sha256,
    expected_training_source_commit,
    expected_training_source_freeze_sha256,
    synthesis_source_commit,
):
    """Persist and independently verify one complete GMM class pool."""
    sampling_identity = _digest(
        getattr(sampling_result, "sampling_identity_sha256", None),
        name="GMM sampling identity",
    )
    if sampling_identity != _digest(
        expected_sampling_identity_sha256,
        name="expected GMM sampling identity",
    ):
        raise ValueError("GMM sampling identity mismatch")
    if (
        getattr(sampling_result, "dataset_id", None) != "cicmaldroid2020"
        or getattr(sampling_result, "class_id", None) != 0
        or getattr(sampling_result, "experimental_seed", None) != 42
    ):
        raise ValueError("GMM sampling result scientific identity mismatch")
    images = _validate_images(getattr(sampling_result, "images", None))
    metadata, evidence = _checkpoint_metadata(
        checkpoint_metadata_raw,
        expected_state_sha256=checkpoint_state_sha256,
    )
    if (
        evidence.get("expected_source_commit")
        != _commit(
            expected_training_source_commit, name="training source commit",
        )
        or evidence.get("source_freeze_sha256")
        != _digest(
            expected_training_source_freeze_sha256,
            name="training source-freeze sha256",
        )
    ):
        raise ValueError("checkpoint training source identity mismatch")
    _commit(synthesis_source_commit, name="synthesis source commit")
    if sampling_purpose != SAMPLING_PURPOSE:
        raise ValueError("GMM sampling purpose mismatch")

    destination = Path(destination).expanduser()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("GMM class-pool destination already exists")
    destination.mkdir(mode=0o700)

    records = []
    batches = []
    for index in range(BATCH_COUNT):
        start = index * BATCH_SIZE
        stop = start + BATCH_SIZE
        batch = _validate_batch(images[start:stop], name="GMM batch")
        filename = f"class0-b{index:03d}.npy"
        buffer = io.BytesIO()
        np.save(buffer, batch, allow_pickle=False)
        raw = buffer.getvalue()
        with (destination / filename).open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        records.append({
            "batch_index": index,
            "filename": filename,
            "start_sample": start,
            "stop_sample": stop,
            "sample_count": BATCH_SIZE,
            "sha256": hashlib.sha256(raw).hexdigest(),
        })
        batches.append(batch)

    # Independent read-back before the manifest can represent a complete pool.
    for record, expected in zip(records, batches):
        path = _require_regular_file(
            destination / record["filename"], name="GMM batch file",
        )
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != record["sha256"]:
            raise RuntimeError("GMM batch read-back digest mismatch")
        observed = np.load(io.BytesIO(raw), allow_pickle=False)
        if not isinstance(observed, np.ndarray):
            raise RuntimeError("GMM batch read-back is not an ndarray")
        _validate_batch(observed, name="GMM batch read-back")
        if not np.array_equal(observed, expected):
            raise RuntimeError("GMM batch read-back order mismatch")

    manifest = {
        "schema": SCHEMA,
        "dataset_id": "cicmaldroid2020",
        "family": "gmm",
        "class_id": 0,
        "experimental_seed": 42,
        "sample_count": SAMPLE_COUNT,
        "batch_size": BATCH_SIZE,
        "batch_count": BATCH_COUNT,
        "canonical_output": {
            "dtype": "float32",
            "image_shape": [40, 40, 3],
            "pool_shape": [2000, 40, 40, 3],
            "range": [0.0, 1.0],
        },
        "sampling_purpose": sampling_purpose,
        "sampling_identity_sha256": sampling_identity,
        "checkpoint": {
            "metadata_file_sha256": hashlib.sha256(
                checkpoint_metadata_raw
            ).hexdigest(),
            "state_file_sha256": metadata["state_sha256"],
        },
        "training_source": {
            "source_commit": evidence["expected_source_commit"],
            "source_freeze_sha256": evidence["source_freeze_sha256"],
            "source_files_sha256": evidence["source_files_sha256"],
        },
        "synthesis_source_commit": synthesis_source_commit,
        "batches": records,
        "publication_status": "WRITTEN_VERIFIED_AND_SYNCED",
    }
    manifest_raw = canonical_json_bytes(manifest) + b"\n"
    pending = destination / ".manifest.pending"
    final = destination / "manifest.json"
    with pending.open("xb") as handle:
        handle.write(manifest_raw)
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(destination)
    if final.exists() or final.is_symlink():
        raise FileExistsError("GMM class-pool manifest already exists")
    os.link(pending, final)
    pending.unlink()
    _fsync_directory(destination)
    return verify_gmm_class_pool(final)


def sample_and_publish_gmm_class_pool(
    checkpoint_directory,
    destination,
    *,
    expected_training_source_commit,
    expected_training_source_freeze_sha256,
    synthesis_source_commit,
):
    """Sample one continuous 2,000-image class-0 result and publish it."""
    from .gaussian_mixture_checkpoint import load_gmm_checkpoint
    from .gaussian_mixture_sampling import (
        SAMPLING_PURPOSE as GOVERNED_SAMPLING_PURPOSE,
        sample_gmm_checkpoint,
        sampling_identity_sha256,
    )

    if GOVERNED_SAMPLING_PURPOSE != SAMPLING_PURPOSE:
        raise RuntimeError("governed GMM sampling purpose mismatch")

    dataset_id = "cicmaldroid2020"
    class_id = 0
    experimental_seed = 42
    result = sample_gmm_checkpoint(
        checkpoint_directory,
        dataset_id=dataset_id,
        class_id=class_id,
        experimental_seed=experimental_seed,
        sample_count=SAMPLE_COUNT,
    )
    checkpoint = load_gmm_checkpoint(
        checkpoint_directory,
        requested_dataset_id=dataset_id,
        requested_class_id=class_id,
        requested_experimental_seed=experimental_seed,
    )
    metadata_raw = _require_regular_file(
        checkpoint.metadata_path, name="GMM checkpoint metadata",
    ).read_bytes()
    state_sha256 = hashlib.sha256(
        _require_regular_file(
            checkpoint.state_path, name="GMM checkpoint state",
        ).read_bytes()
    ).hexdigest()
    return publish_gmm_class_pool(
        destination,
        sampling_result=result,
        expected_sampling_identity_sha256=sampling_identity_sha256(
            dataset_id=dataset_id,
            class_id=class_id,
            experimental_seed=experimental_seed,
        ),
        sampling_purpose=SAMPLING_PURPOSE,
        checkpoint_metadata_raw=metadata_raw,
        checkpoint_state_sha256=state_sha256,
        expected_training_source_commit=expected_training_source_commit,
        expected_training_source_freeze_sha256=(
            expected_training_source_freeze_sha256
        ),
        synthesis_source_commit=synthesis_source_commit,
    )


__all__ = [
    "BATCH_COUNT",
    "BATCH_SIZE",
    "SAMPLING_PURPOSE",
    "SAMPLE_COUNT",
    "SCHEMA",
    "publish_gmm_class_pool",
    "sample_and_publish_gmm_class_pool",
    "verify_gmm_class_pool",
]
