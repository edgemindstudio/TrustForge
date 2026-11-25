"""Focused fixture tests for Diffusion range and pool publication.

No real checkpoint, dataset, training, sampling or Slurm execution is used.
Full-quota accounting is demonstrated with inexpensive deterministic arrays.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from trustforge.models.paper01_revision_2026 import (
    diffusion_production_synthesis as synthesis,
)
from trustforge.models.paper01_revision_2026.diffusion_pool_publication import (
    ACCEPTED_SAMPLING_POLICY,
    DiffusionRangeIdentity,
    PRIMARY_SAMPLES_PER_CLASS,
    publish_diffusion_range,
    verify_diffusion_dataset_pools,
    verify_diffusion_range,
)
from trustforge.provenance import canonical_json_bytes


DATASET = "cicmaldroid2020"
MAPPING = tuple(str(index) for index in range(5))
SYNTHESIS_COMMIT = "d" * 40
TRAINING_COMMIT = "c" * 40
CHECKPOINT_MANIFEST = "a" * 64
CHECKPOINT_STATE = "b" * 64
RECIPE_SHA256 = "e" * 64
RESOURCES = {
    "partition": "talon",
    "nodes": 1,
    "tasks": 1,
    "cpus_per_task": 8,
    "memory_gb": 32,
    "walltime_seconds": 3600,
    "gpu_enabled": False,
    "max_concurrent_jobs": 1,
    "automatic_retries": 0,
}


def make_identity(
    *,
    class_id,
    sample_start,
    sample_stop,
    attempt=1,
    source=SYNTHESIS_COMMIT,
    manifest=CHECKPOINT_MANIFEST,
    state=CHECKPOINT_STATE,
    recipe=RECIPE_SHA256,
    scope="fixture",
    training_commit=TRAINING_COMMIT,
    slurm_job_id="fixture",
):
    return DiffusionRangeIdentity(
        dataset_id=DATASET,
        experimental_seed=42,
        attempt=attempt,
        class_id=class_id,
        class_label=MAPPING[class_id],
        sample_start=sample_start,
        sample_stop=sample_stop,
        checkpoint_manifest_sha256=manifest,
        checkpoint_state_sha256=state,
        checkpoint_training_source_commit=training_commit,
        synthesis_source_commit=source,
        recipe_sha256=recipe,
        sampling_policy=dict(ACCEPTED_SAMPLING_POLICY),
        authorization_id=f"P1R2026-DIFF-FIXTURE-C{class_id}",
        slurm_job_id=slurm_job_id,
        resources=dict(RESOURCES),
        scope=scope,
    )


def make_images(count, *, base=0.25):
    values = np.full(
        (count, 40, 40, 3), base, dtype=np.float32,
    )
    return values


def publish(
    tmp_path,
    *,
    class_id,
    sample_start=0,
    sample_stop=2,
    attempt=1,
    name=None,
    base=0.25,
    **identity_overrides,
):
    identity = make_identity(
        class_id=class_id,
        sample_start=sample_start,
        sample_stop=sample_stop,
        attempt=attempt,
        **identity_overrides,
    )
    destination = tmp_path / (
        name or f"class{class_id}-{sample_start}-{sample_stop}"
    )
    report = publish_diffusion_range(
        identity=identity,
        images=make_images(sample_stop - sample_start, base=base),
        destination=destination,
    )
    return destination, report


@pytest.fixture(scope="session")
def full_pool(tmp_path_factory):
    root = tmp_path_factory.mktemp("diffusion-full-pool")
    manifests = []
    for class_id in range(5):
        partitions = (
            ((0, 1000), (1000, 2000))
            if class_id == 0
            else ((0, 2000),)
        )
        for index, (start, stop) in enumerate(partitions):
            identity = make_identity(
                class_id=class_id,
                sample_start=start,
                sample_stop=stop,
                attempt=(index + 1 if class_id == 0 else class_id + 1),
                slurm_job_id=f"fixture-c{class_id}-{index}",
            )
            destination = root / f"c{class_id}-{start}"
            publish_diffusion_range(
                identity=identity,
                images=np.zeros(
                    (stop - start, 40, 40, 3), dtype=np.float32,
                ),
                destination=destination,
            )
            manifests.append(destination / "manifest.json")
    return root, manifests


def test_publish_and_verify_range_preserves_order(tmp_path):
    images = np.linspace(
        0.0, 1.0, num=10 * 40 * 40 * 3, dtype=np.float32,
    ).reshape((10, 40, 40, 3))
    identity = make_identity(
        class_id=1, sample_start=0, sample_stop=10,
    )
    destination = tmp_path / "range"
    report = publish_diffusion_range(
        identity=identity, images=images, destination=destination,
    )

    assert report["sample_count"] == 10
    assert report["storage_batch_count"] == 2
    assert report["class_label"] == "1"
    assert report["authoritative_artifact"] is False

    first = np.load(
        destination / "class1-s0000-b000.npy", allow_pickle=False,
    )
    second = np.load(
        destination / "class1-s0000-b001.npy", allow_pickle=False,
    )
    assert np.array_equal(first, images[0:8])
    assert np.array_equal(second, images[8:10])
    assert second.shape == (2, 40, 40, 3)


def test_existing_destination_rejected(tmp_path):
    destination, _ = publish(tmp_path, class_id=2)
    identity = make_identity(
        class_id=2, sample_start=0, sample_stop=2,
    )
    with pytest.raises(FileExistsError):
        publish_diffusion_range(
            identity=identity,
            images=make_images(2),
            destination=destination,
        )


def test_corrupted_batch_rejected(tmp_path):
    destination, _ = publish(tmp_path, class_id=3)
    batch = destination / "class3-s0000-b000.npy"
    batch.write_bytes(batch.read_bytes() + b"\x00")

    with pytest.raises(ValueError, match="digest mismatch"):
        verify_diffusion_range(destination / "manifest.json")


def test_missing_batch_rejected(tmp_path):
    destination, _ = publish(tmp_path, class_id=3)
    (destination / "class3-s0000-b000.npy").unlink()

    with pytest.raises(ValueError, match="regular file"):
        verify_diffusion_range(destination / "manifest.json")


def test_substituted_batch_rejected(tmp_path):
    destination, _ = publish(tmp_path, class_id=3)
    replacement = tmp_path / "replacement.npy"
    np.save(
        replacement,
        make_images(2, base=0.75),
        allow_pickle=False,
    )
    (destination / "class3-s0000-b000.npy").write_bytes(
        replacement.read_bytes()
    )

    with pytest.raises(ValueError, match="digest mismatch"):
        verify_diffusion_range(destination / "manifest.json")


def test_incomplete_publication_rejected(tmp_path):
    destination, _ = publish(tmp_path, class_id=4)
    (destination / "manifest.json").unlink()

    with pytest.raises(ValueError, match="regular file"):
        verify_diffusion_range(destination / "manifest.json")


def test_noncanonical_manifest_rejected(tmp_path):
    destination, _ = publish(tmp_path, class_id=4)
    manifest = json.loads(
        (destination / "manifest.json").read_bytes()
    )
    (destination / "manifest.json").write_bytes(
        json.dumps(manifest, indent=2).encode("utf-8")
    )

    with pytest.raises(ValueError, match="noncanonical"):
        verify_diffusion_range(destination / "manifest.json")


@pytest.mark.parametrize(
    "expectation",
    [
        {"expected_dataset_id": "malex"},
        {"expected_seed": 43},
        {"expected_class_id": 1},
        {"expected_class_label": "1"},
        {"expected_checkpoint_manifest_sha256": "f" * 64},
        {"expected_checkpoint_state_sha256": "f" * 64},
        {"expected_recipe_sha256": "f" * 64},
    ],
)
def test_wrong_expectation_rejected(tmp_path, expectation):
    destination, _ = publish(tmp_path, class_id=0)

    with pytest.raises(ValueError, match="expectation mismatch"):
        verify_diffusion_range(
            destination / "manifest.json", **expectation,
        )


def test_tampered_label_rejected_without_expectation(tmp_path):
    destination, _ = publish(tmp_path, class_id=0)

    def mutate(manifest):
        manifest["class_label"] = "9"

    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    mutate(manifest)
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")

    with pytest.raises(ValueError, match="authoritative"):
        verify_diffusion_range(manifest_path)


def test_tampered_run_identity_rejected(tmp_path):
    destination, _ = publish(tmp_path, class_id=0)
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["run_identity"] = "commit0000000_attempt01"
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")

    with pytest.raises(ValueError, match="run identity"):
        verify_diffusion_range(manifest_path)


def test_tampered_policy_rejected(tmp_path):
    destination, _ = publish(tmp_path, class_id=0)
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["sampling_policy"]["timesteps"] = 10
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")

    with pytest.raises(ValueError, match="sampling policy"):
        verify_diffusion_range(manifest_path)


def test_complete_pool_accounting_with_partitioned_class(full_pool):
    _, manifests = full_pool
    report = verify_diffusion_dataset_pools(
        manifests,
        expected_dataset_id=DATASET,
        expected_seed=42,
        expected_class_mapping=MAPPING,
        expected_checkpoint_manifest_sha256=CHECKPOINT_MANIFEST,
        expected_checkpoint_state_sha256=CHECKPOINT_STATE,
        expected_recipe_sha256=RECIPE_SHA256,
    )

    assert report["class_ids"] == [0, 1, 2, 3, 4]
    assert report["class_labels"] == list(MAPPING)
    assert report["sample_count_per_class"] == 2000
    assert report["total_samples"] == 10000
    assert report["class_count"] == 5
    assert report["classes"][0]["range_count"] == 2
    assert report["classes"][1]["range_count"] == 1
    assert report["authoritative_artifact"] is False
    assert report["checkpoint_training_source_commit"] == TRAINING_COMMIT

    class0_jobs = {
        item["slurm_job_id"]
        for item in report["classes"][0]["manifests"]
    }
    class0_attempts = {
        item["attempt"] for item in report["classes"][0]["manifests"]
    }
    assert len(class0_jobs) == 2
    assert class0_attempts == {1, 2}


def test_missing_class_rejected(full_pool):
    _, manifests = full_pool

    with pytest.raises(ValueError, match="incomplete or duplicated"):
        verify_diffusion_dataset_pools(
            [path for path in manifests
             if path.parent.name != "c4-0"]
        )


def test_duplicate_range_rejected(full_pool):
    _, manifests = full_pool
    class0 = [path for path in manifests if path.parent.name.startswith("c0-")]

    with pytest.raises(ValueError, match="overlap|duplicate"):
        verify_diffusion_dataset_pools(manifests + [class0[0]])


def test_overlapping_range_rejected(tmp_path, full_pool):
    _, manifests = full_pool
    extra, _ = publish(
        tmp_path,
        class_id=0,
        sample_start=500,
        sample_stop=1000,
        attempt=9,
        name="overlap",
    )

    with pytest.raises(ValueError, match="overlap"):
        verify_diffusion_dataset_pools(
            manifests + [extra / "manifest.json"]
        )


def test_mixed_checkpoint_rejected(tmp_path, full_pool):
    _, manifests = full_pool
    extra, _ = publish(
        tmp_path,
        class_id=2,
        sample_start=0,
        sample_stop=2000,
        attempt=8,
        name="mixed",
        manifest="f" * 64,
    )

    with pytest.raises(ValueError, match="share one checkpoint|overlap"):
        verify_diffusion_dataset_pools(
            [path for path in manifests
             if path.parent.name != "c2-0"]
            + [extra / "manifest.json"]
        )


def test_complete_pool_verification_does_not_restore_or_sample(
    full_pool, monkeypatch,
):
    _, manifests = full_pool

    def forbidden(*args, **kwargs):
        raise AssertionError("verification restored or sampled")

    monkeypatch.setattr(
        synthesis, "restore_diffusion_model_for_synthesis", forbidden,
    )
    monkeypatch.setattr(
        synthesis, "synthesize_diffusion_batch", forbidden,
    )

    report = verify_diffusion_dataset_pools(manifests)
    assert report["total_samples"] == 10000


def test_pending_marker_rejects_otherwise_valid_range(tmp_path):
    destination, _ = publish(tmp_path, class_id=0)
    marker = destination / ".range.pending"
    marker.write_text("interrupted publication", encoding="utf-8")

    with pytest.raises(ValueError, match="interrupted"):
        verify_diffusion_range(destination / "manifest.json")

    assert marker.is_file()
    assert marker.read_text(encoding="utf-8") == "interrupted publication"


def test_dataset_rejects_interrupted_range(full_pool):
    _, manifests = full_pool
    marker = manifests[0].parent / ".range.pending"
    marker.write_text("interrupted publication", encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="interrupted"):
            verify_diffusion_dataset_pools(manifests)
    finally:
        marker.unlink()


def test_contradictory_training_source_rejected(full_pool):
    _, manifests = full_pool
    target = manifests[-1]
    original = target.read_bytes()
    manifest = json.loads(original)
    manifest["checkpoint"]["training_source_commit"] = "f" * 40
    target.write_bytes(canonical_json_bytes(manifest) + b"\n")
    try:
        with pytest.raises(ValueError, match="checkpoint identity"):
            verify_diffusion_dataset_pools(manifests)
    finally:
        target.write_bytes(original)
