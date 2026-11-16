"""Fixture tests for governed GAN pool publication and evaluation input."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from trustforge.evaluation.paper01_combined_inputs import (
    CombinedTrainingView,
)
from trustforge.evaluation.paper01_gan_pool import GANPoolView
from trustforge.evaluation.paper01_inputs import RealSplitView
from trustforge.models.paper01_revision_2026 import (
    gan_cic_pool_publication as publication,
)
from trustforge.models.paper01_revision_2026.gan_cic_pool_publication import (
    GAN_POOL_BATCHES_PER_CLASS,
    GAN_POOL_BATCH_SIZE,
    GAN_POOL_TOTAL,
)
from trustforge.models.paper01_revision_2026.synthesis_linkage import (
    derive_latent_vectors,
    load_synthesis_recipe,
)
from trustforge.provenance import canonical_json_bytes


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXECUTABLE = "abc1234" + "0" * 33
CHECKPOINT_NAME = "commit42c40fc_attempt01"


def metadata(*, seed=42, mapping=None):
    mapping = list(mapping or ("0", "1", "2", "3", "4"))
    return {
        "schema_version": "3.0",
        "schema_type": "paper01_revision2026_conditional_gan",
        "paper_revision": "paper01_revision_2026",
        "model_family": "conditional_gan",
        "dataset_id": "cicmaldroid2020",
        "experimental_seed": seed,
        "class_mapping": mapping,
        "class_count": len(mapping),
        "generator_iteration": 500,
        "discriminator_updates": 1000,
        "planned_generator_iterations": 500,
        "checkpoint_selection": "terminal_governed_iteration",
        "state_files": {
            "model_state.ckpt.index": "a" * 64,
            "model_state.ckpt.data-00000-of-00001": "b" * 64,
        },
        "dataset_provenance_file": "gan_dataset_provenance.json",
        "dataset_provenance_sha256": "c" * 64,
        "exposure_evidence_file": "gan_exposure_evidence.json",
        "exposure_evidence_sha256": "d" * 64,
        "image_shape": [40, 40, 3],
        "latent_dim": 100,
        "learning_rate": 0.0002,
        "beta_1": 0.5,
        "class_sampling_alpha": 0.85,
        "batch_size": 256,
    }


class RecordingGenerator:
    def __init__(self, *, value=0.0, fail_after=None):
        self.value = value
        self.fail_after = fail_after
        self.calls = []

    def __call__(self, inputs, training=False):
        if (
            self.fail_after is not None
            and len(self.calls) >= self.fail_after
        ):
            raise RuntimeError("injected generator failure")

        latents, labels = inputs
        self.calls.append((np.array(latents), np.array(labels)))
        return np.full(
            (len(latents), 40, 40, 3),
            self.value,
            dtype=np.float32,
        )


def install_loader(monkeypatch, *, meta, generator):
    monkeypatch.setattr(
        publication,
        "capture_git_state",
        lambda root: SimpleNamespace(commit=EXECUTABLE, dirty=False),
    )
    monkeypatch.setattr(
        publication,
        "load_terminal_gan_checkpoint_v3",
        lambda *args, **kwargs: {
            "metadata": meta,
            "generator": generator,
        },
    )


def checkpoint_dir(tmp_path):
    path = tmp_path / CHECKPOINT_NAME
    path.mkdir()
    return path


@pytest.fixture(scope="module")
def published(tmp_path_factory):
    root = tmp_path_factory.mktemp("gan-pool")
    checkpoint = root / CHECKPOINT_NAME
    checkpoint.mkdir()
    meta = metadata()
    generator = RecordingGenerator()
    loader_calls = []

    monkeypatch = pytest.MonkeyPatch()

    def loader(*args, **kwargs):
        loader_calls.append((args, kwargs))
        return {
            "metadata": meta,
            "generator": generator,
        }

    monkeypatch.setattr(
        publication,
        "capture_git_state",
        lambda root: SimpleNamespace(commit=EXECUTABLE, dirty=False),
    )
    monkeypatch.setattr(
        publication,
        "load_terminal_gan_checkpoint_v3",
        loader,
    )

    try:
        manifest_path = publication.publish_gan_cic_pool(
            root / "pool",
            checkpoint_directory=checkpoint,
            dataset_id="cicmaldroid2020",
            experimental_seed=42,
            repository_root=REPOSITORY_ROOT,
        )
    finally:
        monkeypatch.undo()

    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)

    return SimpleNamespace(
        root=root,
        checkpoint=checkpoint,
        manifest_path=manifest_path,
        manifest=manifest,
        manifest_sha256=hashlib.sha256(raw).hexdigest(),
        metadata=meta,
        generator=generator,
        loader_calls=loader_calls,
        recipe=load_synthesis_recipe(REPOSITORY_ROOT),
    )


def view_for(published):
    return GANPoolView(
        published.manifest_path,
        manifest_sha256=published.manifest_sha256,
        family="gan",
        experimental_seed=42,
        recipe_sha256=published.recipe.sha256,
        checkpoint_sha256=hashlib.sha256(
            canonical_json_bytes(published.metadata)
        ).hexdigest(),
    )


def test_complete_pool_manifest_and_batches(published):
    manifest = published.manifest

    assert manifest["schema"] == publication.GAN_POOL_SCHEMA
    assert manifest["status"] == "COMPLETE"
    assert manifest["complete_pool"] is True
    assert manifest["authoritative_artifact"] is True
    assert manifest["expected_total"] == GAN_POOL_TOTAL
    assert manifest["observed_total"] == GAN_POOL_TOTAL
    assert len(manifest["batches"]) == 5 * GAN_POOL_BATCHES_PER_CLASS
    assert manifest["class_mapping"] == ["0", "1", "2", "3", "4"]
    assert manifest["checkpoint_run_identity"] == CHECKPOINT_NAME
    assert manifest["executable_source_commit"] == EXECUTABLE
    assert manifest["checkpoint_sha256"] == hashlib.sha256(
        canonical_json_bytes(published.metadata)
    ).hexdigest()

    first = np.load(
        published.manifest_path.parent / "c0-b000.npy",
        allow_pickle=False,
    )
    assert first.shape == (GAN_POOL_BATCH_SIZE, 40, 40, 3)
    assert first.dtype == np.float32
    np.testing.assert_array_equal(first, np.float32(0.5))

    assert len(published.generator.calls) == (
        5 * GAN_POOL_BATCHES_PER_CLASS
    )
    latents, labels = published.generator.calls[0]
    expected = derive_latent_vectors(
        dataset_id="cicmaldroid2020",
        family="gan",
        experimental_seed=42,
        class_id=0,
        sample_indices=tuple(range(GAN_POOL_BATCH_SIZE)),
    )
    np.testing.assert_array_equal(latents, expected)
    np.testing.assert_array_equal(
        labels[:, 0], np.ones(GAN_POOL_BATCH_SIZE, dtype=np.float32)
    )

    later = published.generator.calls[3 * GAN_POOL_BATCHES_PER_CLASS + 77]
    later_indices = tuple(range(77 * 16, (77 + 1) * 16))
    np.testing.assert_array_equal(
        later[0],
        derive_latent_vectors(
            dataset_id="cicmaldroid2020",
            family="gan",
            experimental_seed=42,
            class_id=3,
            sample_indices=later_indices,
        ),
    )


def test_publisher_uses_strict_loader(published):
    assert len(published.loader_calls) == 1
    args, kwargs = published.loader_calls[0]

    assert Path(args[0]) == published.checkpoint
    assert kwargs == {
        "requested_dataset_id": "cicmaldroid2020",
        "requested_experimental_seed": 42,
        "requested_class_mapping": ["0", "1", "2", "3", "4"],
    }


def test_invalid_checkpoint_rejected(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    destination = tmp_path / "pool"

    def reject(*args, **kwargs):
        raise FileNotFoundError("checkpoint missing")

    monkeypatch.setattr(
        publication,
        "load_terminal_gan_checkpoint_v3",
        reject,
    )

    with pytest.raises(FileNotFoundError):
        publication.publish_gan_cic_pool(
            destination,
            checkpoint_directory=checkpoint,
            dataset_id="cicmaldroid2020",
            experimental_seed=42,
            repository_root=REPOSITORY_ROOT,
        )

    assert not destination.exists()


def test_dirty_execution_source_rejected(tmp_path, monkeypatch):
    checkpoint = checkpoint_dir(tmp_path)
    destination = tmp_path / "pool"

    install_loader(
        monkeypatch,
        meta=metadata(),
        generator=RecordingGenerator(),
    )
    monkeypatch.setattr(
        publication,
        "capture_git_state",
        lambda root: SimpleNamespace(commit=EXECUTABLE, dirty=True),
    )

    with pytest.raises(RuntimeError, match="dirty"):
        publication.publish_gan_cic_pool(
            destination,
            checkpoint_directory=checkpoint,
            dataset_id="cicmaldroid2020",
            experimental_seed=42,
            repository_root=REPOSITORY_ROOT,
        )

    assert not destination.exists()


def test_publication_source_is_distinct_from_checkpoint_identity(published):
    manifest = published.manifest

    assert manifest["checkpoint_run_identity"] == CHECKPOINT_NAME
    assert manifest["executable_source_commit"] == EXECUTABLE
    assert not CHECKPOINT_NAME.startswith(
        f"commit{EXECUTABLE[:7]}_attempt"
    )


def test_checkpoint_identity_mismatch_rejected(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    install_loader(
        monkeypatch,
        meta=metadata(seed=43),
        generator=RecordingGenerator(),
    )
    destination = tmp_path / "pool"

    with pytest.raises(ValueError, match="identity"):
        publication.publish_gan_cic_pool(
            destination,
            checkpoint_directory=checkpoint,
            dataset_id="cicmaldroid2020",
            experimental_seed=42,
            repository_root=REPOSITORY_ROOT,
        )

    assert not destination.exists()


def test_generator_injection_rejected(tmp_path):
    parameters = inspect.signature(
        publication.publish_gan_cic_pool
    ).parameters
    assert "generator" not in parameters
    assert "checkpoint_sha256" not in parameters

    with pytest.raises(TypeError):
        publication.publish_gan_cic_pool(
            tmp_path / "pool",
            checkpoint_directory=tmp_path,
            dataset_id="cicmaldroid2020",
            experimental_seed=42,
            repository_root=REPOSITORY_ROOT,
            generator=RecordingGenerator(),
        )


def test_existing_destination_rejected(tmp_path, monkeypatch):
    checkpoint = checkpoint_dir(tmp_path)
    destination = tmp_path / "pool"
    destination.mkdir()

    install_loader(
        monkeypatch,
        meta=metadata(),
        generator=RecordingGenerator(),
    )

    with pytest.raises(FileExistsError):
        publication.publish_gan_cic_pool(
            destination,
            checkpoint_directory=checkpoint,
            dataset_id="cicmaldroid2020",
            experimental_seed=42,
            repository_root=REPOSITORY_ROOT,
        )

    assert list(destination.iterdir()) == []


def test_incomplete_publication_has_no_manifest(tmp_path, monkeypatch):
    checkpoint = checkpoint_dir(tmp_path)
    destination = tmp_path / "pool"

    install_loader(
        monkeypatch,
        meta=metadata(),
        generator=RecordingGenerator(fail_after=2),
    )

    with pytest.raises(RuntimeError, match="injected"):
        publication.publish_gan_cic_pool(
            destination,
            checkpoint_directory=checkpoint,
            dataset_id="cicmaldroid2020",
            experimental_seed=42,
            repository_root=REPOSITORY_ROOT,
        )

    assert not (destination / "manifest.json").exists()
    failure = json.loads(
        (destination / "FAILURE.json").read_text(encoding="utf-8")
    )
    assert failure["status"] == "INCOMPLETE"
    assert failure["manifest_published"] is False


def test_invalid_generator_output_rejected(tmp_path, monkeypatch):
    checkpoint = checkpoint_dir(tmp_path)
    destination = tmp_path / "pool"

    install_loader(
        monkeypatch,
        meta=metadata(),
        generator=RecordingGenerator(value=2.0),
    )

    with pytest.raises(ValueError):
        publication.publish_gan_cic_pool(
            destination,
            checkpoint_directory=checkpoint,
            dataset_id="cicmaldroid2020",
            experimental_seed=42,
            repository_root=REPOSITORY_ROOT,
        )

    assert not (destination / "manifest.json").exists()


def test_pool_view_preflight_and_batches(published):
    view = view_for(published)
    result = view.preflight()

    assert result["verified_batches"] == 5 * GAN_POOL_BATCHES_PER_CLASS
    assert result["verified_images"] == GAN_POOL_TOTAL
    assert view.size == GAN_POOL_TOTAL
    assert view.identity["executable_source_commit"] == EXECUTABLE

    positions = np.array([0, 1999, 2000, 9999], dtype=np.int64)
    observed, images, labels = view.batch(positions)

    np.testing.assert_array_equal(observed, positions)
    np.testing.assert_array_equal(
        labels.argmax(axis=1), [0, 0, 1, 4]
    )
    assert images.shape == (4, 40, 40, 3)
    np.testing.assert_array_equal(images, np.float32(0.5))


class Adapter:
    def split_size(self, split):
        return 257

    def canonical_batch(self, split, positions):
        images = np.ones((len(positions), 40, 40, 3), dtype=np.float32)
        labels = np.eye(5, dtype=np.float32)[positions % 5]
        return images, labels


def test_combined_view_accepts_gan_pool(published):
    real = RealSplitView(
        adapter=Adapter(),
        dataset_id="cicmaldroid2020",
        split="train",
        provenance_sha256="a" * 64,
    )
    combined = CombinedTrainingView(
        real=real,
        synthetic=view_for(published),
    )

    assert combined.size == 257 + GAN_POOL_TOTAL
    positions = np.array([257, 256, 2257, 0], dtype=np.int64)
    observed, images, labels = combined.batch(positions)

    np.testing.assert_array_equal(observed, positions)
    np.testing.assert_array_equal(labels.argmax(axis=1), [0, 1, 1, 0])
    np.testing.assert_array_equal(images[:, 0, 0, 0], [0.5, 1, 0.5, 1])


@pytest.mark.parametrize(
    "override",
    [
        {"manifest_sha256": "0" * 64},
        {"family": "maf"},
        {"experimental_seed": 43},
        {"recipe_sha256": "0" * 64},
        {"checkpoint_sha256": "0" * 64},
    ],
)
def test_pool_view_wrong_identity_rejected(published, override):
    values = {
        "manifest_sha256": published.manifest_sha256,
        "family": "gan",
        "experimental_seed": 42,
        "recipe_sha256": published.manifest["recipe_sha256"],
        "checkpoint_sha256": published.manifest["checkpoint_sha256"],
    }
    values.update(override)

    with pytest.raises(ValueError):
        GANPoolView(published.manifest_path, **values)


def _small_pool(tmp_path, published):
    root = tmp_path / "pool"
    root.mkdir()
    first = root / "c0-b000.npy"
    np.save(
        first,
        np.full((GAN_POOL_BATCH_SIZE, 40, 40, 3), 0.5, dtype=np.float32),
        allow_pickle=False,
    )
    digest = hashlib.sha256(first.read_bytes()).hexdigest()

    for class_id in range(5):
        for batch_index in range(GAN_POOL_BATCHES_PER_CLASS):
            name = f"c{class_id}-b{batch_index:03d}.npy"
            if (class_id, batch_index) != (0, 0):
                os.link(first, root / name)

    document = json.loads(json.dumps(published.manifest))
    for record in document["batches"]:
        record["sha256"] = digest

    raw = (
        json.dumps(document, sort_keys=True, allow_nan=False).encode()
        + b"\n"
    )
    path = root / "manifest.json"
    path.write_bytes(raw)

    return root, path, hashlib.sha256(raw).hexdigest(), document


def _view(root, path, digest, document):
    return GANPoolView(
        path,
        manifest_sha256=digest,
        family="gan",
        experimental_seed=42,
        recipe_sha256=document["recipe_sha256"],
        checkpoint_sha256=document["checkpoint_sha256"],
    )


def test_corrupt_batch_rejected(tmp_path, published):
    root, path, digest, document = _small_pool(tmp_path, published)
    target = root / "c2-b010.npy"
    target.unlink()
    np.save(target, np.zeros((16, 40, 40, 3), dtype=np.float32))

    with pytest.raises(ValueError, match="digest"):
        _view(root, path, digest, document).preflight()


def test_missing_batch_rejected(tmp_path, published):
    root, path, digest, document = _small_pool(tmp_path, published)
    (root / "c4-b124.npy").unlink()

    with pytest.raises(FileNotFoundError):
        _view(root, path, digest, document).preflight()


def test_incomplete_manifest_rejected(tmp_path, published):
    root, path, digest, document = _small_pool(tmp_path, published)
    document["batches"].pop()
    raw = (
        json.dumps(document, sort_keys=True, allow_nan=False).encode()
        + b"\n"
    )
    path.write_bytes(raw)

    with pytest.raises(ValueError, match="625"):
        GANPoolView(
            path,
            manifest_sha256=hashlib.sha256(raw).hexdigest(),
            family="gan",
            experimental_seed=42,
            recipe_sha256=document["recipe_sha256"],
            checkpoint_sha256=document["checkpoint_sha256"],
        )


def test_failure_marker_rejected(tmp_path, published):
    root, path, digest, document = _small_pool(tmp_path, published)
    (root / publication.FAILURE_FILENAME).write_text(
        "{}\n", encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="incomplete"):
        _view(root, path, digest, document).preflight()
