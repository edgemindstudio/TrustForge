"""Focused fixture tests for VAE production pool publication and verification.

Synthetic training/checkpoint fixtures only. The actual production gate remains
closed, and the real checkpoint is never loaded or synthesized.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("tensorflow")
import tensorflow as tf

from trustforge.evaluation.paper01_vae_pool import VAEPoolView
from trustforge.models.paper01_revision_2026 import (
    variational_autoencoder_checkpoint_publication_coordinator as coordinator,
    variational_autoencoder_training_runner as runner,
    vae_cic_pool_publication as publication,
)
from trustforge.models.paper01_revision_2026.synthesis_linkage import (
    derive_latent_vectors,
    load_synthesis_recipe,
)
from tests.test_variational_autoencoder_checkpoint_provenance_revision_2026 import (
    evidence_for,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = REPO_ROOT / "scripts" / "publish_paper01_vae_pool.py"
CHECKPOINT_SOURCE = "a" * 40
SYNTH_SOURCE = "b" * 40
SEED = 42
CLASS_MAPPING = ("0", "1", "2", "3", "4")
CLASS_COUNT = 5
TRAIN_SIZE = 128
VALIDATION_SIZE = 16

EFFECTIVE_CONFIGURATION = {
    "architecture": "conditional_convolutional_vae",
    "canonical_image_shape": [40, 40, 3],
    "canonical_input_domain": [0.0, 1.0],
    "internal_model_domain": [-1.0, 1.0],
    "latent_dimension": 100,
    "observation_variance": 1.0,
    "kl_coefficient": 1.0,
    "kl_annealing": False,
    "optimizer": "adam",
    "learning_rate": 0.001,
    "batch_size": 64,
    "epochs": 2,
    "validation": "full_governed_validation_after_each_epoch",
    "selection_metric": "validation_total_loss",
    "selection_direction": "minimize",
    "selection_tie_break": "earliest_epoch",
    "checkpoint_roles": ["best", "last"],
    "test_split_usage": "prohibited",
}


class SyntheticAdapter:
    def __init__(self):
        rng = np.random.default_rng(20261121)
        self.images = {
            split: rng.uniform(
                0.1,
                0.9,
                size=(count, 40, 40, 3),
            ).astype(np.float32)
            for split, count in (
                ("train", TRAIN_SIZE),
                ("validation", VALIDATION_SIZE),
            )
        }
        self.labels = {
            split: np.eye(
                CLASS_COUNT,
                dtype=np.float32,
            )[np.arange(count) % CLASS_COUNT]
            for split, count in (
                ("train", TRAIN_SIZE),
                ("validation", VALIDATION_SIZE),
            )
        }

    def split_size(self, split):
        return len(self.images[split])

    def canonical_batch(self, split, logical_positions):
        positions = np.asarray(logical_positions, dtype=np.int64)
        return (
            self.images[split][positions].copy(),
            self.labels[split][positions].copy(),
        )


@pytest.fixture(scope="module")
def published_checkpoint(tmp_path_factory):
    tf.keras.backend.clear_session()
    root = tmp_path_factory.mktemp("vae-checkpoint")
    trained = runner.run_synthetic_vae_training(
        adapter=SyntheticAdapter(),
        num_classes=CLASS_COUNT,
        experimental_seed=SEED,
        epochs=2,
        batch_size=64,
        learning_rate=0.001,
        development_mode="synthetic_only",
    )
    governed = {
        "dataset_id": "cicmaldroid2020",
        "experimental_seed": SEED,
        "class_count": CLASS_COUNT,
        "dataset_provenance": evidence_for("cicmaldroid2020"),
        "training_result": trained,
        "checkpoint_status": "NOT_PREPARED",
        "publication_status": "NOT_PUBLISHED",
    }
    result = coordinator.publish_governed_vae_checkpoints(
        governed_result=governed,
        class_mapping=CLASS_MAPPING,
        effective_configuration=EFFECTIVE_CONFIGURATION,
        publication_root=root / "checkpoint-publication",
        source_commit=CHECKPOINT_SOURCE,
        attempt=2,
        repository_root=REPO_ROOT,
    )
    best = result["roles"]["best"]
    last = result["roles"]["last"]
    return SimpleNamespace(
        root=root,
        publication_root=Path(result["publication_root"]),
        publication=result,
        best=best,
        last=last,
        best_generation=best["generation_id"],
        best_manifest=best["manifest_sha256"],
        best_directory=Path(best["generation_directory"]),
        last_generation=last["generation_id"],
        last_manifest=last["manifest_sha256"],
    )


class RecordingVAEDecoder:
    def __init__(self, *, value=0.0, fail_after=None):
        self.value = value
        self.fail_after = fail_after
        self.calls = []

    def __call__(self, inputs, training=False):
        if (
            self.fail_after is not None
            and len(self.calls) >= self.fail_after
        ):
            raise RuntimeError("injected decoder failure")
        latents, labels = inputs
        assert training is False
        assert latents.shape[1] == 100
        assert labels.shape[1] == CLASS_COUNT
        self.calls.append(
            {
                "latents": np.asarray(latents).copy(),
                "labels": np.asarray(labels).copy(),
            }
        )
        return np.full(
            (len(latents), 40, 40, 3),
            self.value,
            dtype=np.float32,
        )


def _install_runtime(
    monkeypatch,
    *,
    decoder=None,
    commit=SYNTH_SOURCE,
    dirty=False,
):
    decoder = decoder or RecordingVAEDecoder()
    loader_calls = []

    monkeypatch.setattr(
        publication,
        "capture_git_state",
        lambda root: SimpleNamespace(commit=commit, dirty=dirty),
    )

    def loader(**kwargs):
        loader_calls.append(kwargs)
        return decoder

    monkeypatch.setattr(
        publication,
        "load_vae_decoder_for_synthesis",
        loader,
    )
    return decoder, loader_calls


@pytest.fixture(scope="module")
def published_pool(published_checkpoint, tmp_path_factory):
    monkeypatch = pytest.MonkeyPatch()
    decoder, loader_calls = _install_runtime(monkeypatch)
    destination = tmp_path_factory.mktemp("vae-pool") / "pool"
    try:
        manifest_path = publication.publish_vae_pool(
            destination,
            publication_root=published_checkpoint.publication_root,
            generation_id=published_checkpoint.best_generation,
            expected_manifest_sha256=published_checkpoint.best_manifest,
            dataset_id="cicmaldroid2020",
            experimental_seed=SEED,
            class_mapping=CLASS_MAPPING,
            repository_root=REPO_ROOT,
            expected_synthesis_source_commit=SYNTH_SOURCE,
        )
    finally:
        monkeypatch.undo()

    raw = manifest_path.read_bytes()
    return SimpleNamespace(
        manifest_path=manifest_path,
        manifest=json.loads(raw),
        manifest_sha256=hashlib.sha256(raw).hexdigest(),
        root=destination,
        decoder=decoder,
        loader_calls=loader_calls,
        recipe=load_synthesis_recipe(REPO_ROOT),
    )


def _view(published, **overrides):
    values = {
        "manifest_sha256": published.manifest_sha256,
        "family": "vae",
        "experimental_seed": SEED,
        "recipe_sha256": published.manifest["recipe_sha256"],
        "checkpoint_sha256": published.manifest[
            "checkpoint_manifest_sha256"
        ],
    }
    values.update(overrides)
    return VAEPoolView(published.manifest_path, **values)


def _write_manifest(root, document):
    raw = (
        json.dumps(document, sort_keys=True, allow_nan=False).encode()
        + b"\n"
    )
    path = root / "manifest.json"
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest()


def _small_pool(tmp_path, published):
    root = tmp_path / "pool"
    root.mkdir()
    base32 = root / "c0-b000.npy"
    base16 = root / "c0-b062.npy"
    np.save(
        base32,
        np.full(
            (32, 40, 40, 3),
            0.5,
            dtype=np.float32,
        ),
        allow_pickle=False,
    )
    np.save(
        base16,
        np.full(
            (16, 40, 40, 3),
            0.5,
            dtype=np.float32,
        ),
        allow_pickle=False,
    )
    hash32 = hashlib.sha256(base32.read_bytes()).hexdigest()
    hash16 = hashlib.sha256(base16.read_bytes()).hexdigest()
    size32 = base32.stat().st_size
    size16 = base16.stat().st_size

    for class_id in range(CLASS_COUNT):
        for batch_index in range(publication.VAE_POOL_BATCHES_PER_CLASS):
            filename = f"c{class_id}-b{batch_index:03d}.npy"
            target = root / filename
            source = base16 if batch_index == 62 else base32
            if target != source:
                os.link(source, target)

    document = json.loads(json.dumps(published.manifest))
    for record in document["batches"]:
        if record["batch_index"] == 62:
            record["sha256"] = hash16
            record["size_bytes"] = size16
        else:
            record["sha256"] = hash32
            record["size_bytes"] = size32
    path, digest = _write_manifest(root, document)
    return root, path, digest, document


def _view_from_document(path, digest, document):
    return VAEPoolView(
        path,
        manifest_sha256=digest,
        family="vae",
        experimental_seed=SEED,
        recipe_sha256=document["recipe_sha256"],
        checkpoint_sha256=document["checkpoint_manifest_sha256"],
    )


def test_complete_pool_manifest_and_batches(
    published_pool,
    published_checkpoint,
):
    manifest = published_pool.manifest

    assert manifest["schema"] == publication.VAE_POOL_SCHEMA
    assert manifest["dataset_id"] == "cicmaldroid2020"
    assert manifest["family"] == "vae"
    assert manifest["experimental_seed"] == SEED
    assert manifest["class_mapping"] == list(CLASS_MAPPING)
    assert manifest["class_count"] == CLASS_COUNT
    assert manifest["checkpoint_generation_id"] == (
        published_checkpoint.best_generation
    )
    assert manifest["checkpoint_manifest_sha256"] == (
        published_checkpoint.best_manifest
    )
    assert manifest["checkpoint_role"] == "best"
    assert manifest["checkpoint_source_commit"] == CHECKPOINT_SOURCE
    assert manifest["synthesis_source_commit"] == SYNTH_SOURCE
    assert manifest["model_call_batch_size"] == 32
    assert manifest["quota_per_class"] == 2000
    assert manifest["batches_per_class"] == 63
    assert manifest["complete_pool"] is True
    assert manifest["expected_total"] == 10000
    assert manifest["observed_total"] == 10000
    assert manifest["status"] == "COMPLETE"
    assert manifest["authoritative_artifact"] is True
    assert len(manifest["batches"]) == 315

    first = manifest["batches"][0]
    last = manifest["batches"][-1]
    assert first["sample_indices"] == list(range(32))
    assert first["shape"] == [32, 40, 40, 3]
    assert last["sample_indices"] == list(range(1984, 2000))
    assert last["shape"] == [16, 40, 40, 3]

    first_images = np.load(
        published_pool.root / first["filename"],
        allow_pickle=False,
    )
    last_images = np.load(
        published_pool.root / last["filename"],
        allow_pickle=False,
    )
    assert first_images.shape == (32, 40, 40, 3)
    assert last_images.shape == (16, 40, 40, 3)
    assert first_images.dtype == np.float32
    assert last_images.dtype == np.float32
    np.testing.assert_array_equal(first_images, np.float32(0.5))
    np.testing.assert_array_equal(last_images, np.float32(0.5))

    assert len(published_pool.decoder.calls) == 315
    first_call = published_pool.decoder.calls[0]
    np.testing.assert_array_equal(
        first_call["latents"],
        derive_latent_vectors(
            dataset_id="cicmaldroid2020",
            family="vae",
            experimental_seed=SEED,
            class_id=0,
            sample_indices=tuple(range(32)),
        ),
    )
    last_call = published_pool.decoder.calls[-1]
    np.testing.assert_array_equal(
        last_call["latents"],
        derive_latent_vectors(
            dataset_id="cicmaldroid2020",
            family="vae",
            experimental_seed=SEED,
            class_id=4,
            sample_indices=tuple(range(1984, 2000)),
        ),
    )
    assert len(published_pool.loader_calls) == 1
    loader = published_pool.loader_calls[0]
    assert Path(loader["checkpoint_directory"]) == (
        published_checkpoint.best_directory
    )
    assert loader["request"].family == "vae"
    assert loader["request"].class_mapping == CLASS_MAPPING
    assert loader["checkpoint_identity"]["checkpoint_role"] == "best"
    assert loader["checkpoint_identity"]["checkpoint_sha256"] == (
        published_checkpoint.best_manifest
    )


def test_activated_source_gate_still_requires_checkpoint_verification(
    tmp_path,
    monkeypatch,
):
    calls = []

    def reject_checkpoint(*args, **kwargs):
        calls.append("verify")
        raise ValueError("checkpoint identity required")

    monkeypatch.setattr(
        publication,
        "verify_committed_vae_generation",
        reject_checkpoint,
    )
    monkeypatch.setattr(
        publication,
        "load_vae_decoder_for_synthesis",
        lambda **kwargs: calls.append("loader"),
    )

    destination = tmp_path / "pool"
    assert publication.VAE_PRODUCTION_SYNTHESIS_AUTHORIZED is True
    assert (
        publication.require_vae_production_synthesis_authorization()
        is None
    )

    with pytest.raises(
        ValueError,
        match="checkpoint identity required",
    ):
        publication.publish_vae_pool(
            destination,
            publication_root=tmp_path / "checkpoint",
            generation_id="0" * 32,
            expected_manifest_sha256="0" * 64,
            dataset_id="cicmaldroid2020",
            experimental_seed=SEED,
            class_mapping=CLASS_MAPPING,
            repository_root=REPO_ROOT,
            expected_synthesis_source_commit=SYNTH_SOURCE,
        )

    assert calls == ["verify"]
    assert not destination.exists()


def test_wrong_manifest_rejected_before_generation(
    published_checkpoint,
    tmp_path,
    monkeypatch,
):
    decoder, loader_calls = _install_runtime(monkeypatch)
    destination = tmp_path / "pool"

    with pytest.raises(ValueError, match="manifest"):
        publication.publish_vae_pool(
            destination,
            publication_root=published_checkpoint.publication_root,
            generation_id=published_checkpoint.best_generation,
            expected_manifest_sha256="0" * 64,
            dataset_id="cicmaldroid2020",
            experimental_seed=SEED,
            class_mapping=CLASS_MAPPING,
            repository_root=REPO_ROOT,
            expected_synthesis_source_commit=SYNTH_SOURCE,
        )

    assert loader_calls == []
    assert not destination.exists()


def test_last_checkpoint_role_rejected(
    published_checkpoint,
    tmp_path,
    monkeypatch,
):
    decoder, loader_calls = _install_runtime(monkeypatch)
    destination = tmp_path / "pool"

    with pytest.raises(ValueError, match="best"):
        publication.publish_vae_pool(
            destination,
            publication_root=published_checkpoint.publication_root,
            generation_id=published_checkpoint.last_generation,
            expected_manifest_sha256=published_checkpoint.last_manifest,
            dataset_id="cicmaldroid2020",
            experimental_seed=SEED,
            class_mapping=CLASS_MAPPING,
            repository_root=REPO_ROOT,
            expected_synthesis_source_commit=SYNTH_SOURCE,
        )

    assert loader_calls == []
    assert not destination.exists()


@pytest.mark.parametrize(
    "overrides",
    [
        {"dataset_id": "ustc_tfc2016"},
        {"experimental_seed": 43},
        {"class_mapping": CLASS_MAPPING[:-1]},
    ],
)
def test_wrong_identity_rejected(
    published_checkpoint,
    tmp_path,
    monkeypatch,
    overrides,
):
    _install_runtime(monkeypatch)
    values = {
        "publication_root": published_checkpoint.publication_root,
        "generation_id": published_checkpoint.best_generation,
        "expected_manifest_sha256": published_checkpoint.best_manifest,
        "dataset_id": "cicmaldroid2020",
        "experimental_seed": SEED,
        "class_mapping": CLASS_MAPPING,
        "repository_root": REPO_ROOT,
        "expected_synthesis_source_commit": SYNTH_SOURCE,
    }
    values.update(overrides)
    destination = tmp_path / "pool"

    with pytest.raises((ValueError, TypeError)):
        publication.publish_vae_pool(destination, **values)

    assert not destination.exists()


def test_dirty_or_mismatched_synthesis_source_rejected(
    published_checkpoint,
    tmp_path,
    monkeypatch,
):
    _install_runtime(monkeypatch, dirty=True)
    destination = tmp_path / "pool"

    with pytest.raises(RuntimeError, match="dirty"):
        publication.publish_vae_pool(
            destination,
            publication_root=published_checkpoint.publication_root,
            generation_id=published_checkpoint.best_generation,
            expected_manifest_sha256=published_checkpoint.best_manifest,
            dataset_id="cicmaldroid2020",
            experimental_seed=SEED,
            class_mapping=CLASS_MAPPING,
            repository_root=REPO_ROOT,
            expected_synthesis_source_commit=SYNTH_SOURCE,
        )

    assert not destination.exists()

    _install_runtime(
        monkeypatch,
        commit="c" * 40,
    )
    with pytest.raises(PermissionError, match="source"):
        publication.publish_vae_pool(
            destination,
            publication_root=published_checkpoint.publication_root,
            generation_id=published_checkpoint.best_generation,
            expected_manifest_sha256=published_checkpoint.best_manifest,
            dataset_id="cicmaldroid2020",
            experimental_seed=SEED,
            class_mapping=CLASS_MAPPING,
            repository_root=REPO_ROOT,
            expected_synthesis_source_commit=SYNTH_SOURCE,
        )

    assert not destination.exists()


def test_incomplete_publication_has_no_manifest(
    published_checkpoint,
    tmp_path,
    monkeypatch,
):
    decoder = RecordingVAEDecoder(fail_after=5)
    _install_runtime(monkeypatch, decoder=decoder)
    destination = tmp_path / "pool"

    with pytest.raises(RuntimeError, match="injected"):
        publication.publish_vae_pool(
            destination,
            publication_root=published_checkpoint.publication_root,
            generation_id=published_checkpoint.best_generation,
            expected_manifest_sha256=published_checkpoint.best_manifest,
            dataset_id="cicmaldroid2020",
            experimental_seed=SEED,
            class_mapping=CLASS_MAPPING,
            repository_root=REPO_ROOT,
            expected_synthesis_source_commit=SYNTH_SOURCE,
        )

    assert not (destination / publication.MANIFEST_FILENAME).exists()
    failure = json.loads(
        (destination / publication.FAILURE_FILENAME).read_text()
    )
    assert failure["status"] == "INCOMPLETE"
    assert failure["manifest_published"] is False


def test_existing_destination_rejected(
    published_checkpoint,
    tmp_path,
    monkeypatch,
):
    _install_runtime(monkeypatch)
    destination = tmp_path / "pool"
    destination.mkdir()

    with pytest.raises(FileExistsError):
        publication.publish_vae_pool(
            destination,
            publication_root=published_checkpoint.publication_root,
            generation_id=published_checkpoint.best_generation,
            expected_manifest_sha256=published_checkpoint.best_manifest,
            dataset_id="cicmaldroid2020",
            experimental_seed=SEED,
            class_mapping=CLASS_MAPPING,
            repository_root=REPO_ROOT,
            expected_synthesis_source_commit=SYNTH_SOURCE,
        )

    assert list(destination.iterdir()) == []


def test_pool_view_preflight_and_batches(published_pool):
    view = _view(published_pool)
    preflight = view.preflight()

    assert preflight["verified_batches"] == 315
    assert preflight["verified_images"] == 10000
    assert view.size == 10000
    assert view.identity["checkpoint_source_commit"] == CHECKPOINT_SOURCE
    assert view.identity["synthesis_source_commit"] == SYNTH_SOURCE

    positions = np.array([0, 1999, 2000, 9999], dtype=np.int64)
    observed, images, labels = view.batch(positions)
    np.testing.assert_array_equal(observed, positions)
    np.testing.assert_array_equal(labels.argmax(axis=1), [0, 0, 1, 4])
    assert images.shape == (4, 40, 40, 3)
    assert images.dtype == np.float32
    np.testing.assert_array_equal(images, np.float32(0.5))


@pytest.mark.parametrize(
    "overrides",
    [
        {"manifest_sha256": "0" * 64},
        {"family": "gan"},
        {"experimental_seed": 43},
        {"recipe_sha256": "0" * 64},
        {"checkpoint_sha256": "0" * 64},
    ],
)
def test_pool_view_wrong_identity_rejected(published_pool, overrides):
    with pytest.raises(ValueError):
        _view(published_pool, **overrides)


def test_corrupt_batch_rejected(tmp_path, published_pool):
    root, path, digest, document = _small_pool(tmp_path, published_pool)
    target = root / "c2-b010.npy"
    target.unlink()
    np.save(
        target,
        np.zeros((32, 40, 40, 3), dtype=np.float32),
        allow_pickle=False,
    )

    with pytest.raises(ValueError, match="digest"):
        _view_from_document(path, digest, document).preflight()


def test_missing_batch_rejected(tmp_path, published_pool):
    root, path, digest, document = _small_pool(tmp_path, published_pool)
    (root / "c4-b062.npy").unlink()

    with pytest.raises(FileNotFoundError):
        _view_from_document(path, digest, document).preflight()


def test_wrong_shape_rejected(tmp_path, published_pool):
    root, path, digest, document = _small_pool(tmp_path, published_pool)
    target = root / "c1-b001.npy"
    target.unlink()
    np.save(
        target,
        np.zeros((31, 40, 40, 3), dtype=np.float32),
        allow_pickle=False,
    )
    record = document["batches"][63 + 1]
    record["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    record["size_bytes"] = target.stat().st_size
    path, digest = _write_manifest(root, document)

    with pytest.raises(ValueError, match="metadata"):
        _view_from_document(path, digest, document).preflight()


def test_wrong_dtype_rejected(tmp_path, published_pool):
    root, path, digest, document = _small_pool(tmp_path, published_pool)
    target = root / "c1-b002.npy"
    target.unlink()
    np.save(
        target,
        np.zeros((32, 40, 40, 3), dtype=np.float64),
        allow_pickle=False,
    )
    record = document["batches"][63 + 2]
    record["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    record["size_bytes"] = target.stat().st_size
    path, digest = _write_manifest(root, document)

    with pytest.raises(ValueError):
        _view_from_document(path, digest, document).preflight()


def test_nonfinite_or_out_of_domain_rejected(tmp_path, published_pool):
    root, path, digest, document = _small_pool(tmp_path, published_pool)
    target = root / "c1-b003.npy"
    target.unlink()
    values = np.full((32, 40, 40, 3), 2.0, dtype=np.float32)
    np.save(target, values, allow_pickle=False)
    record = document["batches"][63 + 3]
    record["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    record["size_bytes"] = target.stat().st_size
    path, digest = _write_manifest(root, document)

    with pytest.raises(ValueError):
        _view_from_document(path, digest, document).preflight()


def test_duplicate_sample_identities_rejected(
    tmp_path,
    published_pool,
):
    root, path, digest, document = _small_pool(tmp_path, published_pool)
    document["batches"][0]["sample_indices"] = [0] * 32
    path, digest = _write_manifest(root, document)

    with pytest.raises(ValueError, match="sample identity"):
        _view_from_document(path, digest, document)


def test_overlapping_sample_identities_rejected(
    tmp_path,
    published_pool,
):
    root, path, digest, document = _small_pool(tmp_path, published_pool)
    document["batches"][1]["sample_indices"] = list(range(1, 33))
    path, digest = _write_manifest(root, document)

    with pytest.raises(ValueError, match="sample identity"):
        _view_from_document(path, digest, document)


def test_incorrect_class_allocation_rejected(
    tmp_path,
    published_pool,
):
    root, path, digest, document = _small_pool(tmp_path, published_pool)
    document["batches"][0]["class_id"] = 5
    path, digest = _write_manifest(root, document)

    with pytest.raises(ValueError, match="class"):
        _view_from_document(path, digest, document)


def test_incomplete_pool_rejected(tmp_path, published_pool):
    root, path, digest, document = _small_pool(tmp_path, published_pool)
    document["batches"].pop()
    path, digest = _write_manifest(root, document)

    with pytest.raises(ValueError, match="315"):
        _view_from_document(path, digest, document)


def test_inconsistent_completion_metadata_rejected(
    tmp_path,
    published_pool,
):
    root, path, digest, document = _small_pool(tmp_path, published_pool)
    document["complete_pool"] = False
    path, digest = _write_manifest(root, document)

    with pytest.raises(ValueError):
        _view_from_document(path, digest, document)


def test_recipe_authorization_remains_disabled_after_gate_activation(
    published_pool,
):
    recipe = published_pool.recipe.data

    assert recipe["families"]["vae"]["production_execution_authorized"] is False
    assert all(
        value is False
        for value in recipe["authorization"].values()
    )
    assert publication.VAE_PRODUCTION_SYNTHESIS_AUTHORIZED is True


def test_cli_gate_is_active_and_publisher_receives_explicit_identity(
    monkeypatch,
    tmp_path,
):
    spec = importlib.util.spec_from_file_location(
        "paper01_vae_pool_cli",
        CLI_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    calls = []

    def stop_at_publisher(*args, **kwargs):
        calls.append(kwargs)
        raise RuntimeError("publisher reached")

    monkeypatch.setattr(
        module,
        "publish_vae_pool",
        stop_at_publisher,
    )
    args = module.make_parser().parse_args(
        [
            "--repository-root",
            str(REPO_ROOT),
            "--expected-synthesis-source-commit",
            SYNTH_SOURCE,
            "--checkpoint-publication-root",
            str(tmp_path / "checkpoint"),
            "--generation-id",
            "0" * 32,
            "--expected-manifest-sha256",
            "0" * 64,
            "--dataset-id",
            "cicmaldroid2020",
            "--experimental-seed",
            "42",
            "--class-mapping",
            "0,1,2,3,4",
            "--destination",
            str(tmp_path / "pool"),
        ]
    )

    with pytest.raises(RuntimeError, match="publisher reached"):
        module.run(args)

    assert len(calls) == 1
    assert calls[0]["generation_id"] == "0" * 32
    assert calls[0]["expected_manifest_sha256"] == "0" * 64
    assert calls[0]["dataset_id"] == "cicmaldroid2020"
    assert calls[0]["experimental_seed"] == 42
    assert calls[0]["class_mapping"] == CLASS_MAPPING
    assert not (tmp_path / "pool").exists()
