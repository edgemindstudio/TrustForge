"""Focused VAE pool admission and VAE driver authentication checks.

Fixture-only. A complete synthetic VAE pool is materialised in a temporary
directory. No real data, checkpoint, classifier training or evaluation occurs.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from trustforge.evaluation.paper01_combined_inputs import CombinedTrainingView
from trustforge.evaluation.paper01_inputs import RealSplitView
from trustforge.evaluation.paper01_vae_pool import VAEPoolView
from trustforge.models.paper01_revision_2026.synthesis_linkage import (
    CANONICAL_OUTPUT,
    LATENT_STREAM_CONTRACT,
)
from trustforge.models.paper01_revision_2026.vae_cic_pool_publication import (
    VAE_POOL_SCHEMA,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DRIVER_PATH = (
    REPO_ROOT
    / "scripts"
    / "run_paper01_cic_vae_seed42_bounded_evaluation.py"
)
DATASET_ID = "cicmaldroid2020"
FAMILY = "vae"
SEED = 42
CLASS_MAPPING = ["0", "1", "2", "3", "4"]
CLASS_COUNT = 5
SAMPLES_PER_CLASS = 2000
BATCHES_PER_CLASS = 63
MODEL_CALL_BATCH_SIZE = 32
POOL_SAMPLES = 10_000
POOL_BATCHES = 315
GENERATION_ID = "35ca3cd756a88d6b93dcc291f02c4747"
CHECKPOINT_SHA = (
    "f1fa66c2e689b7cfa66890b4b5273f9a6591da121ad22ff5dba9afab53bf812e"
)
CHECKPOINT_SOURCE = "1aec01c26fab51b87362a47e2cfa797bd428c359"
SYNTHESIS_SOURCE = "d4031a9b39565e5382d08e2b28236162498a08ca"
CHECKPOINT_RUN_IDENTITY = "commit1aec01c_attempt02"
RECIPE_SHA = (
    "98f9ef427e1b2a793636d448fe3f272893a24848e4b7257764964015e19987a9"
)
SPLIT_COUNTS = {"train": 8117, "validation": 1158, "test": 2323}
SPLIT_CLASS_COUNTS = {
    "train": (877, 1470, 2732, 1782, 1256),
    "validation": (125, 210, 390, 254, 179),
    "test": (251, 420, 782, 510, 360),
}


class Adapter:
    def __init__(self, *, size=257):
        self._size = size
        self.requests = []

    def split_size(self, split):
        return self._size

    def canonical_batch(self, split, positions):
        self.requests.append((split, positions.copy()))
        images = np.ones((len(positions), 40, 40, 3), dtype=np.float32)
        labels = np.eye(CLASS_COUNT, dtype=np.float32)[positions % CLASS_COUNT]
        return images, labels


def real_view(adapter, split="train"):
    return RealSplitView(
        adapter=adapter,
        dataset_id=DATASET_ID,
        split=split,
        provenance_sha256="a" * 64,
    )


def _sample_indices(batch_index):
    start = batch_index * MODEL_CALL_BATCH_SIZE
    stop = min(start + MODEL_CALL_BATCH_SIZE, SAMPLES_PER_CLASS)
    return list(range(start, stop))


def build_vae_pool(root):
    """Write one complete fixture VAE pool and return its view and manifest."""
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261123)
    records = []
    for class_id in range(CLASS_COUNT):
        for batch_index in range(BATCHES_PER_CLASS):
            indices = _sample_indices(batch_index)
            images = rng.random(
                (len(indices), 40, 40, 3), dtype=np.float32,
            )
            buffer = io.BytesIO()
            np.save(buffer, images, allow_pickle=False)
            raw = buffer.getvalue()
            filename = f"c{class_id}-b{batch_index:03d}.npy"
            (root / filename).write_bytes(raw)
            records.append({
                "class_id": class_id,
                "batch_index": batch_index,
                "sample_indices": indices,
                "filename": filename,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "size_bytes": len(raw),
                "shape": [len(indices), 40, 40, 3],
                "dtype": "float32",
            })

    manifest = {
        "schema": VAE_POOL_SCHEMA,
        "dataset_id": DATASET_ID,
        "family": FAMILY,
        "experimental_seed": SEED,
        "class_mapping": list(CLASS_MAPPING),
        "class_count": CLASS_COUNT,
        "checkpoint_generation_id": GENERATION_ID,
        "checkpoint_manifest_sha256": CHECKPOINT_SHA,
        "checkpoint_role": "best",
        "checkpoint_run_identity": CHECKPOINT_RUN_IDENTITY,
        "checkpoint_source_commit": CHECKPOINT_SOURCE,
        "synthesis_source_commit": SYNTHESIS_SOURCE,
        "recipe_sha256": RECIPE_SHA,
        "latent_stream_contract": LATENT_STREAM_CONTRACT,
        "rng": {
            "contract": LATENT_STREAM_CONTRACT,
            "bit_generator": "PCG64",
            "latent_dimension": 100,
        },
        "model_call_batch_size": MODEL_CALL_BATCH_SIZE,
        "quota_per_class": SAMPLES_PER_CLASS,
        "batches_per_class": BATCHES_PER_CLASS,
        "complete_pool": True,
        "expected_total": POOL_SAMPLES,
        "observed_total": POOL_SAMPLES,
        "status": "COMPLETE",
        "authoritative_artifact": True,
        "canonical_output": CANONICAL_OUTPUT,
        "batches": records,
    }
    raw = (
        json.dumps(manifest, sort_keys=True, allow_nan=False).encode()
        + b"\n"
    )
    manifest_path = root / "manifest.json"
    manifest_path.write_bytes(raw)
    return manifest_path, manifest, hashlib.sha256(raw).hexdigest()


def open_view(manifest_path, manifest_sha256):
    return VAEPoolView(
        manifest_path,
        manifest_sha256=manifest_sha256,
        family=FAMILY,
        experimental_seed=SEED,
        recipe_sha256=RECIPE_SHA,
        checkpoint_sha256=CHECKPOINT_SHA,
    )


@pytest.fixture(scope="module")
def vae_pool(tmp_path_factory):
    root = tmp_path_factory.mktemp("vae-admission-pool")
    manifest_path, manifest, manifest_sha256 = build_vae_pool(root)
    return {
        "manifest_path": manifest_path,
        "manifest": manifest,
        "manifest_sha256": manifest_sha256,
        "view": open_view(manifest_path, manifest_sha256),
    }


class _StubVAEPool(VAEPoolView):
    """VAE-typed view used to exercise admission guards without new files."""

    def __init__(self, identity, *, size=POOL_SAMPLES):
        self._stub_identity = dict(identity)
        self._stub_size = size

    @property
    def size(self):
        return self._stub_size

    @property
    def identity(self):
        return dict(self._stub_identity)

    def preflight(self):
        return {
            "identity": self.identity,
            "verified_batches": POOL_BATCHES,
            "verified_images": POOL_SAMPLES,
        }

    def batch(self, positions):
        requested = positions.copy()
        classes = requested // SAMPLES_PER_CLASS
        return (
            requested,
            np.ones((len(requested), 40, 40, 3), dtype=np.float32),
            np.eye(CLASS_COUNT, dtype=np.float32)[classes],
        )


def test_vae_pool_is_admitted_and_identity_retained(vae_pool):
    adapter = Adapter()
    combined = CombinedTrainingView(
        real=real_view(adapter), synthetic=vae_pool["view"],
    )

    assert combined.size == 257 + POOL_SAMPLES
    identity = combined.identity
    assert identity["condition"] == "real_plus_synthetic"
    assert identity["position_order"] == "real_then_synthetic"
    assert identity["synthetic"]["manifest_sha256"] == (
        vae_pool["manifest_sha256"]
    )
    assert identity["synthetic"]["checkpoint_generation_id"] == GENERATION_ID
    assert identity["synthetic"]["checkpoint_manifest_sha256"] == CHECKPOINT_SHA
    assert identity["synthetic"]["checkpoint_role"] == "best"
    assert identity["synthetic"]["synthesis_source_commit"] == SYNTHESIS_SOURCE
    # Admission preflight must complete before any classifier batch request.
    assert adapter.requests == []


def test_vae_pool_wrong_dataset_identity_rejected(vae_pool):
    identity = dict(vae_pool["view"].identity)
    identity["dataset_id"] = "malex"
    with pytest.raises(ValueError, match="dataset/class contracts differ"):
        CombinedTrainingView(
            real=real_view(Adapter()),
            synthetic=_StubVAEPool(identity),
        )


def test_vae_pool_wrong_real_split_rejected(vae_pool):
    with pytest.raises(ValueError, match="training split"):
        CombinedTrainingView(
            real=real_view(Adapter(), "validation"),
            synthetic=vae_pool["view"],
        )


def test_unsupported_pool_type_rejected(vae_pool):
    with pytest.raises(TypeError, match="governed synthetic pool view"):
        CombinedTrainingView(real=real_view(Adapter()), synthetic=object())


def test_incomplete_vae_pool_rejected(vae_pool, tmp_path):
    root = tmp_path / "incomplete"
    root.mkdir()
    manifest_path, manifest, manifest_sha256 = build_vae_pool(root)
    (root / manifest["batches"][0]["filename"]).unlink()
    view = open_view(manifest_path, manifest_sha256)
    with pytest.raises(FileNotFoundError):
        CombinedTrainingView(real=real_view(Adapter()), synthetic=view)


def test_real_then_synthetic_positions_and_cross_boundary(vae_pool):
    adapter = Adapter()
    combined = CombinedTrainingView(
        real=real_view(adapter), synthetic=vae_pool["view"],
    )
    positions = np.array([256, 257, 258, 2256, 10256], dtype=np.int64)

    observed, images, labels = combined.batch(positions)

    np.testing.assert_array_equal(observed, positions)
    assert images.shape == (5, 40, 40, 3)
    assert images.dtype == np.float32
    assert np.isfinite(images).all()
    # 256 -> real class 1; 257/258 -> synthetic locals 0/1 (class 0);
    # 2256 -> synthetic local 1999 (class 0); 10256 -> local 9999 (class 4).
    np.testing.assert_array_equal(
        labels.argmax(axis=1), [1, 0, 0, 0, 4],
    )
    assert combined.sample_identity(256)["source"] == "real"
    synthetic_identity = combined.sample_identity(257)
    assert synthetic_identity["source"] == "synthetic"
    assert synthetic_identity["manifest_sha256"] == (
        vae_pool["manifest_sha256"]
    )
    assert synthetic_identity["logical_position"] == 0
    np.testing.assert_array_equal(adapter.requests[0][1], [256])


def _load_driver():
    spec = importlib.util.spec_from_file_location(
        "paper01_vae_evaluation_driver", DRIVER_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_driver_binding_pins_accepted_identities():
    driver = _load_driver()
    binding = driver.evaluation_binding(SEED)
    assert binding["family"] == FAMILY
    assert binding["pool_manifest_sha256"] == (
        "08777645ccda4cb9659f8ebd9109b6b5b50d36e1189986df289537dda43fbcc6"
    )
    assert binding["checkpoint_manifest_sha256"] == CHECKPOINT_SHA
    assert binding["checkpoint_generation_id"] == GENERATION_ID
    assert binding["checkpoint_source_commit"] == CHECKPOINT_SOURCE
    assert binding["synthesis_source_commit"] == SYNTHESIS_SOURCE
    assert binding["checkpoint_role"] == "best"
    assert binding["model_call_batch_size"] == MODEL_CALL_BATCH_SIZE
    assert binding["batches_per_class"] == BATCHES_PER_CLASS
    with pytest.raises(ValueError):
        driver.evaluation_binding(43)


def test_driver_manifest_authentication_accepts_fixture(
    vae_pool, monkeypatch,
):
    driver = _load_driver()
    monkeypatch.setattr(
        driver, "POOL_MANIFEST_SHA256", vae_pool["manifest_sha256"],
    )
    raw = vae_pool["manifest_path"].read_bytes()
    manifest = driver._authenticate_vae_manifest(
        raw, expected_sha=vae_pool["manifest_sha256"],
    )
    assert manifest["family"] == FAMILY
    assert len(manifest["batches"]) == POOL_BATCHES


def test_driver_manifest_authentication_rejects_wrong_digest(vae_pool):
    driver = _load_driver()
    raw = vae_pool["manifest_path"].read_bytes()
    with pytest.raises(ValueError, match="pool manifest digest mismatch"):
        driver._authenticate_vae_manifest(raw, expected_sha="b" * 64)


def test_driver_manifest_authentication_rejects_role_and_coverage(
    vae_pool, monkeypatch,
):
    driver = _load_driver()
    monkeypatch.setattr(
        driver, "POOL_MANIFEST_SHA256", vae_pool["manifest_sha256"],
    )
    for mutate in (
        lambda value: value.__setitem__("checkpoint_role", "last"),
        lambda value: value.__setitem__("family", "gan"),
        lambda value: value["batches"].pop(),
        lambda value: value.__setitem__("complete_pool", False),
    ):
        document = json.loads(
            vae_pool["manifest_path"].read_text(encoding="utf-8")
        )
        mutate(document)
        raw = (
            json.dumps(document, sort_keys=True, allow_nan=False).encode()
            + b"\n"
        )
        with pytest.raises(ValueError):
            driver._authenticate_vae_manifest(
                raw, expected_sha=hashlib.sha256(raw).hexdigest(),
            )


def test_driver_lineage_summary_rejects_conflict(vae_pool, monkeypatch):
    driver = _load_driver()
    monkeypatch.setattr(
        driver, "POOL_MANIFEST_SHA256", vae_pool["manifest_sha256"],
    )
    view = vae_pool["view"]
    summary = driver._lineage_summary(
        pool_identity=view.identity,
        manifest=vae_pool["manifest"],
        source_commit=SYNTHESIS_SOURCE,
    )
    assert summary["generator_lineage_sha256"] == summary["sha256"]
    bad = dict(vae_pool["manifest"])
    bad["synthesis_source_commit"] = "0" * 40
    with pytest.raises(ValueError):
        driver._lineage_summary(
            pool_identity=view.identity,
            manifest=bad,
            source_commit=SYNTHESIS_SOURCE,
        )


def _paired_evidence(*, family=FAMILY, manifest_sha256=None):
    arms = {
        name: {
            "report": {"selected_weights_sha256": "c" * 64},
            "pool_consumption": (
                {"evaluator_seed": SEED}
                if name == "real_plus_synthetic" else None
            ),
        }
        for name in ("real_only", "real_plus_synthetic")
    }
    return {
        "status": "PAIRED_WEIGHTS_PREPARED_NOT_EVALUATED",
        "test_evaluated": False,
        "complete_evaluation": False,
        "initial_weights_sha256": "d" * 64,
        "arms": arms,
        "augmented_identity": {
            "condition": "real_plus_synthetic",
            "position_order": "real_then_synthetic",
            "real": {"size": 8117},
            "synthetic": {
                "dataset_id": DATASET_ID,
                "family": family,
                "experimental_seed": SEED,
                "manifest_sha256": manifest_sha256 or "e" * 64,
                "checkpoint_manifest_sha256": CHECKPOINT_SHA,
            },
        },
    }


def test_driver_training_evidence_binds_vae_identity(
    vae_pool, monkeypatch,
):
    driver = _load_driver()
    monkeypatch.setattr(
        driver, "POOL_MANIFEST_SHA256", vae_pool["manifest_sha256"],
    )
    paired = _paired_evidence()
    augmented_identity = paired["augmented_identity"]
    augmented_identity["synthetic"]["manifest_sha256"] = (
        vae_pool["manifest_sha256"]
    )
    result = driver._require_training_evidence(
        {k: v for k, v in paired.items() if k != "augmented_identity"},
        augmented_identity=augmented_identity,
    )
    assert result is paired["arms"]["real_plus_synthetic"]["pool_consumption"]


@pytest.mark.parametrize(
    "override",
    [
        {"family": "gan"},
        {"manifest_sha256": "f" * 64},
    ],
)
def test_driver_training_evidence_rejects_conflict(
    override, vae_pool, monkeypatch,
):
    driver = _load_driver()
    monkeypatch.setattr(
        driver, "POOL_MANIFEST_SHA256", vae_pool["manifest_sha256"],
    )
    paired = _paired_evidence(**override)
    with pytest.raises(ValueError):
        driver._require_training_evidence(
            {k: v for k, v in paired.items() if k != "augmented_identity"},
            augmented_identity=paired["augmented_identity"],
        )


class SplitAdapter:
    def split_size(self, split):
        return SPLIT_COUNTS[split]

    def labels(self, split):
        return np.repeat(
            np.arange(CLASS_COUNT, dtype=np.int64),
            np.asarray(SPLIT_CLASS_COUNTS[split], dtype=np.int64),
        )


def test_driver_split_identity_matches_governed_counts():
    driver = _load_driver()
    evidence = driver._verify_split_identities(
        SplitAdapter(), include_test=True,
    )
    assert evidence["split_sizes"] == SPLIT_COUNTS
    assert evidence["split_class_counts"] == {
        split: tuple(counts)
        for split, counts in SPLIT_CLASS_COUNTS.items()
    }


def test_driver_train_phase_binds_artifacts_without_test_access(
    tmp_path, monkeypatch,
):
    from types import SimpleNamespace

    driver = _load_driver()
    monkeypatch.setattr(
        driver, "POOL_MANIFEST_SHA256", "e" * 64,
    )
    monkeypatch.setenv("SLURM_JOB_ID", "999999")

    real_identity = {
        "dataset_id": DATASET_ID, "split": "train", "size": 8117,
        "class_count": CLASS_COUNT, "declared_provenance_sha256": "a" * 64,
    }
    validation_identity = {
        "dataset_id": DATASET_ID, "split": "validation", "size": 1158,
        "class_count": CLASS_COUNT, "declared_provenance_sha256": "a" * 64,
    }
    synthetic_identity = {
        "dataset_id": DATASET_ID, "family": FAMILY,
        "experimental_seed": SEED, "manifest_sha256": "e" * 64,
        "checkpoint_manifest_sha256": CHECKPOINT_SHA,
    }
    augmented_identity = {
        "condition": "real_plus_synthetic",
        "position_order": "real_then_synthetic",
        "real": real_identity,
        "synthetic": synthetic_identity,
    }
    real = SimpleNamespace(size=8117, identity=real_identity)
    validation = SimpleNamespace(size=1158, identity=validation_identity)
    augmented = SimpleNamespace(size=18117, identity=augmented_identity)

    monkeypatch.setattr(
        driver, "_prepare_inputs",
        lambda args, *, include_test_identity: (
            _assert_no_test(include_test_identity)
            or {
                "root": tmp_path, "head": "a" * 40,
                "sources": {"src/trustforge/evaluation/x.py": "b" * 64},
                "config_raw": b"{}",
                "recipes": {
                    "classifier": b"classifier",
                    "metric": b"metric",
                },
                "provenance_sha256": "a" * 64,
                "components": {},
                "pool_sha256": "e" * 64,
                "lineage": {"sha256": "c" * 64},
                "pool": SimpleNamespace(identity=synthetic_identity),
            }
        ),
    )
    monkeypatch.setattr(driver, "_environment", lambda: {"stub": True})
    monkeypatch.setattr(
        driver, "_build_training_views",
        lambda session, components: (real, validation, augmented),
    )
    monkeypatch.setattr(driver, "_verify_imports", lambda *a, **k: 0)
    monkeypatch.setattr(driver, "_verify_source", lambda *a, **k: None)

    paired = {
        "status": "PAIRED_WEIGHTS_PREPARED_NOT_EVALUATED",
        "test_evaluated": False,
        "complete_evaluation": False,
        "initial_weights_sha256": "d" * 64,
        "arms": {
            "real_only": {
                "report": {"selected_weights_sha256": "1" * 64},
                "selected_weights": (np.zeros(4, dtype=np.float32),),
                "pool_consumption": None,
            },
            "real_plus_synthetic": {
                "report": {"selected_weights_sha256": "2" * 64},
                "selected_weights": (np.ones(4, dtype=np.float32),),
                "pool_consumption": {"schema": "stub"},
            },
        },
    }
    from trustforge.evaluation import (
        paper01_classifier_bundle as bundle_module,
        paper01_classifier_manifest as manifest_module,
        paper01_classifier_state as state_module,
        paper01_classifier_tensorflow as tensorflow_module,
    )

    monkeypatch.setattr(
        tensorflow_module, "require_compute_node", lambda: None,
    )
    monkeypatch.setattr(
        tensorflow_module, "prepare_paired_classifiers",
        lambda **kwargs: paired,
    )
    monkeypatch.setattr(
        state_module, "encode_classifier_weights",
        lambda **kwargs: b"state",
    )
    monkeypatch.setattr(
        manifest_module, "prepare_classifier_manifest",
        lambda **kwargs: {"context": kwargs["context"]},
    )
    monkeypatch.setattr(
        manifest_module, "canonical_manifest",
        lambda value: b"manifest",
    )
    published = []
    monkeypatch.setattr(
        bundle_module, "publish_classifier_bundle",
        lambda *args, **kwargs: published.append((args, kwargs)),
    )

    args = SimpleNamespace(
        run_dir=tmp_path / "run", expected_head="a" * 40,
        pool_manifest_sha256="e" * 64,
    )
    report = driver._run_train(args)

    run = args.run_dir
    assert (run / "context_evidence.json").is_file()
    assert (run / "pool_consumption.json").is_file()
    assert (run / "train_report.json").is_file()
    assert (run / "expectations.json").is_file()
    assert report["test_accessed"] is False
    assert report["complete_evaluation"] is False
    assert [call[0][1] for call in published] == [
        "real_only", "real_plus_synthetic",
    ]
    receipt = json.loads((run / "expectations.json").read_text())
    assert set(receipt["arms"]) == {"real_only", "real_plus_synthetic"}
    assert receipt["synthetic_pool_manifest_sha256"] == "e" * 64
    context = json.loads((run / "context_evidence.json").read_text())
    assert context["augmented_training_identity"] == augmented_identity


def _assert_no_test(include_test_identity):
    assert include_test_identity is False
    return None
