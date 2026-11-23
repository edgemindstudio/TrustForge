"""Focused Diffusion Stage 4 production-integration fixtures.

No TensorFlow training, image synthesis, real data or Slurm execution occurs.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from trustforge.models.paper01_revision_2026.diffusion_checkpoint_bundle import (
    DiffusionBundleError,
    load_diffusion_checkpoint_bundle,
)
from trustforge.models.paper01_revision_2026.diffusion_checkpoint_state import (
    prepare_diffusion_checkpoint_states,
)
from trustforge.models.paper01_revision_2026.synthesis_linkage import (
    SynthesisRequest,
    load_synthesis_recipe,
    restore_diffusion_model_for_synthesis,
)


REPO = Path(__file__).resolve().parents[1]
CLI_PATH = (
    REPO / "scripts" / "train_paper01_diffusion_revision_production.py"
)
DATASET = "cicmaldroid2020"
MAPPING = tuple(str(index) for index in range(5))
COMMIT = "a" * 40
RUN_IDENTITY = "commitaaaaaaa_attempt01"
PROVENANCE = {"fixture_only": True}
CONFIGURATION = {
    "image_shape": [40, 40, 3], "depth": 2,
    "base_filters": 64, "time_embedding_dim": 128,
    "timesteps": 1000, "schedule": "linear",
    "beta_start": 1e-4, "beta_end": 2e-2,
    "epochs": 2, "batch_size": 64,
    "learning_rate": 2e-4, "beta_1": 0.9,
}


class DummyModel:
    def __init__(self):
        self.values = [np.zeros((2, 3), dtype=np.float32)]

    def get_weights(self):
        return [value.copy() for value in self.values]

    def set_weights(self, weights):
        self.values = [value.copy() for value in weights]


def load_cli():
    spec = importlib.util.spec_from_file_location(
        "paper01_diffusion_stage4_cli", CLI_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def cli():
    return load_cli()


def governed_result():
    history = (
        {
            "epoch": 0, "training_examples": 3,
            "validation_examples": 2, "completed_updates": 1,
            "train_mse": 1.2, "validation_mse": 0.7,
        },
        {
            "epoch": 1, "training_examples": 3,
            "validation_examples": 2, "completed_updates": 2,
            "train_mse": 1.0, "validation_mse": 0.8,
        },
    )
    return {
        "dataset_id": DATASET,
        "experimental_seed": 42,
        "class_count": 5,
        "dataset_provenance": PROVENANCE,
        "scientific_configuration": CONFIGURATION,
        "checkpoint_status": "NOT_PREPARED",
        "publication_status": "NOT_PUBLISHED",
        "training_result": {
            "history": history, "best_epoch": 0,
            "best_validation_mse": 0.7,
            "best_weights": (np.full((2, 3), 1, dtype=np.float32),),
            "last_weights": (np.full((2, 3), 2, dtype=np.float32),),
            "completed_epochs": 2, "completed_updates": 2,
            "train_size": 3, "validation_size": 2,
        },
    }


def request():
    return {
        "dataset_id": DATASET,
        "experimental_seed": 42,
        "attempt": 1,
        "expected_source_commit": COMMIT,
        "run_identity": RUN_IDENTITY,
    }


def environment():
    return SimpleNamespace(
        actual_source_commit=COMMIT, repository_root=REPO,
    )


def publish(cli, tmp_path, result=None):
    return cli.publish_governed_checkpoint(
        result=governed_result() if result is None else result,
        request=request(),
        environment=environment(),
        adapter=object(),
        checkpoint_root=tmp_path,
    )


def make_args(tmp_path):
    return SimpleNamespace(
        dataset_id=DATASET,
        experimental_seed=42,
        attempt=1,
        expected_source_commit=COMMIT,
        repository_root=REPO,
        checkpoint_root=tmp_path,
        ustc_source_dir=None,
        cic_source_csv=Path("/fixture/source.csv"),
        cic_feature_columns=Path("/fixture/features.json"),
        malnet_image_root=None,
        malnet_preparation_dir=None,
        malnet_mapping_path=None,
        malex_label_archive=None,
        malex_benign_root=None,
        malex_malware_root=None,
    )


def test_training_result_publishes_attempt_scoped_bundle(cli, tmp_path):
    publication = publish(cli, tmp_path)

    expected_directory = (
        tmp_path
        / DATASET
        / "seed42"
        / "checkpoints"
        / "production"
        / RUN_IDENTITY
    )
    assert Path(publication["checkpoint_directory"]) == expected_directory
    assert (expected_directory / "manifest.json").is_file()

    loaded = load_diffusion_checkpoint_bundle(
        expected_directory,
        expected_manifest_sha256=publication["manifest_sha256"],
        requested_dataset_id=DATASET,
        requested_experimental_seed=42,
        requested_class_mapping=MAPPING,
        requested_role="best",
        expected_source_commit=COMMIT,
    )
    assert loaded["best"]["state_sha256"] == (
        publication["best_state_sha256"]
    )
    assert loaded["best"]["checkpoint_role"] == "best"
    assert publication["last_state_sha256"] != (
        publication["best_state_sha256"]
    )
    np.testing.assert_array_equal(
        loaded["best"]["weights"][0],
        np.full((2, 3), 1, dtype=np.float32),
    )


def test_publication_requires_eligible_training_result(cli, tmp_path):
    result = governed_result()
    result["checkpoint_status"] = "PREPARED"
    with pytest.raises(ValueError, match="not eligible"):
        publish(cli, tmp_path, result=result)


def test_wrong_training_identity_rejected(cli, tmp_path):
    result = governed_result()
    result["dataset_id"] = "malex"
    with pytest.raises(ValueError, match="identity differs"):
        publish(cli, tmp_path, result=result)


def test_publication_failure_does_not_return_success(cli, tmp_path):
    destination = (
        tmp_path
        / DATASET
        / "seed42"
        / "checkpoints"
        / "production"
        / RUN_IDENTITY
    )
    destination.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        publish(cli, tmp_path)


def test_entry_point_publishes_and_reports_manifest(cli, tmp_path):
    result = governed_result()

    class Governed:
        DIFFUSION_REAL_DATASET_TRAINING_ENABLED = True

        @staticmethod
        def run_governed_diffusion_training(**kwargs):
            return result

    cli._load_governed_module = lambda: Governed
    cli.require_environment = lambda args: environment()
    cli.require_module_binding = lambda **kwargs: SimpleNamespace(
        module_paths={"governed": "fixture"},
        model_builder_path=str(REPO / "diffusion" / "models.py"),
    )
    cli.build_adapter = lambda args: object()

    summary = cli.run(make_args(tmp_path))

    assert summary["checkpoint_status"] == "PREPARED"
    assert summary["publication_status"] == "PUBLISHED"
    publication = summary["checkpoint_publication"]
    assert publication["checkpoint_role_primary"] == "best"
    assert (Path(publication["checkpoint_directory"]) / "manifest.json").is_file()
    assert publication["manifest_sha256"] != publication["best_state_sha256"]


def test_entry_point_publication_failure_prints_no_success(
    cli, tmp_path, capsys,
):
    destination = (
        tmp_path
        / DATASET
        / "seed42"
        / "checkpoints"
        / "production"
        / RUN_IDENTITY
    )
    destination.mkdir(parents=True)

    class Governed:
        DIFFUSION_REAL_DATASET_TRAINING_ENABLED = True

        @staticmethod
        def run_governed_diffusion_training(**kwargs):
            return governed_result()

    cli._load_governed_module = lambda: Governed
    cli.require_environment = lambda args: environment()
    cli.require_module_binding = lambda **kwargs: SimpleNamespace(
        module_paths={"governed": "fixture"},
        model_builder_path=str(REPO / "diffusion" / "models.py"),
    )
    cli.build_adapter = lambda args: object()

    with pytest.raises(FileExistsError):
        cli.run(make_args(tmp_path))
    assert "DIFFUSION_REVISION_PRODUCTION_RESULT" not in capsys.readouterr().out


def test_published_best_restores_for_synthesis(cli, tmp_path):
    publication = publish(cli, tmp_path)
    recipe = load_synthesis_recipe(REPO)
    synthesis_request = SynthesisRequest(
        dataset_id=DATASET, family="diffusion", experimental_seed=42,
        class_id=0, sample_indices=(0,), class_mapping=MAPPING,
        scope="production_batch",
    )
    identity = {
        "dataset_id": DATASET, "experimental_seed": 42,
        "class_mapping": list(MAPPING), "checkpoint_role": "best",
        "checkpoint_sha256": publication["best_state_sha256"],
        "source_commit": COMMIT,
    }
    model = restore_diffusion_model_for_synthesis(
        checkpoint_directory=publication["checkpoint_directory"],
        expected_manifest_sha256=publication["manifest_sha256"],
        prepared=None, model_factory=DummyModel,
        request=synthesis_request, recipe=recipe,
        checkpoint_identity=identity,
    )
    np.testing.assert_array_equal(
        model.get_weights()[0],
        np.full((2, 3), 1, dtype=np.float32),
    )


def test_requested_digest_must_match_published_best(cli, tmp_path):
    publication = publish(cli, tmp_path)
    recipe = load_synthesis_recipe(REPO)
    synthesis_request = SynthesisRequest(
        dataset_id=DATASET, family="diffusion", experimental_seed=42,
        class_id=0, sample_indices=(0,), class_mapping=MAPPING,
        scope="production_batch",
    )
    identity = {
        "dataset_id": DATASET, "experimental_seed": 42,
        "class_mapping": list(MAPPING), "checkpoint_role": "best",
        "checkpoint_sha256": "0" * 64,
        "source_commit": COMMIT,
    }
    with pytest.raises(ValueError, match="does not match"):
        restore_diffusion_model_for_synthesis(
            checkpoint_directory=publication["checkpoint_directory"],
            expected_manifest_sha256=publication["manifest_sha256"],
            prepared=None, model_factory=DummyModel,
            request=synthesis_request, recipe=recipe,
            checkpoint_identity=identity,
        )


@pytest.mark.parametrize("overrides", [
    {"checkpoint_role": "last"},
    {"dataset_id": "malex"},
    {"experimental_seed": 43},
    {"class_mapping": list(reversed(MAPPING))},
])
def test_wrong_synthesis_identity_rejected(cli, tmp_path, overrides):
    publication = publish(cli, tmp_path)
    recipe = load_synthesis_recipe(REPO)
    synthesis_request = SynthesisRequest(
        dataset_id=DATASET, family="diffusion", experimental_seed=42,
        class_id=0, sample_indices=(0,), class_mapping=MAPPING,
        scope="production_batch",
    )
    identity = {
        "dataset_id": DATASET, "experimental_seed": 42,
        "class_mapping": list(MAPPING), "checkpoint_role": "best",
        "checkpoint_sha256": publication["best_state_sha256"],
        "source_commit": COMMIT,
    }
    identity.update(overrides)
    with pytest.raises(ValueError):
        restore_diffusion_model_for_synthesis(
            checkpoint_directory=publication["checkpoint_directory"],
            expected_manifest_sha256=publication["manifest_sha256"],
            prepared=None, model_factory=DummyModel,
            request=synthesis_request, recipe=recipe,
            checkpoint_identity=identity,
        )


def test_missing_manifest_digest_rejected(cli, tmp_path):
    publication = publish(cli, tmp_path)
    recipe = load_synthesis_recipe(REPO)
    synthesis_request = SynthesisRequest(
        dataset_id=DATASET, family="diffusion", experimental_seed=42,
        class_id=0, sample_indices=(0,), class_mapping=MAPPING,
        scope="production_batch",
    )
    identity = {
        "dataset_id": DATASET, "experimental_seed": 42,
        "class_mapping": list(MAPPING), "checkpoint_role": "best",
        "checkpoint_sha256": publication["best_state_sha256"],
    }
    with pytest.raises(ValueError, match="manifest SHA-256"):
        restore_diffusion_model_for_synthesis(
            checkpoint_directory=publication["checkpoint_directory"],
            expected_manifest_sha256=None,
            prepared=None, model_factory=DummyModel,
            request=synthesis_request, recipe=recipe,
            checkpoint_identity=identity,
        )


def test_corrupted_published_bundle_rejected(cli, tmp_path):
    publication = publish(cli, tmp_path)
    path = Path(publication["checkpoint_directory"]) / "best" / "w000.npy"
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 0xFF
    path.write_bytes(bytes(raw))
    recipe = load_synthesis_recipe(REPO)
    synthesis_request = SynthesisRequest(
        dataset_id=DATASET, family="diffusion", experimental_seed=42,
        class_id=0, sample_indices=(0,), class_mapping=MAPPING,
        scope="production_batch",
    )
    identity = {
        "dataset_id": DATASET, "experimental_seed": 42,
        "class_mapping": list(MAPPING), "checkpoint_role": "best",
        "checkpoint_sha256": publication["best_state_sha256"],
    }
    with pytest.raises(DiffusionBundleError, match="tensor integrity"):
        restore_diffusion_model_for_synthesis(
            checkpoint_directory=publication["checkpoint_directory"],
            expected_manifest_sha256=publication["manifest_sha256"],
            prepared=None, model_factory=DummyModel,
            request=synthesis_request, recipe=recipe,
            checkpoint_identity=identity,
        )


def test_prepared_only_production_request_rejected(cli):
    recipe = load_synthesis_recipe(REPO)
    prepared = prepare_diffusion_checkpoint_states(
        governed_result=governed_result(), class_mapping=MAPPING,
    )
    synthesis_request = SynthesisRequest(
        dataset_id=DATASET, family="diffusion", experimental_seed=42,
        class_id=0, sample_indices=(0,), class_mapping=MAPPING,
        scope="production_batch",
    )
    identity = {
        "dataset_id": DATASET, "experimental_seed": 42,
        "class_mapping": list(MAPPING), "checkpoint_role": "best",
        "checkpoint_sha256": prepared["best"]["state_sha256"],
    }
    with pytest.raises(ValueError, match="published checkpoint directory"):
        restore_diffusion_model_for_synthesis(
            prepared=prepared["best"], model_factory=DummyModel,
            request=synthesis_request, recipe=recipe,
            checkpoint_identity=identity,
        )


def test_both_checkpoint_sources_rejected(cli, tmp_path):
    publication = publish(cli, tmp_path)
    recipe = load_synthesis_recipe(REPO)
    prepared = prepare_diffusion_checkpoint_states(
        governed_result=governed_result(), class_mapping=MAPPING,
    )
    synthesis_request = SynthesisRequest(
        dataset_id=DATASET, family="diffusion", experimental_seed=42,
        class_id=0, sample_indices=(0,), class_mapping=MAPPING,
        scope="fixture",
    )
    identity = {
        "dataset_id": DATASET, "experimental_seed": 42,
        "class_mapping": list(MAPPING), "checkpoint_role": "best",
        "checkpoint_sha256": prepared["best"]["state_sha256"],
    }
    with pytest.raises(ValueError, match="exactly one"):
        restore_diffusion_model_for_synthesis(
            prepared=prepared["best"],
            checkpoint_directory=publication["checkpoint_directory"],
            expected_manifest_sha256=publication["manifest_sha256"],
            model_factory=DummyModel, request=synthesis_request,
            recipe=recipe, checkpoint_identity=identity,
        )


def test_existing_fixture_prepared_path_still_restores(cli):
    recipe = load_synthesis_recipe(REPO)
    prepared = prepare_diffusion_checkpoint_states(
        governed_result=governed_result(), class_mapping=MAPPING,
    )
    synthesis_request = SynthesisRequest(
        dataset_id=DATASET, family="diffusion", experimental_seed=42,
        class_id=0, sample_indices=(0,), class_mapping=MAPPING,
        scope="fixture",
    )
    identity = {
        "dataset_id": DATASET, "experimental_seed": 42,
        "class_mapping": list(MAPPING), "checkpoint_role": "best",
    }
    model = restore_diffusion_model_for_synthesis(
        prepared=prepared["best"], model_factory=DummyModel,
        request=synthesis_request, recipe=recipe,
        checkpoint_identity=identity,
    )
    np.testing.assert_array_equal(
        model.get_weights()[0],
        np.full((2, 3), 1, dtype=np.float32),
    )
