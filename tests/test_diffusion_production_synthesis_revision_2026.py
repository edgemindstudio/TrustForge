"""Focused fixture tests for governed Diffusion production synthesis.

No real cybersecurity data, accepted real checkpoint, training, production
sampling or Slurm execution is used. Checkpoint bundles and models are local
fixtures.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from trustforge.models.paper01_revision_2026 import (
    diffusion_production_synthesis as synthesis,
)
from trustforge.models.paper01_revision_2026.diffusion_checkpoint_bundle import (
    publish_diffusion_checkpoint_bundle,
)
from trustforge.models.paper01_revision_2026.diffusion_checkpoint_state import (
    prepare_diffusion_checkpoint_states,
)
from trustforge.models.paper01_revision_2026.diffusion_pool_publication import (
    ACCEPTED_SAMPLING_POLICY,
    DiffusionRangeIdentity,
    verify_diffusion_range,
)
from trustforge.models.paper01_revision_2026.synthesis_linkage import (
    load_synthesis_recipe,
)


REPO = Path(__file__).resolve().parents[1]
DATASET = "cicmaldroid2020"
MAPPING = tuple(str(index) for index in range(5))
SYNTHESIS_COMMIT = "a" * 40
TRAINING_COMMIT = "b" * 40
RUN_IDENTITY = f"commit{SYNTHESIS_COMMIT[:7]}_attempt01"
CONFIGURATION = {
    "image_shape": [40, 40, 3],
    "depth": 2,
    "base_filters": 64,
    "time_embedding_dim": 128,
    "timesteps": 1000,
    "schedule": "linear",
    "beta_start": 1e-4,
    "beta_end": 2e-2,
    "epochs": 2,
    "batch_size": 64,
    "learning_rate": 2e-4,
    "beta_1": 0.9,
}
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
ALLOCATION = {
    "JobId": "900001",
    "Partition": "talon",
    "NumNodes": "1",
    "NumTasks": "1",
    "NumCPUs": "8",
    "AllocTRES": "cpu=8,mem=32G,node=1,billing=8",
    "TimeLimit": "01:00:00",
    "Requeue": "0",
    "Restarts": "0",
    "Gres": "(null)",
}


class DummyModel:
    def __init__(self):
        self.values = [np.zeros((2, 3), dtype=np.float32)]

    def get_weights(self):
        return [value.copy() for value in self.values]

    def set_weights(self, weights):
        self.values = [value.copy() for value in weights]


class SamplingModel(DummyModel):
    instances = []

    def __init__(self):
        super().__init__()
        self.calls = 0
        self.batch_sizes = []
        SamplingModel.instances.append(self)

    def __call__(self, inputs, training=False):
        x = np.asarray(inputs[0])
        self.calls += 1
        self.batch_sizes.append(x.shape[0])
        return np.zeros(x.shape, dtype=np.float32)


class DummyBuilder:
    def __init__(self, model_type=DummyModel):
        self.model_type = model_type
        self.calls = []

    def build_diffusion_model(self, **kwargs):
        self.calls.append(dict(kwargs))
        return self.model_type()


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
        "dataset_provenance": {"fixture_only": True},
        "scientific_configuration": CONFIGURATION,
        "checkpoint_status": "NOT_PREPARED",
        "publication_status": "NOT_PUBLISHED",
        "training_result": {
            "history": history,
            "best_epoch": 0,
            "best_validation_mse": 0.7,
            "best_weights": (
                np.full((2, 3), 1, dtype=np.float32),
            ),
            "last_weights": (
                np.full((2, 3), 2, dtype=np.float32),
            ),
            "completed_epochs": 2,
            "completed_updates": 2,
            "train_size": 3,
            "validation_size": 2,
        },
    }


@pytest.fixture()
def bundle(tmp_path):
    prepared = prepare_diffusion_checkpoint_states(
        governed_result=governed_result(),
        class_mapping=MAPPING,
    )
    destination = tmp_path / "checkpoint"
    publication = publish_diffusion_checkpoint_bundle(
        prepared=prepared,
        destination=destination,
        source_commit=TRAINING_COMMIT,
        dataset_provenance={"fixture_only": True},
    )
    return {
        "directory": destination,
        "manifest_sha256": publication["manifest_sha256"],
        "state_sha256": prepared["best"]["state_sha256"],
    }


def make_request(
    tmp_path,
    bundle,
    *,
    class_id=0,
    sample_start=0,
    sample_stop=2,
    attempt=1,
    output_name="output",
    commit=SYNTHESIS_COMMIT,
    recipe_sha256=None,
    manifest_sha256=None,
    state_sha256=None,
    training_commit=TRAINING_COMMIT,
):
    return synthesis.DiffusionProductionSynthesisRequest(
        dataset_id=DATASET,
        experimental_seed=42,
        attempt=attempt,
        expected_authorization_id="P1R2026-DIFF-FIXTURE-A01",
        class_id=class_id,
        sample_start=sample_start,
        sample_stop=sample_stop,
        expected_synthesis_source_commit=commit,
        repository_root=REPO,
        checkpoint_directory=bundle["directory"],
        expected_checkpoint_manifest_sha256=(
            bundle["manifest_sha256"]
            if manifest_sha256 is None
            else manifest_sha256
        ),
        expected_checkpoint_state_sha256=(
            bundle["state_sha256"]
            if state_sha256 is None
            else state_sha256
        ),
        expected_checkpoint_training_source_commit=training_commit,
        expected_recipe_sha256=(
            load_synthesis_recipe(REPO).sha256
            if recipe_sha256 is None
            else recipe_sha256
        ),
        output_destination=tmp_path / output_name,
    )


def make_restoration_request(tmp_path, bundle, **overrides):
    values = {
        "dataset_id": DATASET,
        "experimental_seed": 42,
        "attempt": 1,
        "expected_authorization_id": "P1R2026-DIFF-FIXTURE-A01",
        "expected_synthesis_source_commit": SYNTHESIS_COMMIT,
        "repository_root": REPO,
        "checkpoint_directory": bundle["directory"],
        "expected_checkpoint_manifest_sha256": bundle["manifest_sha256"],
        "expected_checkpoint_state_sha256": bundle["state_sha256"],
        "expected_checkpoint_training_source_commit": TRAINING_COMMIT,
        "expected_recipe_sha256": load_synthesis_recipe(REPO).sha256,
    }
    values.update(overrides)
    return synthesis.DiffusionRestorationVerificationRequest(**values)


def authorization_record(request, *, mode, resources=None, **overrides):
    synthesis_mode = mode == "synthesis"
    record = {
        "schema": synthesis.SCHEMA,
        "paper": synthesis.PAPER,
        "revision": synthesis.REVISION,
        "family": synthesis.FAMILY,
        "mode": mode,
        "dataset_id": request.dataset_id,
        "experimental_seed": request.experimental_seed,
        "attempt": request.attempt,
        "class_id": request.class_id if synthesis_mode else None,
        "sample_start": request.sample_start if synthesis_mode else None,
        "sample_stop": request.sample_stop if synthesis_mode else None,
        "checkpoint_directory": str(request.checkpoint_directory),
        "checkpoint_manifest_sha256": (
            request.expected_checkpoint_manifest_sha256
        ),
        "checkpoint_state_sha256":
            request.expected_checkpoint_state_sha256,
        "checkpoint_training_source_commit": (
            request.expected_checkpoint_training_source_commit
        ),
        "synthesis_source_commit": (
            request.expected_synthesis_source_commit
        ),
        "recipe_sha256": request.expected_recipe_sha256,
        "output_destination": (
            str(request.output_destination) if synthesis_mode else None
        ),
        "authorization_id": "P1R2026-DIFF-FIXTURE-A01",
        "resources": dict(resources or RESOURCES),
    }
    record.update(overrides)
    return record


def write_authorization(monkeypatch, tmp_path, request, *, mode, **overrides):
    monkeypatch.setattr(
        synthesis, "AUTHORIZATION_ROOT", tmp_path / "authorizations",
    )
    path = synthesis.authorization_path(request, mode=mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            authorization_record(request, mode=mode, **overrides)
        ),
        encoding="utf-8",
    )
    path.chmod(0o600)
    return path


def enable_fixture_execution(
    monkeypatch,
    *,
    model_type=DummyModel,
    allocation=None,
):
    monkeypatch.setattr(
        synthesis, "DIFFUSION_PRODUCTION_SYNTHESIS_ENABLED", True,
    )
    monkeypatch.setattr(
        synthesis,
        "capture_slurm_allocation",
        lambda: dict(allocation or ALLOCATION),
    )
    monkeypatch.setattr(
        synthesis,
        "require_source_binding",
        lambda request: synthesis.SynthesisSourceBinding(
            SYNTHESIS_COMMIT, SYNTHESIS_COMMIT, REPO,
        ),
    )
    monkeypatch.setattr(
        synthesis,
        "require_module_binding",
        lambda **kwargs: synthesis.ModuleBinding(
            module_paths={"fixture": "fixture"},
            model_builder_path=str(REPO / "diffusion" / "models.py"),
        ),
    )
    builder = DummyBuilder(model_type)
    monkeypatch.setattr(
        synthesis, "_load_model_builder_module", lambda: builder,
    )
    monkeypatch.setattr(
        synthesis,
        "_build_model_factory",
        lambda model_builder, num_classes: (
            lambda: model_builder.model_type()
        ),
    )
    return builder


def fixture_images(count, offset=0):
    total = count * 40 * 40 * 3
    values = np.linspace(
        0.0, 1.0, num=total, dtype=np.float32,
    )
    values = np.roll(values, offset).astype(np.float32)
    return values.reshape((count, 40, 40, 3))


# Authorization and execution identity


def test_disabled_gate_rejects_before_checkpoint_access(
    tmp_path, bundle, monkeypatch,
):
    assert synthesis.DIFFUSION_PRODUCTION_SYNTHESIS_ENABLED is False
    request = make_request(tmp_path, bundle)
    write_authorization(monkeypatch, tmp_path, request, mode="synthesis")
    calls = []
    monkeypatch.setattr(
        synthesis,
        "restore_diffusion_model_for_synthesis",
        lambda **kwargs: calls.append(kwargs),
    )

    with pytest.raises(
        synthesis.DiffusionProductionAuthorizationError,
        match="not authorized",
    ):
        synthesis.execute_diffusion_production_synthesis(request)
    assert calls == []


def test_missing_authorization_rejects_before_checkpoint(
    tmp_path, bundle, monkeypatch,
):
    enable_fixture_execution(monkeypatch)
    request = make_request(tmp_path, bundle)
    monkeypatch.setattr(
        synthesis, "AUTHORIZATION_ROOT", tmp_path / "authorizations",
    )
    calls = []
    monkeypatch.setattr(
        synthesis,
        "restore_diffusion_model_for_synthesis",
        lambda **kwargs: calls.append(kwargs),
    )

    with pytest.raises(
        synthesis.DiffusionProductionAuthorizationError,
        match="absent",
    ):
        synthesis.execute_diffusion_production_synthesis(request)
    assert calls == []


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"checkpoint_manifest_sha256": "f" * 64}, "checkpoint_manifest"),
        ({"checkpoint_state_sha256": "f" * 64}, "checkpoint_state"),
        (
            {"checkpoint_training_source_commit": "f" * 40},
            "checkpoint_training",
        ),
        ({"recipe_sha256": "f" * 64}, "recipe"),
        ({"synthesis_source_commit": "f" * 40}, "synthesis_source"),
        ({"authorization_id": "P1R2026-DIFF-OTHER"}, "authorization_id"),
    ],
)
def test_mismatched_authorization_identity_rejected(
    tmp_path, bundle, monkeypatch, overrides, message,
):
    enable_fixture_execution(monkeypatch)
    request = make_request(tmp_path, bundle)
    write_authorization(
        monkeypatch, tmp_path, request, mode="synthesis", **overrides,
    )

    with pytest.raises(
        synthesis.DiffusionProductionAuthorizationError, match=message,
    ):
        synthesis.execute_diffusion_production_synthesis(request)


def test_wrong_request_range_and_class_rejected(tmp_path, bundle):
    with pytest.raises(ValueError, match="sample_stop"):
        make_request(tmp_path, bundle, sample_stop=2001)
    with pytest.raises(ValueError, match="less than"):
        make_request(tmp_path, bundle, sample_start=5, sample_stop=5)
    with pytest.raises(ValueError, match="class universe"):
        make_request(tmp_path, bundle, class_id=5)


def test_missing_slurm_allocation_rejected(
    tmp_path, bundle, monkeypatch,
):
    enable_fixture_execution(monkeypatch)
    request = make_request(tmp_path, bundle)
    write_authorization(monkeypatch, tmp_path, request, mode="synthesis")

    def missing():
        raise synthesis.DiffusionProductionAuthorizationError(
            "a Slurm job identity is required"
        )

    monkeypatch.setattr(synthesis, "capture_slurm_allocation", missing)

    with pytest.raises(
        synthesis.DiffusionProductionAuthorizationError, match="Slurm",
    ):
        synthesis.execute_diffusion_production_synthesis(request)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"NumCPUs": "4"}, "CPU"),
        ({"AllocTRES": "cpu=8,mem=16G,node=1"}, "memory"),
        ({"TimeLimit": "02:00:00"}, "walltime"),
        ({"NumNodes": "2"}, "node"),
        ({"Partition": "other"}, "partition"),
        ({"Requeue": "1"}, "requeue"),
    ],
)
def test_allocation_mismatch_rejected(
    tmp_path, bundle, monkeypatch, overrides, message,
):
    allocation = dict(ALLOCATION)
    allocation.update(overrides)
    enable_fixture_execution(monkeypatch, allocation=allocation)
    request = make_request(tmp_path, bundle)
    write_authorization(monkeypatch, tmp_path, request, mode="synthesis")

    with pytest.raises(
        synthesis.DiffusionProductionAuthorizationError, match=message,
    ):
        synthesis.execute_diffusion_production_synthesis(request)


def test_wrong_recipe_identity_rejected_before_authorization(
    tmp_path, bundle, monkeypatch,
):
    enable_fixture_execution(monkeypatch)
    request = make_request(tmp_path, bundle, recipe_sha256="f" * 64)
    monkeypatch.setattr(
        synthesis, "AUTHORIZATION_ROOT", tmp_path / "authorizations",
    )

    with pytest.raises(
        synthesis.DiffusionProductionSynthesisError, match="recipe",
    ):
        synthesis.execute_diffusion_production_synthesis(request)


# Restoration linkage


def test_restoration_only_restores_without_sampling(
    tmp_path, bundle, monkeypatch,
):
    enable_fixture_execution(monkeypatch)
    request = make_restoration_request(tmp_path, bundle)
    write_authorization(monkeypatch, tmp_path, request, mode="restoration")

    def no_sampling(**kwargs):
        raise AssertionError("restoration-only mode sampled")

    monkeypatch.setattr(
        synthesis, "synthesize_diffusion_batch", no_sampling,
    )

    report = synthesis.verify_diffusion_restoration(request)

    assert report["status"] == "RESTORATION_VERIFIED"
    assert report["restoration_status"] == "VERIFIED"
    assert report["sampling_performed"] is False
    assert report["images_generated"] == 0
    assert report["restored_parameter_count"] == 1
    assert report["checkpoint_state_sha256"] == bundle["state_sha256"]
    assert report["authorization_id"] == "P1R2026-DIFF-FIXTURE-A01"
    assert report["slurm_job_id"] == "900001"


def test_restoration_rejects_wrong_role_or_digest(
    tmp_path, bundle, monkeypatch,
):
    enable_fixture_execution(monkeypatch)
    request = make_restoration_request(
        tmp_path, bundle, expected_checkpoint_state_sha256="f" * 64,
    )
    write_authorization(monkeypatch, tmp_path, request, mode="restoration")

    with pytest.raises(ValueError):
        synthesis.verify_diffusion_restoration(request)


def test_restoration_mode_does_not_accept_sampling_scope(
    tmp_path, bundle, monkeypatch,
):
    enable_fixture_execution(monkeypatch)
    request = make_restoration_request(tmp_path, bundle)
    write_authorization(
        monkeypatch, tmp_path, request, mode="restoration",
        class_id=0, sample_start=0, sample_stop=2,
        output_destination=str(tmp_path / "out"),
    )

    with pytest.raises(
        synthesis.DiffusionProductionAuthorizationError,
        match="must not bind sampling scope",
    ):
        synthesis.verify_diffusion_restoration(request)


# Synthesis integration


def test_synthesis_consumes_linkage_and_publishes(
    tmp_path, bundle, monkeypatch,
):
    enable_fixture_execution(monkeypatch)
    request = make_request(
        tmp_path, bundle, sample_start=3, sample_stop=7,
    )
    write_authorization(monkeypatch, tmp_path, request, mode="synthesis")

    seen = {}

    def fake_synthesize(*, model, request, recipe, checkpoint_identity):
        seen["indices"] = tuple(request.sample_indices)
        seen["class_id"] = request.class_id
        seen["scope"] = request.scope
        seen["batch_size"] = recipe.data["families"]["diffusion"][
            "model_call_batch_size"
        ]
        seen["timesteps"] = recipe.data["families"]["diffusion"][
            "timesteps"
        ]
        seen["checkpoint"] = checkpoint_identity
        images = fixture_images(len(request.sample_indices), offset=17)
        return {"images": images, "record": {}}

    monkeypatch.setattr(
        synthesis, "synthesize_diffusion_batch", fake_synthesize,
    )

    summary = synthesis.execute_diffusion_production_synthesis(request)

    assert summary["status"] == "SYNTHESIS_COMPLETE"
    assert summary["sample_count"] == 4
    assert summary["run_identity"] == RUN_IDENTITY
    assert summary["slurm_job_id"] == "900001"
    assert summary["authoritative_artifact"] is True
    assert seen["indices"] == (3, 4, 5, 6)
    assert seen["class_id"] == 0
    assert seen["scope"] == "production_batch"
    assert seen["batch_size"] == 1
    assert seen["timesteps"] == 1000
    assert seen["checkpoint"]["checkpoint_sha256"] == bundle["state_sha256"]

    manifest = request.output_destination / "manifest.json"
    assert manifest.is_file()
    report = verify_diffusion_range(
        manifest,
        expected_dataset_id=DATASET,
        expected_seed=42,
        expected_class_id=0,
        expected_class_label="0",
        expected_recipe_sha256=request.expected_recipe_sha256,
    )
    assert report["sample_start"] == 3
    assert report["sample_stop"] == 7
    assert report["sample_count"] == 4
    assert report["storage_batch_count"] == 1
    assert report["authoritative_artifact"] is True


def test_real_governed_sampler_used_with_model_call_batch_one(
    tmp_path, bundle, monkeypatch,
):
    SamplingModel.instances = []
    enable_fixture_execution(
        monkeypatch, model_type=SamplingModel,
    )
    request = make_request(tmp_path, bundle, sample_stop=1)
    write_authorization(monkeypatch, tmp_path, request, mode="synthesis")

    summary = synthesis.execute_diffusion_production_synthesis(request)
    report = verify_diffusion_range(
        request.output_destination / "manifest.json"
    )

    assert summary["sample_count"] == 1
    assert report["sample_count"] == 1
    assert report["authoritative_artifact"] is True

    (model,) = SamplingModel.instances
    assert model.calls == 1000
    assert set(model.batch_sizes) == {1}


def test_invalid_synthesis_output_rejected_before_complete_publication(
    tmp_path, bundle, monkeypatch,
):
    enable_fixture_execution(monkeypatch)
    request = make_request(tmp_path, bundle)
    write_authorization(monkeypatch, tmp_path, request, mode="synthesis")

    def bad_output(*, model, request, recipe, checkpoint_identity):
        images = fixture_images(len(request.sample_indices))
        images[0, 0, 0, 0] = np.float32(np.nan)
        return {"images": images, "record": {}}

    monkeypatch.setattr(
        synthesis, "synthesize_diffusion_batch", bad_output,
    )

    with pytest.raises(ValueError, match="non-finite"):
        synthesis.execute_diffusion_production_synthesis(request)
    assert not (
        request.output_destination / "manifest.json"
    ).exists()


def test_partitioned_ranges_preserve_sample_identities(
    tmp_path, bundle, monkeypatch,
):
    enable_fixture_execution(monkeypatch)
    recorded = []

    def fake_synthesize(*, model, request, recipe, checkpoint_identity):
        recorded.append(tuple(request.sample_indices))
        images = fixture_images(len(request.sample_indices))
        return {"images": images, "record": {}}

    monkeypatch.setattr(
        synthesis, "synthesize_diffusion_batch", fake_synthesize,
    )

    first = make_request(
        tmp_path, bundle, sample_start=0, sample_stop=2,
        output_name="first",
    )
    second = make_request(
        tmp_path, bundle, sample_start=2, sample_stop=4,
        output_name="second",
    )
    write_authorization(monkeypatch, tmp_path, first, mode="synthesis")
    synthesis.execute_diffusion_production_synthesis(first)
    write_authorization(monkeypatch, tmp_path, second, mode="synthesis")
    synthesis.execute_diffusion_production_synthesis(second)

    assert recorded == [(0, 1), (2, 3)]


def test_attempt_and_job_do_not_change_sample_identities(
    tmp_path, bundle, monkeypatch,
):
    enable_fixture_execution(monkeypatch)
    identities = []

    def fake_synthesize(*, model, request, recipe, checkpoint_identity):
        identities.append((
            tuple(request.sample_indices),
            request.class_id,
            request.experimental_seed,
        ))
        images = fixture_images(len(request.sample_indices))
        return {"images": images, "record": {}}

    monkeypatch.setattr(
        synthesis, "synthesize_diffusion_batch", fake_synthesize,
    )

    first = make_request(
        tmp_path, bundle, attempt=1, output_name="a1",
    )
    second = make_request(
        tmp_path, bundle, attempt=2, output_name="a2",
    )
    write_authorization(monkeypatch, tmp_path, first, mode="synthesis")
    synthesis.execute_diffusion_production_synthesis(first)
    write_authorization(monkeypatch, tmp_path, second, mode="synthesis")
    synthesis.execute_diffusion_production_synthesis(second)

    assert identities[0][0] == identities[1][0] == (0, 1)
    assert identities[0][1:] == identities[1][1:]


def test_fixture_publication_is_not_authoritative(
    tmp_path, bundle,
):
    identity = DiffusionRangeIdentity(
        dataset_id=DATASET,
        experimental_seed=42,
        attempt=1,
        class_id=0,
        class_label="0",
        sample_start=0,
        sample_stop=2,
        checkpoint_manifest_sha256=bundle["manifest_sha256"],
        checkpoint_state_sha256=bundle["state_sha256"],
        checkpoint_training_source_commit=TRAINING_COMMIT,
        synthesis_source_commit=SYNTHESIS_COMMIT,
        recipe_sha256=load_synthesis_recipe(REPO).sha256,
        sampling_policy=dict(ACCEPTED_SAMPLING_POLICY),
        authorization_id="P1R2026-DIFF-FIXTURE",
        slurm_job_id="fixture",
        resources=dict(RESOURCES),
        scope="fixture",
    )
    from trustforge.models.paper01_revision_2026.diffusion_pool_publication import (
        publish_diffusion_range,
    )

    report = publish_diffusion_range(
        identity=identity,
        images=fixture_images(2),
        destination=tmp_path / "fixture-range",
    )
    assert report["authoritative_artifact"] is False


# Launcher and static contract


LAUNCHER_PATH = (
    REPO
    / "slurm"
    / "paper01_revision_2026"
    / "diffusion_production_synthesis.slurm"
)
CLI_PATH = (
    REPO / "scripts" / "run_paper01_diffusion_production_synthesis.py"
)


def load_cli():
    spec = importlib.util.spec_from_file_location(
        "paper01_diffusion_synthesis_cli", CLI_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load Diffusion synthesis CLI")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_launcher_bash_syntax():
    completed = subprocess.run(
        ["bash", "-n", str(LAUNCHER_PATH)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_launcher_contract_is_synthesis_only():
    source = LAUNCHER_PATH.read_text(encoding="utf-8")

    assert "#SBATCH --no-requeue" in source
    assert "#SBATCH --requeue" not in source
    assert "train_paper01_diffusion_revision_production" not in source
    assert "run_governed_diffusion_training" not in source
    assert "sbatch " not in source
    assert "srun " not in source
    assert "DIFFUSION_PYTHON" in source
    assert "run_paper01_diffusion_production_synthesis.py" in source
    assert "RESOURCE_ENVELOPE=FROM_EXTERNAL_AUTHORIZATION" in source
    assert "AUTOMATIC_RETRY=DISABLED" in source


def test_launcher_requires_slurm_and_identity_environment():
    completed = subprocess.run(
        ["bash", str(LAUNCHER_PATH)],
        capture_output=True,
        text=True,
        check=False,
        env={},
    )
    assert completed.returncode != 0
    assert "Slurm allocation required" in completed.stderr


def test_cli_requires_synthesis_scope_arguments():
    cli = load_cli()

    with pytest.raises(SystemExit):
        cli.main([
            "--mode", "synthesis",
            "--dataset-id", DATASET,
            "--experimental-seed", "42",
            "--attempt", "1",
            "--expected-authorization-id", "P1R2026-DIFF-FIXTURE-A01",
            "--expected-synthesis-source-commit", SYNTHESIS_COMMIT,
            "--repository-root", str(REPO),
            "--checkpoint-dir", "/fixture/checkpoint",
            "--expected-checkpoint-manifest-sha256", "a" * 64,
            "--expected-checkpoint-state-sha256", "b" * 64,
            "--expected-checkpoint-training-source-commit", "c" * 40,
            "--expected-recipe-sha256", "d" * 64,
        ])


def test_scientific_gates_and_recipe_unchanged():
    from trustforge.models.paper01_revision_2026.diffusion_governed_training import (
        DIFFUSION_REAL_DATASET_TRAINING_ENABLED,
    )
    from trustforge.models.paper01_revision_2026.restricted_boltzmann_production_entry import (
        RBM_PRODUCTION_SAMPLING_ENABLED,
    )
    from trustforge.models.paper01_revision_2026.synthesis_linkage import (
        RECIPE_SHA256,
    )

    assert synthesis.DIFFUSION_PRODUCTION_SYNTHESIS_ENABLED is False
    assert DIFFUSION_REAL_DATASET_TRAINING_ENABLED is False
    assert RBM_PRODUCTION_SAMPLING_ENABLED is False
    assert load_synthesis_recipe(REPO).sha256 == RECIPE_SHA256
    assert RECIPE_SHA256 == (
        "98f9ef427e1b2a793636d448fe3f272893a24848e4b7257764964015e19987a9"
    )


def test_gpu_allocation_must_match_envelope():
    gpu_resources = dict(RESOURCES)
    gpu_resources["gpu_enabled"] = True
    gpu_allocation = dict(ALLOCATION)
    gpu_allocation["AllocTRES"] = "cpu=8,mem=32G,node=1,gres/gpu=1"
    gpu_allocation["Gres"] = "gpu:1"

    accepted = synthesis.validate_allocation(
        resources=gpu_resources, allocation=gpu_allocation,
    )
    assert accepted["JobId"] == "900001"

    with pytest.raises(
        synthesis.DiffusionProductionAuthorizationError, match="GPU",
    ):
        synthesis.validate_allocation(
            resources=gpu_resources, allocation=ALLOCATION,
        )
    with pytest.raises(
        synthesis.DiffusionProductionAuthorizationError, match="GPU",
    ):
        synthesis.validate_allocation(
            resources=RESOURCES, allocation=gpu_allocation,
        )


def test_resource_envelope_fields_are_required():
    with pytest.raises(
        synthesis.DiffusionProductionAuthorizationError,
        match="fields",
    ):
        synthesis._validate_resources({"partition": "talon"})
    with pytest.raises(
        synthesis.DiffusionProductionAuthorizationError,
        match="automatic retries",
    ):
        synthesis._validate_resources({
            **RESOURCES, "automatic_retries": 1,
        })


def test_equivalent_memory_unit_representation_accepted():
    resources = dict(RESOURCES)
    allocation = {
        "JobId": "900001",
        "Partition": "talon",
        "NumNodes": "1",
        "NumTasks": "1",
        "NumCPUs": "8",
        "AllocTRES": "cpu=8,mem=32768M,node=1,billing=8",
        "TimeLimit": "01:00:00",
        "Requeue": "0",
        "Restarts": "0",
        "Gres": "(null)",
    }
    accepted = synthesis.validate_allocation(
        resources=resources, allocation=allocation,
    )
    assert accepted["JobId"] == "900001"


def test_requested_only_tres_rejected():
    allocation = {
        "JobId": "900001",
        "Partition": "talon",
        "NumNodes": "1",
        "NumTasks": "1",
        "NumCPUs": "8",
        "ReqTRES": "cpu=8,mem=32G,node=1",
        "TimeLimit": "01:00:00",
        "Requeue": "0",
        "Restarts": "0",
        "Gres": "(null)",
    }
    with pytest.raises(
        synthesis.DiffusionProductionAuthorizationError,
        match="allocated-resource evidence",
    ):
        synthesis.validate_allocation(
            resources=RESOURCES, allocation=allocation,
        )


def test_legacy_actual_tres_accepted():
    allocation = {
        "JobId": "900001",
        "Partition": "talon",
        "NumNodes": "1",
        "NumTasks": "1",
        "NumCPUs": "8",
        "TRES": "cpu=8,mem=32G,node=1,billing=8",
        "TimeLimit": "01:00:00",
        "Requeue": "0",
        "Restarts": "0",
        "Gres": "(null)",
    }
    accepted = synthesis.validate_allocation(
        resources=RESOURCES, allocation=allocation,
    )
    assert accepted["JobId"] == "900001"


def test_actual_allocation_wins_over_conflicting_requested_resources():
    allocation = {
        "JobId": "900001",
        "Partition": "talon",
        "NumNodes": "1",
        "NumTasks": "1",
        "NumCPUs": "8",
        "AllocTRES": "cpu=8,mem=32G,node=1,billing=8",
        "ReqTRES": "cpu=8,mem=64G,node=1,billing=8",
        "TimeLimit": "01:00:00",
        "Requeue": "0",
        "Restarts": "0",
        "Gres": "(null)",
    }
    accepted = synthesis.validate_allocation(
        resources=RESOURCES, allocation=allocation,
    )
    assert accepted["JobId"] == "900001"


def test_malformed_actual_tres_not_replaced_by_requested_resources():
    allocation = {
        "JobId": "900001",
        "Partition": "talon",
        "NumNodes": "1",
        "NumTasks": "1",
        "NumCPUs": "8",
        "AllocTRES": "malformed",
        "ReqTRES": "cpu=8,mem=32G,node=1",
        "TimeLimit": "01:00:00",
        "Requeue": "0",
        "Restarts": "0",
        "Gres": "(null)",
    }
    with pytest.raises(
        synthesis.DiffusionProductionAuthorizationError,
        match="malformed",
    ):
        synthesis.validate_allocation(
            resources=RESOURCES, allocation=allocation,
        )


def test_missing_actual_allocation_evidence_rejected():
    allocation = {
        "JobId": "900001",
        "Partition": "talon",
        "NumNodes": "1",
        "NumTasks": "1",
        "NumCPUs": "8",
        "TimeLimit": "01:00:00",
        "Requeue": "0",
        "Restarts": "0",
        "Gres": "(null)",
    }
    with pytest.raises(
        synthesis.DiffusionProductionAuthorizationError,
        match="allocated-resource evidence",
    ):
        synthesis.validate_allocation(
            resources=RESOURCES, allocation=allocation,
        )


@pytest.mark.parametrize("memory", ["16G", "64G"])
def test_non_equivalent_memory_rejected(memory):
    allocation = {
        "JobId": "900001",
        "Partition": "talon",
        "NumNodes": "1",
        "NumTasks": "1",
        "NumCPUs": "8",
        "AllocTRES": f"cpu=8,mem={memory},node=1,billing=8",
        "TimeLimit": "01:00:00",
        "Requeue": "0",
        "Restarts": "0",
        "Gres": "(null)",
    }
    with pytest.raises(
        synthesis.DiffusionProductionAuthorizationError,
        match="memory",
    ):
        synthesis.validate_allocation(
            resources=RESOURCES, allocation=allocation,
        )
