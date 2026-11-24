"""Focused fixture-scope tests for RBM class-pool publication.

No real checkpoint, dataset, Talon access or Slurm submission is used. Every
execution object is constructed locally from synthetic fixture images.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

from trustforge.models.paper01_revision_2026 import (
    restricted_boltzmann_class_pool as pool,
    restricted_boltzmann_production_entry as entry,
    restricted_boltzmann_production_sampling as sampler,
)
from trustforge.provenance import canonical_json_bytes


REPO_ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = REPO_ROOT / "scripts" / "run_paper01_rbm_production_sampling.py"
LAUNCHER_PATH = (
    REPO_ROOT
    / "slurm"
    / "paper01_revision_2026"
    / "rbm_production_sampling.slurm"
)
DIFFUSION_GATE_PATH = (
    REPO_ROOT
    / "src"
    / "trustforge"
    / "models"
    / "paper01_revision_2026"
    / "diffusion_governed_training.py"
)

COMMIT = "a" * 40
STATE_SHA = "c" * 64
METADATA_SHA = "e" * 64
SEED = 42


def load_cli():
    spec = importlib.util.spec_from_file_location(
        "paper01_rbm_production_cli",
        CLI_PATH,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load RBM production CLI")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _images(class_id=0):
    values = np.linspace(
        0.0,
        1.0,
        num=pool.SAMPLE_COUNT * 40 * 40 * 3,
        dtype=np.float32,
    )
    return values.reshape((pool.SAMPLE_COUNT, 40, 40, 3))


def make_execution(
    tmp_path,
    *,
    class_id=0,
    dataset_id="malex",
    attempt=1,
    images=None,
    authorization_id=None,
    slurm_job_id=None,
    checkpoint_dir=None,
    state_sha256=STATE_SHA,
    metadata_sha256=METADATA_SHA,
    sample_start=0,
    sample_stop=pool.SAMPLE_COUNT,
):
    request = entry.RBMProductionSamplingRequest(
        dataset_id=dataset_id,
        experimental_seed=SEED,
        attempt=attempt,
        class_id=class_id,
        sample_start=sample_start,
        sample_stop=sample_stop,
        expected_source_commit=COMMIT,
        repository_root=REPO_ROOT,
        checkpoint_directory=(
            Path(checkpoint_dir)
            if checkpoint_dir is not None
            else tmp_path / "checkpoint"
        ),
        expected_checkpoint_state_sha256=state_sha256,
    )

    mapping = entry.authoritative_class_mapping(
        dataset_id=dataset_id,
        repository_root=REPO_ROOT,
    )
    policy = entry.policy_for_request(request, mapping)

    grant = entry.RBMProductionSamplingGrant(
        authorization_id=(
            authorization_id
            if authorization_id is not None
            else f"P1R2026-RBM-FIXTURE-C{class_id}"
        ),
        dataset_id=dataset_id,
        experimental_seed=SEED,
        attempt=attempt,
        class_id=class_id,
        sample_start=sample_start,
        sample_stop=sample_stop,
        source_commit=COMMIT,
        checkpoint_directory=request.checkpoint_directory,
        checkpoint_state_sha256=state_sha256,
        policy_digest=policy.digest(),
        resources=dict(entry.DEFAULT_RESOURCES),
    )

    provenance = sampler.RBMProductionSamplingProvenance(
        dataset_id=dataset_id,
        experimental_seed=SEED,
        class_id=class_id,
        class_label=mapping[class_id],
        sample_indices=tuple(range(sample_start, sample_stop)),
        sample_count=sample_stop - sample_start,
        transition_count=sampler.PRODUCTION_TRANSITION_COUNT,
        burn_in=sampler.PRODUCTION_BURN_IN,
        initialization=policy.initialization,
        class_conditioning=policy.class_conditioning,
        chain_semantics=policy.chain_semantics,
        internal_dtype=policy.internal_dtype,
        canonical_shape=policy.canonical_shape,
        canonical_dtype=policy.canonical_dtype,
        checkpoint_state_sha256=state_sha256,
        policy_digest=policy.digest(),
    )

    result = sampler.RBMProductionSamplingResult(
        images=images if images is not None else _images(class_id),
        provenance=provenance,
    )

    return entry.RBMProductionSamplingExecution(
        request=request,
        source_binding={
            "requested_source_commit": COMMIT,
            "resolved_source_commit": COMMIT,
            "repository_root": str(REPO_ROOT),
            "module_paths": {
                "restricted_boltzmann_production_entry":
                    "src/trustforge/models/paper01_revision_2026/"
                    "restricted_boltzmann_production_entry.py",
            },
        },
        grant=grant,
        checkpoint_metadata_sha256=metadata_sha256,
        result=result,
        slurm_job_id=(
            slurm_job_id if slurm_job_id is not None else "317232"
        ),
    )


def _rewrite_manifest(manifest_path, mutate):
    manifest = json.loads(Path(manifest_path).read_bytes())
    mutate(manifest)
    Path(manifest_path).write_bytes(canonical_json_bytes(manifest) + b"\n")


@pytest.fixture(scope="session")
def complete_pool(tmp_path_factory):
    root = tmp_path_factory.mktemp("rbm-complete-pool")
    destination = root / "class0"
    report = pool.publish_rbm_class_pool(
        execution=make_execution(root),
        destination=destination,
        expected_checkpoint_metadata_sha256=METADATA_SHA,
        expected_checkpoint_state_sha256=STATE_SHA,
    )
    return destination, report


# Successful publication and verification


def test_publish_creates_complete_verified_pool(complete_pool):
    destination, report = complete_pool

    assert report["schema"] == pool.SCHEMA
    assert report["dataset_id"] == "malex"
    assert report["family"] == pool.FAMILY
    assert report["experimental_seed"] == SEED
    assert report["class_id"] == 0
    assert report["class_label"] == "benign"
    assert report["sample_count"] == 2000
    assert report["batch_count"] == 250
    assert report["checkpoint_metadata_file_sha256"] == METADATA_SHA
    assert report["checkpoint_state_file_sha256"] == STATE_SHA
    assert len(report["batch_filenames"]) == 250

    assert (destination / "manifest.json").is_file()
    assert len(list(destination.glob("*.npy"))) == 250

    first = np.load(destination / "class0-b000.npy", allow_pickle=False)
    assert first.dtype == np.float32
    assert first.shape == (8, 40, 40, 3)
    assert np.array_equal(first, _images(0)[0:8])

    last = np.load(destination / "class0-b249.npy", allow_pickle=False)
    assert np.array_equal(last, _images(0)[1992:2000])

    independent = pool.verify_rbm_class_pool(
        destination / "manifest.json",
        expected_dataset_id="malex",
        expected_seed=SEED,
        expected_class_id=0,
        expected_class_label="benign",
        expected_checkpoint_metadata_sha256=METADATA_SHA,
        expected_checkpoint_state_sha256=STATE_SHA,
        expected_policy_digest=report["sampling_policy_digest"],
    )
    assert independent["manifest_sha256"] == report["manifest_sha256"]


def test_publish_refuses_existing_destination(complete_pool, tmp_path):
    destination, _ = complete_pool

    with pytest.raises(FileExistsError):
        pool.publish_rbm_class_pool(
            execution=make_execution(tmp_path),
            destination=destination,
        )


def test_dataset_pools_accept_two_classes_with_distinct_lineage(tmp_path):
    class0 = tmp_path / "class0"
    class1 = tmp_path / "class1"

    pool.publish_rbm_class_pool(
        execution=make_execution(
            tmp_path,
            class_id=0,
            attempt=1,
            authorization_id="P1R2026-RBM-MALEX-C0-A01",
            slurm_job_id="317300",
        ),
        destination=class0,
        expected_checkpoint_metadata_sha256=METADATA_SHA,
        expected_checkpoint_state_sha256=STATE_SHA,
    )
    pool.publish_rbm_class_pool(
        execution=make_execution(
            tmp_path,
            class_id=1,
            attempt=2,
            authorization_id="P1R2026-RBM-MALEX-C1-A02",
            slurm_job_id="317301",
        ),
        destination=class1,
        expected_checkpoint_metadata_sha256=METADATA_SHA,
        expected_checkpoint_state_sha256=STATE_SHA,
    )

    report = pool.verify_rbm_dataset_pools(
        [class0 / "manifest.json", class1 / "manifest.json"],
        expected_dataset_id="malex",
        expected_seed=SEED,
        expected_class_mapping=("benign", "malware"),
        expected_checkpoint_metadata_sha256=METADATA_SHA,
        expected_checkpoint_state_sha256=STATE_SHA,
    )

    assert report["class_ids"] == [0, 1]
    assert report["class_labels"] == ["benign", "malware"]
    assert report["sample_count_per_class"] == 2000
    assert report["total_samples"] == 4000
    assert report["class_count"] == 2

    authorizations = {
        manifest["authorization_id"] for manifest in report["manifests"]
    }
    jobs = {manifest["slurm_job_id"] for manifest in report["manifests"]}
    assert authorizations == {
        "P1R2026-RBM-MALEX-C0-A01",
        "P1R2026-RBM-MALEX-C1-A02",
    }
    assert jobs == {"317300", "317301"}

    # Manifest order on disk must not matter.
    reversed_report = pool.verify_rbm_dataset_pools(
        [class1 / "manifest.json", class0 / "manifest.json"],
    )
    assert reversed_report["class_ids"] == [0, 1]


# Rejection behavior


def test_dataset_pools_reject_missing_class(complete_pool):
    destination, _ = complete_pool

    with pytest.raises(ValueError, match="incomplete or duplicated"):
        pool.verify_rbm_dataset_pools([destination / "manifest.json"])


def test_dataset_pools_reject_duplicate_class(complete_pool):
    destination, _ = complete_pool

    with pytest.raises(ValueError, match="incomplete or duplicated"):
        pool.verify_rbm_dataset_pools(
            [destination / "manifest.json", destination / "manifest.json"]
        )


def test_dataset_pools_reject_mixed_checkpoint(tmp_path):
    class0 = tmp_path / "class0"
    class1 = tmp_path / "class1"

    pool.publish_rbm_class_pool(
        execution=make_execution(tmp_path, class_id=0),
        destination=class0,
    )
    pool.publish_rbm_class_pool(
        execution=make_execution(
            tmp_path,
            class_id=1,
            metadata_sha256="f" * 64,
        ),
        destination=class1,
    )

    with pytest.raises(ValueError, match="share one checkpoint"):
        pool.verify_rbm_dataset_pools(
            [class0 / "manifest.json", class1 / "manifest.json"]
        )


@pytest.mark.parametrize(
    "expectation",
    [
        {"expected_dataset_id": "cicmaldroid2020"},
        {"expected_seed": 43},
        {"expected_class_id": 1},
        {"expected_class_label": "malware"},
        {"expected_checkpoint_metadata_sha256": "f" * 64},
        {"expected_checkpoint_state_sha256": "f" * 64},
        {"expected_policy_digest": "f" * 64},
    ],
)
def test_verification_rejects_wrong_expectation(complete_pool, expectation):
    destination, _ = complete_pool

    with pytest.raises(ValueError, match="expectation mismatch"):
        pool.verify_rbm_class_pool(
            destination / "manifest.json",
            **expectation,
        )


def test_verification_rejects_corrupted_batch(complete_pool, tmp_path):
    source, _ = complete_pool
    destination = tmp_path / "copy"
    shutil.copytree(source, destination)

    batch = destination / "class0-b042.npy"
    batch.write_bytes(batch.read_bytes() + b"\x00")

    with pytest.raises(ValueError, match="digest mismatch"):
        pool.verify_rbm_class_pool(destination / "manifest.json")


def test_verification_rejects_missing_batch(complete_pool, tmp_path):
    source, _ = complete_pool
    destination = tmp_path / "copy"
    shutil.copytree(source, destination)

    (destination / "class0-b007.npy").unlink()

    with pytest.raises(ValueError, match="regular file"):
        pool.verify_rbm_class_pool(destination / "manifest.json")


def test_verification_rejects_duplicated_batch_record(
    complete_pool, tmp_path,
):
    source, _ = complete_pool
    destination = tmp_path / "copy"
    shutil.copytree(source, destination)

    def mutate(manifest):
        manifest["batches"][1]["filename"] = manifest["batches"][0][
            "filename"
        ]

    _rewrite_manifest(destination / "manifest.json", mutate)

    with pytest.raises(ValueError, match="invalid RBM batch record"):
        pool.verify_rbm_class_pool(destination / "manifest.json")


def test_verification_rejects_missing_batch_record(complete_pool, tmp_path):
    source, _ = complete_pool
    destination = tmp_path / "copy"
    shutil.copytree(source, destination)

    _rewrite_manifest(
        destination / "manifest.json",
        lambda manifest: manifest["batches"].pop(),
    )

    with pytest.raises(ValueError, match="250 ordered batches"):
        pool.verify_rbm_class_pool(destination / "manifest.json")


def test_verification_rejects_incomplete_publication(complete_pool, tmp_path):
    source, _ = complete_pool
    destination = tmp_path / "copy"
    shutil.copytree(source, destination)

    (destination / "manifest.json").unlink()

    with pytest.raises(ValueError, match="regular file"):
        pool.verify_rbm_class_pool(destination / "manifest.json")


def test_verification_rejects_tampered_label(complete_pool, tmp_path):
    source, _ = complete_pool
    destination = tmp_path / "copy"
    shutil.copytree(source, destination)

    _rewrite_manifest(
        destination / "manifest.json",
        lambda manifest: manifest.__setitem__("class_label", "malware"),
    )

    with pytest.raises(ValueError, match="authoritative"):
        pool.verify_rbm_class_pool(
            destination / "manifest.json",
            expected_class_label="benign",
        )


def test_verification_rejects_tampered_label_without_expectation(
    complete_pool, tmp_path,
):
    source, _ = complete_pool
    destination = tmp_path / "copy"
    shutil.copytree(source, destination)

    _rewrite_manifest(
        destination / "manifest.json",
        lambda manifest: manifest.__setitem__("class_label", "malware"),
    )

    with pytest.raises(ValueError, match="authoritative"):
        pool.verify_rbm_class_pool(destination / "manifest.json")


def test_dataset_verification_rejects_tampered_label(
    complete_pool, tmp_path,
):
    source, _ = complete_pool
    destination = tmp_path / "copy"
    shutil.copytree(source, destination)

    _rewrite_manifest(
        destination / "manifest.json",
        lambda manifest: manifest.__setitem__("class_label", "malware"),
    )

    with pytest.raises(ValueError, match="authoritative"):
        pool.verify_rbm_dataset_pools(
            [destination / "manifest.json"]
        )


def test_verification_rejects_incorrect_policy_digest_without_expectation(
    complete_pool, tmp_path,
):
    source, _ = complete_pool
    destination = tmp_path / "copy"
    shutil.copytree(source, destination)

    _rewrite_manifest(
        destination / "manifest.json",
        lambda manifest: manifest.__setitem__(
            "sampling_policy_digest", "f" * 64,
        ),
    )

    with pytest.raises(ValueError, match="reconstructed governed policy"):
        pool.verify_rbm_class_pool(destination / "manifest.json")


def test_verification_rejects_inconsistent_run_identity(
    complete_pool, tmp_path,
):
    source, _ = complete_pool
    destination = tmp_path / "copy"
    shutil.copytree(source, destination)

    _rewrite_manifest(
        destination / "manifest.json",
        lambda manifest: manifest.__setitem__(
            "run_identity", "commit0000000_attempt01",
        ),
    )

    with pytest.raises(ValueError, match="run identity"):
        pool.verify_rbm_class_pool(destination / "manifest.json")


def test_verification_rejects_tampered_policy(complete_pool, tmp_path):
    source, _ = complete_pool
    destination = tmp_path / "copy"
    shutil.copytree(source, destination)

    _rewrite_manifest(
        destination / "manifest.json",
        lambda manifest: manifest["sampling_provenance"].__setitem__(
            "burn_in", 1
        ),
    )

    with pytest.raises(ValueError, match="sampling policy mismatch"):
        pool.verify_rbm_class_pool(destination / "manifest.json")


def test_verification_rejects_noncanonical_manifest(
    complete_pool, tmp_path,
):
    source, _ = complete_pool
    destination = tmp_path / "copy"
    shutil.copytree(source, destination)

    manifest = json.loads(
        (destination / "manifest.json").read_bytes()
    )
    (destination / "manifest.json").write_bytes(
        json.dumps(manifest, indent=2).encode("utf-8")
    )

    with pytest.raises(ValueError, match="noncanonical"):
        pool.verify_rbm_class_pool(destination / "manifest.json")


def test_publish_rejects_incomplete_sample_range(tmp_path):
    with pytest.raises(ValueError, match="0..1999"):
        pool.publish_rbm_class_pool(
            execution=make_execution(tmp_path, sample_stop=1999),
            destination=tmp_path / "pool",
        )

    assert not (tmp_path / "pool").exists()


def test_publish_rejects_wrong_image_count(tmp_path):
    images = _images(0)[:1999]

    with pytest.raises(ValueError, match="2,000 canonical images"):
        pool.publish_rbm_class_pool(
            execution=make_execution(tmp_path, images=images),
            destination=tmp_path / "pool",
        )


def test_publish_rejects_wrong_image_shape(tmp_path):
    images = np.zeros((2000, 40, 40, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="2,000 canonical images"):
        pool.publish_rbm_class_pool(
            execution=make_execution(tmp_path, images=images),
            destination=tmp_path / "pool",
        )


@pytest.mark.parametrize(
    "mutate, error",
    [
        (
            lambda images: images.astype(np.float64),
            TypeError,
        ),
        (
            lambda images: np.where(
                np.arange(2000).reshape(-1, 1, 1, 1) == 0,
                np.float64(2.0),
                images,
            ).astype(np.float32),
            ValueError,
        ),
        (
            lambda images: np.where(
                np.arange(2000).reshape(-1, 1, 1, 1) == 0,
                np.float32(np.nan),
                images,
            ).astype(np.float32),
            ValueError,
        ),
    ],
)
def test_publish_rejects_invalid_images(tmp_path, mutate, error):
    images = mutate(_images(0))

    with pytest.raises(error):
        pool.publish_rbm_class_pool(
            execution=make_execution(tmp_path, images=images),
            destination=tmp_path / "pool",
        )


def test_publish_rejects_mismatched_checkpoint_expectation(tmp_path):
    with pytest.raises(ValueError, match="checkpoint metadata expectation"):
        pool.publish_rbm_class_pool(
            execution=make_execution(tmp_path),
            destination=tmp_path / "pool",
            expected_checkpoint_metadata_sha256="f" * 64,
        )


def test_publish_rejects_mismatched_policy_expectation(tmp_path):
    with pytest.raises(ValueError, match="sampling policy expectation"):
        pool.publish_rbm_class_pool(
            execution=make_execution(tmp_path),
            destination=tmp_path / "pool",
            expected_policy_digest="f" * 64,
        )


def test_publish_requires_governed_execution_contract(tmp_path):
    from types import SimpleNamespace

    with pytest.raises(TypeError, match="RBMProductionSamplingExecution"):
        pool.publish_rbm_class_pool(
            execution=SimpleNamespace(images=np.zeros((2000, 40, 40, 3))),
            destination=tmp_path / "pool",
        )


def test_production_publication_fails_closed_without_authorization(
    tmp_path,
):
    assert entry.RBM_PRODUCTION_SAMPLING_ENABLED is False

    request = entry.RBMProductionSamplingRequest(
        dataset_id="malex",
        experimental_seed=SEED,
        attempt=1,
        class_id=0,
        sample_start=0,
        sample_stop=2000,
        expected_source_commit=COMMIT,
        repository_root=REPO_ROOT,
        checkpoint_directory=tmp_path / "checkpoint",
        expected_checkpoint_state_sha256=STATE_SHA,
    )

    with pytest.raises(entry.RBMProductionAuthorizationError):
        entry.execute_rbm_production_sampling(request)


# CLI wiring


def test_cli_publishes_only_after_governed_execution(
    tmp_path, monkeypatch, capsys,
):
    cli = load_cli()
    execution = make_execution(
        tmp_path,
        class_id=0,
        checkpoint_dir=tmp_path / "checkpoint",
    )
    calls = []

    def fake_execute(request):
        calls.append(request)
        return execution

    monkeypatch.setattr(
        entry, "execute_rbm_production_sampling", fake_execute,
    )

    destination = tmp_path / "published"

    exit_code = cli.main(
        [
            "--dataset-id",
            "malex",
            "--experimental-seed",
            "42",
            "--attempt",
            "1",
            "--class-id",
            "0",
            "--sample-start",
            "0",
            "--sample-stop",
            "2000",
            "--expected-source-commit",
            COMMIT,
            "--repository-root",
            str(REPO_ROOT),
            "--checkpoint-dir",
            str(tmp_path / "checkpoint"),
            "--expected-checkpoint-state-sha256",
            STATE_SHA,
            "--publish-pool-dir",
            str(destination),
            "--expected-checkpoint-metadata-sha256",
            METADATA_SHA,
            "--expected-policy-digest",
            execution.grant.policy_digest,
        ]
    )

    assert exit_code == 0
    assert len(calls) == 1
    assert (destination / "manifest.json").is_file()

    output = capsys.readouterr().out
    assert "RBM_PRODUCTION_SAMPLING_RESULT" in output
    assert "RBM_CLASS_POOL_PUBLICATION_RESULT" in output


def test_cli_publication_requires_explicit_expectations(tmp_path):
    cli = load_cli()

    with pytest.raises(SystemExit):
        cli.main(
            [
                "--dataset-id",
                "malex",
                "--experimental-seed",
                "42",
                "--attempt",
                "1",
                "--class-id",
                "0",
                "--sample-start",
                "0",
                "--sample-stop",
                "2000",
                "--expected-source-commit",
                COMMIT,
                "--repository-root",
                str(REPO_ROOT),
                "--checkpoint-dir",
                str(tmp_path / "checkpoint"),
                "--expected-checkpoint-state-sha256",
                STATE_SHA,
                "--publish-pool-dir",
                str(tmp_path / "published"),
            ]
        )


# Scientific and activation preservation


def test_sampling_constants_unchanged():
    assert sampler.PRODUCTION_TRANSITION_COUNT == 10
    assert sampler.PRODUCTION_BURN_IN == 0
    assert sampler.PRIMARY_SAMPLES_PER_CLASS == 2000
    assert sampler.CANONICAL_SHAPE == (40, 40, 3)


def test_activation_gates_remain_disabled():
    assert entry.RBM_PRODUCTION_SAMPLING_ENABLED is False
    assert (
        "DIFFUSION_REAL_DATASET_TRAINING_ENABLED = False"
        in DIFFUSION_GATE_PATH.read_text(encoding="utf-8")
    )
    assert "POOL_PUBLICATION=DISABLED" in LAUNCHER_PATH.read_text(
        encoding="utf-8"
    )
