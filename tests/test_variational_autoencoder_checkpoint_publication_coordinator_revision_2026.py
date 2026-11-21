"""Focused integration tests for the VAE publication coordinator.

Synthetic fixtures only. The tiny training run produces disposable VAE
weights; fixture provenance is not real dataset evidence. No real data,
production checkpoint or synthesis is used.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip("tensorflow")

import tensorflow as tf

from trustforge.models.paper01_revision_2026 import (
    variational_autoencoder_checkpoint_committed_generation as committed,
    variational_autoencoder_checkpoint_loader as loader,
    variational_autoencoder_checkpoint_publication_coordinator as coordinator,
    variational_autoencoder_training_runner as runner,
)
from trustforge.models.paper01_revision_2026.variational_autoencoder_checkpoint_commit_publisher import (
    IndeterminateVAECommitError,
)
from tests.test_variational_autoencoder_checkpoint_provenance_revision_2026 import (
    evidence_for,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = (
    REPO_ROOT
    / "scripts"
    / "train_paper01_vae_revision_production.py"
)
SOURCE_COMMIT = "9eb426b551f2bd4f07eac6eeac60027e2b9f64ca"
DATASET_ID = "cicmaldroid2020"
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
    "validation":
        "full_governed_validation_after_each_epoch",
    "selection_metric": "validation_total_loss",
    "selection_direction": "minimize",
    "selection_tie_break": "earliest_epoch",
    "checkpoint_roles": ["best", "last"],
    "test_split_usage": "prohibited",
}


class SyntheticAdapter:
    """Small deterministic adapter for synthetic training fixtures."""

    def __init__(self):
        rng = np.random.default_rng(20261120)
        counts = {
            "train": TRAIN_SIZE,
            "validation": VALIDATION_SIZE,
        }
        self.images = {
            split: rng.uniform(
                0.1,
                0.9,
                size=(count, 40, 40, 3),
            ).astype(np.float32)
            for split, count in counts.items()
        }
        self.labels = {
            split: np.eye(
                CLASS_COUNT,
                dtype=np.float32,
            )[np.arange(count) % CLASS_COUNT]
            for split, count in counts.items()
        }

    def split_size(self, split):
        return len(self.images[split])

    def canonical_batch(self, split, logical_positions):
        positions = np.asarray(
            logical_positions,
            dtype=np.int64,
        )
        return (
            self.images[split][positions].copy(),
            self.labels[split][positions].copy(),
        )


@pytest.fixture(scope="module")
def governed_result():
    tf.keras.backend.clear_session()

    trained = runner.run_synthetic_vae_training(
        adapter=SyntheticAdapter(),
        num_classes=CLASS_COUNT,
        experimental_seed=SEED,
        epochs=2,
        batch_size=64,
        learning_rate=0.001,
        development_mode="synthetic_only",
    )

    return {
        "dataset_id": DATASET_ID,
        "experimental_seed": SEED,
        "class_count": CLASS_COUNT,
        "dataset_provenance": evidence_for(DATASET_ID),
        "training_result": trained,
        "checkpoint_status": "NOT_PREPARED",
        "publication_status": "NOT_PUBLISHED",
    }


def publish(governed_result, publication_root, **overrides):
    values = {
        "governed_result": governed_result,
        "class_mapping": CLASS_MAPPING,
        "effective_configuration": EFFECTIVE_CONFIGURATION,
        "publication_root": publication_root,
        "source_commit": SOURCE_COMMIT,
        "attempt": 1,
        "repository_root": REPO_ROOT,
    }
    values.update(overrides)
    return coordinator.publish_governed_vae_checkpoints(**values)


def test_publish_best_and_last_and_restore_best(
    tmp_path,
    monkeypatch,
    governed_result,
):
    monkeypatch.setenv("SLURM_JOB_ID", "synthetic-test-only")

    root = tmp_path / "publication"
    publication = publish(governed_result, root)

    assert publication["publication_status"] == "PUBLISHED"
    assert publication["crash_durability_verified"] is False
    assert publication["observation_version"] == (
        coordinator.VAE_OBSERVATION_VERSION
    )
    assert coordinator.VAE_OBSERVATION_VERSION == (
        "gaussian_fixed_variance_1p0_v1"
    )
    assert publication["source_commit"] == SOURCE_COMMIT
    assert publication["attempt"] == 1
    assert publication["run_identity"] == (
        "commit9eb426b_attempt01"
    )

    roles = publication["roles"]
    assert set(roles) == {"best", "last"}

    trained = governed_result["training_result"]
    best_epoch = trained["epoch_selection"]["best_epoch"]

    for role_name in ("best", "last"):
        role = roles[role_name]
        metadata = role["metadata"]

        assert len(role["generation_id"]) == 32
        assert Path(role["generation_directory"]).is_dir()
        assert (
            root
            / "commits"
            / role["commit_filename"]
        ).is_file()
        assert role["manifest_sha256"] == (
            role["commit_record"]["manifest_sha256"]
        )

        assert metadata["observation_version"] == (
            "gaussian_fixed_variance_1p0_v1"
        )
        assert metadata["architecture_version"] == (
            "paper01_revision2026_conditional_vae_arch_v1"
        )
        assert metadata["checkpoint_role"] == role_name
        assert metadata["dataset_id"] == DATASET_ID
        assert metadata["experimental_seed"] == SEED
        assert metadata["class_mapping"] == list(CLASS_MAPPING)
        assert metadata["selection_policy_id"] == (
            coordinator.SELECTION_POLICY_ID
        )
        assert metadata["validation_seed_policy_id"] == (
            coordinator.VALIDATION_SEED_POLICY_ID
        )

        configuration = metadata["effective_configuration"]
        assert configuration["source_commit"] == SOURCE_COMMIT
        assert configuration["attempt"] == 1
        assert configuration["run_identity"] == (
            "commit9eb426b_attempt01"
        )
        assert configuration["batch_size"] == 64
        assert configuration["learning_rate"] == 0.001
        assert configuration["epochs"] == 2

    assert roles["best"]["metadata"]["completed_epochs"] == (
        best_epoch + 1
    )
    assert roles["best"]["metadata"]["completed_updates"] == (
        trained["history"][best_epoch]["completed_updates"]
    )
    assert roles["last"]["metadata"]["completed_epochs"] == 2
    assert roles["last"]["metadata"]["completed_updates"] == (
        trained["history"][1]["completed_updates"]
    )

    identity = publication["checkpoint_identity"]
    assert identity == {
        "dataset_id": DATASET_ID,
        "experimental_seed": SEED,
        "class_mapping": list(CLASS_MAPPING),
        "checkpoint_role": "best",
        "checkpoint_sha256": roles["best"]["manifest_sha256"],
        "source_commit": SOURCE_COMMIT,
    }
    assert publication["best_loader_directory"] == (
        roles["best"]["generation_directory"]
    )

    verified = committed.verify_committed_vae_generation(
        root,
        generation_id=roles["best"]["generation_id"],
        requested_dataset_id=DATASET_ID,
        requested_experimental_seed=SEED,
        requested_class_mapping=CLASS_MAPPING,
    )
    assert verified["metadata"] == roles["best"]["metadata"]

    loaded = loader.load_governed_vae_checkpoint(
        publication["best_loader_directory"],
        requested_dataset_id=DATASET_ID,
        requested_experimental_seed=SEED,
        requested_class_mapping=CLASS_MAPPING,
    )

    assert loaded["metadata"] == roles["best"]["metadata"]
    assert loaded["metadata"]["checkpoint_role"] == "best"

    best_weights = trained["best_weights"]

    for actual, expected in zip(
        loaded["encoder"].get_weights(),
        best_weights["encoder_weights"],
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)

    for actual, expected in zip(
        loaded["decoder"].get_weights(),
        best_weights["decoder_weights"],
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)


def test_duplicate_publication_is_rejected(
    tmp_path,
    governed_result,
):
    root = tmp_path / "publication"
    first = publish(governed_result, root)

    commit_path = (
        root
        / "commits"
        / first["roles"]["best"]["commit_filename"]
    )
    before = commit_path.read_bytes()

    with pytest.raises(FileExistsError):
        publish(governed_result, root)

    assert commit_path.read_bytes() == before


@pytest.mark.parametrize(
    "failure, expected_exception",
    [
        ("runtime", RuntimeError),
        ("indeterminate", IndeterminateVAECommitError),
    ],
)
def test_incomplete_publication_never_reports_success(
    tmp_path,
    governed_result,
    monkeypatch,
    failure,
    expected_exception,
):
    root = tmp_path / ("publication-" + failure)

    def fail_publish(*args, **kwargs):
        if failure == "indeterminate":
            raise IndeterminateVAECommitError(
                "synthetic indeterminate commit"
            )
        raise RuntimeError("synthetic commit failure")

    monkeypatch.setattr(
        coordinator,
        "publish_candidate_vae_commit",
        fail_publish,
    )

    with pytest.raises(expected_exception):
        publish(governed_result, root)

    assert (root / "generations").is_dir()
    assert list((root / "generations").iterdir())
    assert not list(
        (root / "commits").glob("*.commit.json")
    )


@pytest.mark.parametrize(
    "case",
    [
        "dataset",
        "seed",
        "class_mapping",
        "source_commit",
        "attempt",
        "configuration",
        "status",
    ],
)
def test_invalid_identity_is_rejected(
    tmp_path,
    governed_result,
    case,
):
    root = tmp_path / ("invalid-" + case)
    result = dict(governed_result)
    overrides = {}

    if case == "dataset":
        result["dataset_id"] = "unknown"
    elif case == "seed":
        result["experimental_seed"] = 41
    elif case == "class_mapping":
        overrides["class_mapping"] = CLASS_MAPPING[:-1]
    elif case == "source_commit":
        overrides["source_commit"] = "not-a-commit"
    elif case == "attempt":
        overrides["attempt"] = 0
    elif case == "configuration":
        overrides["effective_configuration"] = {"epochs": 2}
    elif case == "status":
        result["publication_status"] = "PUBLISHED"

    with pytest.raises((ValueError, TypeError)):
        publish(
            result,
            root,
            **overrides,
        )

    assert not root.exists()


def test_unsafe_publication_destinations_are_rejected(
    tmp_path,
    governed_result,
):
    existing = tmp_path / "existing"
    existing.mkdir()

    with pytest.raises(FileExistsError):
        publish(governed_result, existing)

    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)

    with pytest.raises(ValueError):
        publish(governed_result, link)

    inside = REPO_ROOT / "vae-publication-preflight-test"
    assert not inside.exists()

    with pytest.raises(ValueError):
        publish(governed_result, inside)

    assert not inside.exists()


def _cli_base_args():
    return [
        "--dataset-id",
        DATASET_ID,
        "--experimental-seed",
        "42",
        "--attempt",
        "1",
        "--expected-source-commit",
        "a" * 40,
        "--repository-root",
        str(REPO_ROOT),
        "--cic-source-csv",
        "source.csv",
        "--cic-feature-columns",
        "features.json",
    ]


def _load_cli():
    spec = importlib.util.spec_from_file_location(
        "paper01_vae_publication_cli",
        CLI_PATH,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load VAE production CLI")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cli_publication_options_are_default_off():
    cli = _load_cli()
    base = [
        "--dataset-id",
        DATASET_ID,
        "--experimental-seed",
        "42",
        "--attempt",
        "1",
        "--expected-source-commit",
        "a" * 40,
        "--repository-root",
        str(REPO_ROOT),
        "--cic-source-csv",
        "source.csv",
        "--cic-feature-columns",
        "features.json",
    ]

    parser = cli.make_parser()

    default = parser.parse_args(base)
    assert default.publish_checkpoints is False
    assert default.publication_root is None

    publish_only = parser.parse_args(
        base + ["--publish-checkpoints"]
    )
    assert publish_only.publish_checkpoints is True
    assert publish_only.publication_root is None

    with pytest.raises(ValueError, match="publication-root"):
        cli.run(publish_only)

    root_only = parser.parse_args(
        base
        + [
            "--publication-root",
            str(REPO_ROOT / "publication-test-root"),
        ]
    )
    assert root_only.publish_checkpoints is False

    with pytest.raises(
        ValueError,
        match="requires --publish-checkpoints",
    ):
        cli.run(root_only)


def test_production_publication_request_denied_without_permit(
    tmp_path,
    monkeypatch,
):
    from trustforge.models.paper01_revision_2026 import (
        variational_autoencoder_checkpoint_publication_permit as permit_module,
    )

    cli = _load_cli()
    args = cli.make_parser().parse_args(
        _cli_base_args()
        + [
            "--publish-checkpoints",
            "--publication-root",
            str(tmp_path / "publication"),
        ]
    )

    calls = []

    def unexpected(name):
        def stop(*args, **kwargs):
            calls.append(name)
            raise AssertionError(name + " must not run")

        return stop

    monkeypatch.setattr(
        cli,
        "VAE_AUTHORITATIVE_CHECKPOINT_PUBLICATION_AUTHORIZED",
        True,
    )
    monkeypatch.setattr(
        permit_module,
        "PERMIT_PATH",
        tmp_path / "missing-permit.json",
    )
    monkeypatch.setattr(
        cli,
        "require_environment",
        unexpected("environment"),
    )
    monkeypatch.setattr(
        cli,
        "build_adapter",
        unexpected("adapter"),
    )
    monkeypatch.setattr(
        cli,
        "_load_governed_module",
        unexpected("governed"),
    )

    with pytest.raises(
        PermissionError,
        match="permit is absent",
    ):
        cli.run(args)

    assert calls == []
    assert not (tmp_path / "publication").exists()


def test_training_only_default_does_not_require_publication_authorization(
    monkeypatch,
):
    cli = _load_cli()
    args = cli.make_parser().parse_args(_cli_base_args())
    calls = []

    monkeypatch.setattr(
        cli,
        "require_vae_checkpoint_publication_authorization",
        lambda: calls.append("publication"),
    )

    def stop_at_environment(args):
        raise RuntimeError("environment reached")

    monkeypatch.setattr(
        cli,
        "require_environment",
        stop_at_environment,
    )

    with pytest.raises(RuntimeError, match="environment reached"):
        cli.run(args)

    assert calls == []


def test_source_activation_enables_only_the_permit_route():
    cli = _load_cli()

    assert (
        cli.VAE_AUTHORITATIVE_CHECKPOINT_PUBLICATION_AUTHORIZED
        is True
    )

    # The source-level activation reaches the route; it is not itself a
    # publication grant. The permit-dependent rejection is covered by
    # test_production_publication_request_denied_without_permit.
    cli.require_vae_checkpoint_publication_authorization()
