"""Fail-closed GAN checkpoint v3 restoration tests."""

import json
import runpy
from pathlib import Path

import pytest

from trustforge.models.paper01_revision_2026 import (
    gan_checkpoint_v3_persistence as publication,
    gan_checkpoint_v3_restoration as restoration,
    gan_model_factory,
)
from trustforge.provenance import canonical_json_bytes

helpers = runpy.run_path(
    "tests/test_paper01_gan_checkpoint_v3_real_integration.py"
)


class RestoredModel:
    trainable_variables = []


class RestoredOptimizer:
    def __init__(self, value):
        self.iterations = helpers["FakeIteration"](value)

    def build(self, variables):
        pass


class RestoreStatus:
    def assert_consumed(self):
        pass


class RestoreCheckpoint:
    def restore(self, prefix):
        assert Path(prefix + ".index").is_file()
        return RestoreStatus()


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    return helpers["prepared"].__wrapped__(
        tmp_path, monkeypatch
    )


@pytest.fixture
def published(prepared):
    root, arguments = prepared
    metadata = publication.persist_terminal_gan_checkpoint_v3(
        root, **arguments
    )
    return root, metadata


def request(root, **changes):
    arguments = {
        "requested_dataset_id": "malex",
        "requested_experimental_seed": 42,
        "requested_class_mapping": ["benign", "malware"],
    }
    arguments.update(changes)
    return restoration.load_terminal_gan_checkpoint_v3(
        root, **arguments
    )


@pytest.fixture
def construction_guard(monkeypatch):
    calls = []

    def build(*args, **kwargs):
        calls.append("MODEL_CONSTRUCTED")
        raise AssertionError(
            "Model construction occurred before verification"
        )

    monkeypatch.setattr(gan_model_factory, "build_revision_gan", build)
    return calls


def test_verified_restoration(published, monkeypatch):
    root, metadata = published

    def build(config):
        return {
            "generator": RestoredModel(),
            "discriminator": RestoredModel(),
            "generator_optimizer": RestoredOptimizer(
                metadata["generator_iteration"]
            ),
            "discriminator_optimizer": RestoredOptimizer(
                metadata["discriminator_updates"]
            ),
        }

    monkeypatch.setattr(
        gan_model_factory, "build_revision_gan", build
    )
    monkeypatch.setattr(
        restoration, "_checkpoint_objects",
        lambda *args: RestoreCheckpoint(),
    )

    result = request(root)

    assert result["metadata"] == metadata
    assert result["exposure_evidence"]["dataset_id"] == "malex"
    assert result["dataset_provenance"]["dataset_id"] == "malex"


@pytest.mark.parametrize(
    "filename",
    [
        "model_state.ckpt.index",
        "model_state.ckpt.data-00000-of-00001",
        "gan_exposure_evidence.json",
        "gan_dataset_provenance.json",
    ],
)
def test_corruption_rejected_before_model_construction(
    published, construction_guard, filename
):
    root, metadata = published
    path = root / filename
    path.write_bytes(path.read_bytes() + b"tampered")

    with pytest.raises(ValueError):
        request(root)

    assert construction_guard == []


def test_wrong_seed_rejected_before_models(
    published, construction_guard
):
    root, metadata = published

    with pytest.raises(ValueError, match="seed mismatch"):
        request(root, requested_experimental_seed=43)

    assert construction_guard == []


def test_wrong_dataset_rejected_before_models(
    published, construction_guard
):
    root, metadata = published

    with pytest.raises(ValueError, match="dataset identity"):
        request(root, requested_dataset_id="ustc_tfc2016")

    assert construction_guard == []


def test_wrong_class_order_rejected_before_models(
    published, construction_guard
):
    root, metadata = published

    with pytest.raises(ValueError, match="class mapping"):
        request(
            root,
            requested_class_mapping=["malware", "benign"],
        )

    assert construction_guard == []


def test_publication_lock_rejected_before_models(
    published, construction_guard
):
    root, metadata = published
    (root / ".gan_checkpoint_publication.lock").mkdir()

    with pytest.raises(RuntimeError, match="incomplete"):
        request(root)

    assert construction_guard == []


def test_missing_metadata_rejected_before_models(
    published, construction_guard
):
    root, metadata = published
    (root / "model_metadata.json").unlink()

    with pytest.raises(FileNotFoundError):
        request(root)

    assert construction_guard == []


def test_noncanonical_metadata_rejected_before_models(
    published, construction_guard
):
    root, metadata = published
    (root / "model_metadata.json").write_bytes(
        canonical_json_bytes(metadata)
    )

    with pytest.raises(ValueError, match="not canonical"):
        request(root)

    assert construction_guard == []


def test_v2_metadata_rejected_before_models(
    published, construction_guard
):
    root, metadata = published
    metadata = dict(metadata)
    metadata["schema_version"] = "2.0"
    (root / "model_metadata.json").write_bytes(
        canonical_json_bytes(metadata) + b"\n"
    )

    with pytest.raises(ValueError, match="schema version"):
        request(root)

    assert construction_guard == []


def test_metadata_symlink_rejected_before_models(
    published, construction_guard
):
    root, metadata = published
    original = root / "model_metadata.json"
    destination = root / "other_metadata.json"
    original.rename(destination)
    original.symlink_to(destination)

    with pytest.raises(FileNotFoundError):
        request(root)

    assert construction_guard == []


def test_no_slurm_rejected_before_models(
    published, construction_guard, monkeypatch
):
    root, metadata = published
    monkeypatch.delenv("SLURM_JOB_ID")

    with pytest.raises(RuntimeError, match="Slurm"):
        request(root)

    assert construction_guard == []


def test_restored_iteration_mismatch_rejected(
    published, monkeypatch
):
    root, metadata = published

    def build(config):
        return {
            "generator": RestoredModel(),
            "discriminator": RestoredModel(),
            "generator_optimizer": RestoredOptimizer(
                metadata["generator_iteration"] - 1
            ),
            "discriminator_optimizer": RestoredOptimizer(
                metadata["discriminator_updates"]
            ),
        }

    monkeypatch.setattr(
        gan_model_factory, "build_revision_gan", build
    )
    monkeypatch.setattr(
        restoration, "_checkpoint_objects",
        lambda *args: RestoreCheckpoint(),
    )

    with pytest.raises(ValueError, match="generator iteration"):
        request(root)
