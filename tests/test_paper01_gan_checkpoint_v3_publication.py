"""Fixture-only checkpoint v3 publication lifecycle tests.

No TensorFlow checkpoint or production dataset is used.
The separate artifact and bundle suites cover integrity validation.
"""

from pathlib import Path

import pytest

from trustforge.models.paper01_revision_2026 import (
    gan_checkpoint_v3_persistence as persistence,
)
from trustforge.models.paper01_revision_2026.gan_checkpoint_v3_provenance import (
    SCHEMA_TYPE,
    SCHEMA_VERSION,
    USTC_PARTITION_POLICY,
    USTC_LIMITATIONS,
)
from trustforge.models.paper01_revision_2026.gan_dataset_provenance import (
    USTC_NCHW_SOURCE_SHA256,
)


class FakeIteration:
    def __init__(self, value):
        self.value = value

    def numpy(self):
        return self.value


class FakeOptimizer:
    def __init__(self, iterations):
        self.iterations = FakeIteration(iterations)


class FakeDimension:
    def __init__(self, classes):
        self.shape = (None, classes)


class FakeModel:
    def __init__(self, classes):
        self.inputs = [object(), FakeDimension(classes)]


class FakeCheckpoint:
    def write(self, prefix):
        prefix = Path(prefix)
        (prefix.parent / (prefix.name + ".index")).write_bytes(
            b"fixture-index"
        )
        (
            prefix.parent /
            (prefix.name + ".data-00000-of-00001")
        ).write_bytes(b"fixture-state")


def provenance():
    return {
        "schema_type": SCHEMA_TYPE,
        "schema_version": SCHEMA_VERSION,
        "dataset_id": "ustc_tfc2016",
        "source_contract": "ustc_nchw_revision_2026",
        "source_fingerprints": dict(USTC_NCHW_SOURCE_SHA256),
        "partition_policy": dict(USTC_PARTITION_POLICY),
        "verification_method":
            "sha256_full_file_stream_on_compute_node",
        "verification_scope": "four_original_nchw_numpy_files",
        "verification_limitations": list(USTC_LIMITATIONS),
    }


def kwargs():
    return {
        "dataset_id": "ustc_tfc2016",
        "experimental_seed": 42,
        "class_mapping": [str(x) for x in range(9)],
        "planned_generator_iterations": 10,
        "exposure_evidence": {"fixture": True},
        "dataset_provenance": provenance(),
        "generator": FakeModel(9),
        "discriminator": FakeModel(9),
        "generator_optimizer": FakeOptimizer(10),
        "discriminator_optimizer": FakeOptimizer(20),
    }


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "synthetic-test-only")
    monkeypatch.setattr(
        persistence,
        "_checkpoint_objects",
        lambda *args: FakeCheckpoint(),
    )

    checks = []

    def verify(directory, metadata):
        directory = Path(directory)
        checks.append(directory)

        for name in (
            *metadata["state_files"],
            "gan_exposure_evidence.json",
            "gan_dataset_provenance.json",
        ):
            assert (directory / name).is_file()

        assert not (
            directory / "model_metadata.json"
        ).exists()

    monkeypatch.setattr(
        persistence,
        "_verify_unpublished_v3_artifacts",
        verify,
    )

    return tmp_path, checks


def test_successful_publication_metadata_last(setup):
    root, checks = setup
    metadata = persistence.persist_terminal_gan_checkpoint_v3(
        root, **kwargs()
    )

    assert metadata["schema_version"] == "3.0"
    assert len(checks) == 2
    assert checks[0] != root
    assert checks[1] == root

    for name in (
        "model_metadata.json",
        "gan_exposure_evidence.json",
        "gan_dataset_provenance.json",
        "model_state.ckpt.index",
        "model_state.ckpt.data-00000-of-00001",
    ):
        assert (root / name).is_file()

    assert not (
        root / ".gan_checkpoint_publication.lock"
    ).exists()


def test_duplicate_publication_rejected(setup):
    root, _ = setup
    persistence.persist_terminal_gan_checkpoint_v3(
        root, **kwargs()
    )

    with pytest.raises(FileExistsError):
        persistence.persist_terminal_gan_checkpoint_v3(
            root, **kwargs()
        )

    assert (root / "model_metadata.json").is_file()
    assert not (
        root / ".gan_checkpoint_publication.lock"
    ).exists()


def test_missing_provenance_rejected_before_publication(setup):
    root, _ = setup
    arguments = kwargs()
    del arguments["dataset_provenance"]["source_fingerprints"]

    with pytest.raises(ValueError):
        persistence.persist_terminal_gan_checkpoint_v3(
            root, **arguments
        )

    assert list(root.iterdir()) == []


def test_invalid_terminal_iteration_rejected(setup):
    root, _ = setup
    arguments = kwargs()
    arguments["generator_optimizer"] = FakeOptimizer(9)

    with pytest.raises(ValueError, match="terminal iteration"):
        persistence.persist_terminal_gan_checkpoint_v3(
            root, **arguments
        )

    assert list(root.iterdir()) == []


def test_preexisting_provenance_rejected(setup):
    root, _ = setup
    (root / "gan_dataset_provenance.json").write_bytes(
        b"existing"
    )

    with pytest.raises(FileExistsError):
        persistence.persist_terminal_gan_checkpoint_v3(
            root, **kwargs()
        )

    assert (
        root / "gan_dataset_provenance.json"
    ).read_bytes() == b"existing"

    assert not (root / "model_metadata.json").exists()


def test_publication_lock_collision_rejected(setup):
    root, _ = setup
    lock = root / ".gan_checkpoint_publication.lock"
    lock.mkdir()

    with pytest.raises(FileExistsError):
        persistence.persist_terminal_gan_checkpoint_v3(
            root, **kwargs()
        )

    assert lock.is_dir()
    assert not (root / "model_metadata.json").exists()


def test_staging_verification_failure_rolls_back(
    setup, monkeypatch
):
    root, checks = setup

    def fail_stage(directory, metadata):
        raise ValueError("injected staging failure")

    monkeypatch.setattr(
        persistence,
        "_verify_unpublished_v3_artifacts",
        fail_stage,
    )

    with pytest.raises(ValueError, match="injected staging failure"):
        persistence.persist_terminal_gan_checkpoint_v3(
            root, **kwargs()
        )

    assert list(root.iterdir()) == []


def test_published_verification_failure_rolls_back(
    setup, monkeypatch
):
    root, checks = setup
    original = persistence._verify_unpublished_v3_artifacts

    def fail_at_root(directory, metadata):
        original(directory, metadata)
        if Path(directory) == root:
            raise ValueError("injected published verification failure")

    monkeypatch.setattr(
        persistence,
        "_verify_unpublished_v3_artifacts",
        fail_at_root,
    )

    with pytest.raises(
        ValueError,
        match="injected published verification failure",
    ):
        persistence.persist_terminal_gan_checkpoint_v3(
            root, **kwargs()
        )

    assert len(checks) == 2
    assert list(root.iterdir()) == []


def test_hardlink_failure_rolls_back_partial_publication(
    setup, monkeypatch
):
    root, _ = setup
    original_link = persistence.os.link
    calls = []

    def failing_link(source, destination):
        calls.append(destination)
        if len(calls) == 2:
            raise OSError("injected hardlink failure")
        return original_link(source, destination)

    monkeypatch.setattr(
        persistence.os,
        "link",
        failing_link,
    )

    with pytest.raises(OSError, match="injected hardlink failure"):
        persistence.persist_terminal_gan_checkpoint_v3(
            root, **kwargs()
        )

    assert len(calls) == 2
    assert list(root.iterdir()) == []


def test_no_slurm_rejected_before_publication(
    setup, monkeypatch
):
    root, _ = setup
    monkeypatch.delenv("SLURM_JOB_ID")

    with pytest.raises(RuntimeError, match="Slurm"):
        persistence.persist_terminal_gan_checkpoint_v3(
            root, **kwargs()
        )

    assert list(root.iterdir()) == []


def test_final_directory_sync_failure_rolls_back(
    setup, monkeypatch
):
    """A failed final fsync must not leave authoritative metadata."""
    root, checks = setup
    original_sync = persistence._fsync_directory
    calls = []

    def failing_sync(directory):
        if Path(directory) == root:
            calls.append(len(calls) + 1)
            if len(calls) == 2:
                raise OSError("injected final metadata fsync failure")
        return original_sync(directory)

    monkeypatch.setattr(
        persistence,
        "_fsync_directory",
        failing_sync,
    )

    with pytest.raises(
        OSError,
        match="injected final metadata fsync failure",
    ):
        persistence.persist_terminal_gan_checkpoint_v3(
            root, **kwargs()
        )

    assert len(checks) == 2
    assert len(calls) >= 3
    assert not (root / "model_metadata.json").exists()
    assert not (root / "gan_exposure_evidence.json").exists()
    assert not (root / "gan_dataset_provenance.json").exists()
    assert not list(root.glob("model_state.ckpt.*"))
    assert not (root / ".gan_checkpoint_publication.lock").exists()
    assert list(root.iterdir()) == []
