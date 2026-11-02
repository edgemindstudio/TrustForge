"""Integrated synthetic VAE commit-publication regressions.

All checkpoint artifacts are disposable. No real-data training.
No production checkpoint publication or crash-durability claims.
"""

import os

import optree
import keras
import pytest
import tensorflow as tf

from trustforge.models.paper01_revision_2026 import (
    variational_autoencoder_checkpoint_commit_publisher as publisher,
    variational_autoencoder_checkpoint_committed_generation as verifier,
)

from tests.test_variational_autoencoder_checkpoint_loader_roundtrip_revision_2026 import (
    synthetic_bundle,
    DATASET_ID,
    SEED,
    CLASS_MAPPING,
)

from tests.test_variational_autoencoder_checkpoint_unpublished_transaction_filesystem_revision_2026 import (
    transaction_inputs,
    invoke,
    open_parent,
    GENERATION_ID,
)


@pytest.fixture
def publication_setup(transaction_inputs):
    generations, prepared = transaction_inputs
    root = generations.parent

    commits = root / "commits"
    commits.mkdir(mode=0o700)
    commits.chmod(0o700)

    generations_fd = open_parent(generations)
    commits_fd = open_parent(commits)

    generation_fd = None

    try:
        unpublished = invoke(generations_fd, prepared)
        generation_fd = unpublished["generation_fd"]

        assert unpublished["publication_status"] == "UNPUBLISHED"

        yield {
            "root": root,
            "generations": generations,
            "commits": commits,
            "generations_fd": generations_fd,
            "commits_fd": commits_fd,
            "generation_fd": generation_fd,
            "prepared": prepared,
        }

    finally:
        if generation_fd is not None:
            os.close(generation_fd)
        os.close(commits_fd)
        os.close(generations_fd)


def publish(case):
    return publisher.publish_candidate_vae_commit(
        case["generations_fd"],
        case["commits_fd"],
        case["generation_fd"],
        GENERATION_ID,
        prepared=case["prepared"],
        requested_dataset_id=DATASET_ID,
        requested_experimental_seed=SEED,
        requested_class_mapping=CLASS_MAPPING,
    )


def final_marker(case):
    return case["commits"] / (GENERATION_ID + ".commit.json")


def staged_names(case):
    return sorted(
        p.name
        for p in case["commits"].iterdir()
        if p.name.startswith(".vae-staged-")
    )


def assert_descriptors_open(case):
    os.fstat(case["generations_fd"])
    os.fstat(case["commits_fd"])
    os.fstat(case["generation_fd"])


def test_successful_commit_and_independent_reader(publication_setup):
    case = publication_setup

    result = publish(case)

    assert result["generation_id"] == GENERATION_ID
    assert result["outcome"] == "committed_synced"
    assert result["crash_durability_verified"] is False
    assert final_marker(case).is_file()
    assert staged_names(case) == []

    verified = verifier.verify_committed_vae_generation(
        case["root"],
        generation_id=GENERATION_ID,
        requested_dataset_id=DATASET_ID,
        requested_experimental_seed=SEED,
        requested_class_mapping=CLASS_MAPPING,
    )

    assert verified is not None
    assert_descriptors_open(case)


def test_duplicate_commit_never_overwrites(publication_setup):
    case = publication_setup

    publish(case)
    before = final_marker(case).read_bytes()

    with pytest.raises(FileExistsError):
        publish(case)

    assert final_marker(case).read_bytes() == before
    assert len(staged_names(case)) == 1
    assert_descriptors_open(case)


def test_staging_write_failure_has_no_commit(
    publication_setup,
    monkeypatch,
):
    case = publication_setup

    def fail_write(*args, **kwargs):
        raise OSError("synthetic staging write failure")

    with monkeypatch.context() as patch:
        patch.setattr(
            publisher,
            "write_exclusive_vae_artifact",
            fail_write,
        )

        with pytest.raises(
            OSError,
            match="synthetic staging write failure",
        ):
            publish(case)

    assert not final_marker(case).exists()
    assert staged_names(case) == []
    assert_descriptors_open(case)


def test_prelink_directory_sync_failure_has_no_commit(
    publication_setup,
    monkeypatch,
):
    case = publication_setup
    original_fsync = os.fsync

    def fail_prelink_sync(fd):
        if fd == case["commits_fd"]:
            raise OSError("synthetic prelink sync failure")
        return original_fsync(fd)

    with monkeypatch.context() as patch:
        patch.setattr(publisher.os, "fsync", fail_prelink_sync)

        with pytest.raises(
            OSError,
            match="synthetic prelink sync failure",
        ):
            publish(case)

    assert not final_marker(case).exists()
    assert len(staged_names(case)) == 1
    assert_descriptors_open(case)


def test_link_failure_is_indeterminate(
    publication_setup,
    monkeypatch,
):
    case = publication_setup

    def fail_link(*args, **kwargs):
        raise OSError("synthetic link failure")

    with monkeypatch.context() as patch:
        patch.setattr(publisher.os, "link", fail_link)

        with pytest.raises(
            publisher.IndeterminateVAECommitError,
            match="indeterminate",
        ):
            publish(case)

    assert not final_marker(case).exists()
    assert len(staged_names(case)) == 1
    assert_descriptors_open(case)


def test_successful_link_followed_by_error_is_indeterminate(
    publication_setup,
    monkeypatch,
):
    case = publication_setup
    original_link = os.link

    def link_then_fail(*args, **kwargs):
        original_link(*args, **kwargs)
        raise OSError("synthetic ambiguous link result")

    with monkeypatch.context() as patch:
        patch.setattr(publisher.os, "link", link_then_fail)

        with pytest.raises(
            publisher.IndeterminateVAECommitError,
            match="indeterminate",
        ):
            publish(case)

    assert final_marker(case).is_file()
    assert len(staged_names(case)) == 1
    assert_descriptors_open(case)


def test_postlink_directory_sync_failure_is_indeterminate(
    publication_setup,
    monkeypatch,
):
    case = publication_setup
    original_fsync = os.fsync

    def fail_postlink_sync(fd):
        if (
            fd == case["commits_fd"]
            and final_marker(case).exists()
        ):
            raise OSError("synthetic postlink sync failure")

        return original_fsync(fd)

    with monkeypatch.context() as patch:
        patch.setattr(publisher.os, "fsync", fail_postlink_sync)

        with pytest.raises(
            publisher.IndeterminateVAECommitError,
            match="indeterminate",
        ):
            publish(case)

    assert final_marker(case).is_file()
    assert_descriptors_open(case)
