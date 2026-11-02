"""Real filesystem regressions for unpublished VAE transactions.

Uses disposable synthetic artifacts only. No authoritative publication.
"""

import copy
import os

import optree
import keras
import pytest
import tensorflow as tf

from trustforge.models.paper01_revision_2026 import (
    variational_autoencoder_checkpoint_unpublished_transaction as transaction,
    variational_autoencoder_checkpoint_publication_input as publication_input,
)

from tests.test_variational_autoencoder_checkpoint_loader_roundtrip_revision_2026 import (
    synthetic_bundle,
    DATASET_ID,
    SEED,
    CLASS_MAPPING,
)


GENERATION_ID = "a" * 32


@pytest.fixture
def transaction_inputs(synthetic_bundle, tmp_path):
    source = synthetic_bundle["directory"]
    names = (
        set(publication_input.STATE_FILENAMES)
        | {
            "vae_checkpoint_manifest.json",
            "vae_dataset_provenance.json",
        }
    )

    prepared = {
        "metadata": copy.deepcopy(synthetic_bundle["metadata"]),
        "dataset_provenance": copy.deepcopy(
            synthetic_bundle["provenance"]
        ),
        "artifacts": {
            name: (source / name).read_bytes()
            for name in names
        },
    }

    publication_input.validate_vae_publication_input(prepared)

    generations = tmp_path / "generations"
    generations.mkdir(mode=0o700)
    generations.chmod(0o700)

    return generations, prepared


def invoke(fd, prepared):
    return transaction.prepare_unpublished_vae_generation(
        fd,
        GENERATION_ID,
        prepared=prepared,
        requested_dataset_id=DATASET_ID,
        requested_experimental_seed=SEED,
        requested_class_mapping=CLASS_MAPPING,
    )


def open_parent(path):
    return os.open(
        path,
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | os.O_CLOEXEC,
    )


def test_valid_unpublished_generation(transaction_inputs):
    generations, prepared = transaction_inputs

    parent_fd = open_parent(generations)
    try:
        result = invoke(parent_fd, prepared)
        generation_fd = result["generation_fd"]

        try:
            assert result["publication_status"] == "UNPUBLISHED"
            assert result["generation_id"] == GENERATION_ID

            assert set(result["written_artifacts"]) == set(
                prepared["artifacts"]
            )

            assert result["verified"]["metadata"] == (
                prepared["metadata"]
            )

            os.fstat(generation_fd)

            directory = generations / GENERATION_ID

            for name, raw in prepared["artifacts"].items():
                assert (directory / name).read_bytes() == raw

            assert not (generations / "commits").exists()
        finally:
            os.close(generation_fd)

        os.fstat(parent_fd)
    finally:
        os.close(parent_fd)


def test_invalid_package_rejected_before_reservation(
    transaction_inputs,
):
    generations, _ = transaction_inputs

    parent_fd = open_parent(generations)
    try:
        with pytest.raises(TypeError):
            invoke(parent_fd, None)

        assert list(generations.iterdir()) == []
        os.fstat(parent_fd)
    finally:
        os.close(parent_fd)


def test_partial_write_failure_preserves_generation(
    transaction_inputs,
    monkeypatch,
):
    generations, prepared = transaction_inputs

    original = transaction.write_exclusive_vae_artifact
    calls = []
    owned_fds = []

    def failing_writer(fd, name, raw):
        owned_fds.append(fd)
        calls.append(name)

        if len(calls) == 2:
            raise OSError("synthetic second-write failure")

        return original(fd, name, raw)

    monkeypatch.setattr(
        transaction,
        "write_exclusive_vae_artifact",
        failing_writer,
    )

    parent_fd = open_parent(generations)
    try:
        with pytest.raises(
            OSError,
            match="synthetic second-write failure",
        ):
            invoke(parent_fd, prepared)

        directory = generations / GENERATION_ID
        assert directory.is_dir()
        assert len(calls) == 2

        assert (
            directory / calls[0]
        ).read_bytes() == prepared["artifacts"][calls[0]]

        assert not (directory / calls[1]).exists()

        assert len(set(owned_fds)) == 1

        with pytest.raises(OSError):
            os.fstat(owned_fds[0])

        os.fstat(parent_fd)
        assert not (generations / "commits").exists()
    finally:
        os.close(parent_fd)


def test_duplicate_generation_rejected_without_overwrite(
    transaction_inputs,
):
    generations, prepared = transaction_inputs

    parent_fd = open_parent(generations)
    try:
        result = invoke(parent_fd, prepared)

        try:
            original_bytes = {
                name: (
                    generations / GENERATION_ID / name
                ).read_bytes()
                for name in prepared["artifacts"]
            }

            with pytest.raises(FileExistsError):
                invoke(parent_fd, prepared)

            for name, raw in original_bytes.items():
                assert (
                    generations / GENERATION_ID / name
                ).read_bytes() == raw
        finally:
            os.close(result["generation_fd"])
    finally:
        os.close(parent_fd)
