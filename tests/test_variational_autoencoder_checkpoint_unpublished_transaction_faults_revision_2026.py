"""Unpublished VAE generation transaction fault regressions.

Disposable synthetic evidence only; no authoritative publication.
"""

import os

import optree
import keras
import pytest
import tensorflow as tf

from trustforge.models.paper01_revision_2026 import (
    variational_autoencoder_checkpoint_unpublished_transaction as transaction,
)

from tests.test_variational_autoencoder_checkpoint_loader_roundtrip_revision_2026 import (
    synthetic_bundle,
)

from tests.test_variational_autoencoder_checkpoint_unpublished_transaction_filesystem_revision_2026 import (
    transaction_inputs,
    invoke,
    open_parent,
    GENERATION_ID,
)


def assert_preserved(generations, prepared, expected_names):
    generation = generations / GENERATION_ID
    assert generation.is_dir()

    for name in expected_names:
        assert (generation / name).read_bytes() == (
            prepared["artifacts"][name]
        )

    assert not (generations / "commits").exists()


def assert_closed(fd):
    with pytest.raises(OSError):
        os.fstat(fd)


def test_generation_directory_fsync_failure_preserves_artifacts(
    transaction_inputs,
    monkeypatch,
):
    generations, prepared = transaction_inputs
    original_fsync = os.fsync
    intercepted = []

    def failing_fsync(fd):
        generation = generations / GENERATION_ID

        if generation.is_dir():
            current = os.fstat(fd)
            target = generation.stat()

            if (
                current.st_dev == target.st_dev
                and current.st_ino == target.st_ino
            ):
                intercepted.append(fd)
                raise OSError("synthetic generation fsync failure")

        return original_fsync(fd)

    monkeypatch.setattr(
        transaction.os,
        "fsync",
        failing_fsync,
    )

    parent_fd = open_parent(generations)
    try:
        with pytest.raises(
            OSError,
            match="synthetic generation fsync failure",
        ):
            invoke(parent_fd, prepared)

        assert len(intercepted) == 1
        assert_closed(intercepted[0])

        assert_preserved(
            generations,
            prepared,
            set(prepared["artifacts"]),
        )
        os.fstat(parent_fd)
    finally:
        os.close(parent_fd)


def test_precommit_verification_failure_preserves_artifacts(
    transaction_inputs,
    monkeypatch,
):
    generations, prepared = transaction_inputs
    intercepted = []

    def failing_verifier(fd, **kwargs):
        intercepted.append(fd)
        raise ValueError("synthetic precommit rejection")

    monkeypatch.setattr(
        transaction,
        "verify_precommit_vae_generation",
        failing_verifier,
    )

    parent_fd = open_parent(generations)
    try:
        with pytest.raises(
            ValueError,
            match="synthetic precommit rejection",
        ):
            invoke(parent_fd, prepared)

        assert len(intercepted) == 1
        assert_closed(intercepted[0])

        assert_preserved(
            generations,
            prepared,
            set(prepared["artifacts"]),
        )
        os.fstat(parent_fd)
    finally:
        os.close(parent_fd)


def test_post_write_exception_preserves_written_artifacts(
    transaction_inputs,
    monkeypatch,
):
    generations, prepared = transaction_inputs
    original_writer = transaction.write_exclusive_vae_artifact
    calls = []

    def write_then_fail(fd, name, raw):
        result = original_writer(fd, name, raw)
        calls.append((fd, name))

        if len(calls) == 2:
            raise OSError("synthetic post-write failure")

        return result

    monkeypatch.setattr(
        transaction,
        "write_exclusive_vae_artifact",
        write_then_fail,
    )

    parent_fd = open_parent(generations)
    try:
        with pytest.raises(
            OSError,
            match="synthetic post-write failure",
        ):
            invoke(parent_fd, prepared)

        assert len(calls) == 2
        assert calls[0][0] == calls[1][0]
        assert_closed(calls[0][0])

        assert_preserved(
            generations,
            prepared,
            {name for _, name in calls},
        )
        os.fstat(parent_fd)
    finally:
        os.close(parent_fd)
