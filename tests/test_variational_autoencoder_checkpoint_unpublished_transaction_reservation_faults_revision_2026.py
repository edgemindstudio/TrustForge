"""Reservation-stage recovery tests for unpublished VAE generations.

Disposable synthetic fixtures only. No commit publication or training.
"""

import os

import optree
import keras
import pytest
import tensorflow as tf

from trustforge.models.paper01_revision_2026 import (
    variational_autoencoder_checkpoint_publication_reservation as reservation,
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


def assert_empty_unpublished_generation(generations):
    generation = generations / GENERATION_ID
    assert generation.is_dir()
    assert list(generation.iterdir()) == []
    assert not (generations / "commits").exists()


def test_post_mkdir_open_failure_preserves_empty_generation(
    transaction_inputs,
    monkeypatch,
):
    generations, prepared = transaction_inputs
    parent_fd = open_parent(generations)
    original_open = os.open
    intercepted = []

    def fail_generation_open(name, flags, *args, **kwargs):
        if (
            name == GENERATION_ID
            and kwargs.get("dir_fd") == parent_fd
        ):
            intercepted.append(name)
            raise OSError("synthetic reservation open failure")
        return original_open(name, flags, *args, **kwargs)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(reservation.os, "open", fail_generation_open)

            with pytest.raises(
                OSError,
                match="synthetic reservation open failure",
            ):
                invoke(parent_fd, prepared)

        assert intercepted == [GENERATION_ID]
        assert_empty_unpublished_generation(generations)
        os.fstat(parent_fd)
    finally:
        os.close(parent_fd)


def test_parent_fsync_failure_closes_child_and_preserves_directory(
    transaction_inputs,
    monkeypatch,
):
    generations, prepared = transaction_inputs
    parent_fd = open_parent(generations)
    original_open = os.open
    original_fsync = os.fsync
    child_fds = []
    parent_sync_attempts = []

    def observe_generation_open(name, flags, *args, **kwargs):
        fd = original_open(name, flags, *args, **kwargs)
        if (
            name == GENERATION_ID
            and kwargs.get("dir_fd") == parent_fd
        ):
            child_fds.append(fd)
        return fd

    def fail_parent_fsync(fd):
        if fd == parent_fd:
            parent_sync_attempts.append(fd)
            raise OSError("synthetic reservation parent fsync failure")
        return original_fsync(fd)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(
                reservation.os,
                "open",
                observe_generation_open,
            )
            patch.setattr(
                reservation.os,
                "fsync",
                fail_parent_fsync,
            )

            with pytest.raises(
                OSError,
                match="synthetic reservation parent fsync failure",
            ):
                invoke(parent_fd, prepared)

        assert parent_sync_attempts == [parent_fd]
        assert len(child_fds) == 1

        with pytest.raises(OSError):
            os.fstat(child_fds[0])

        assert_empty_unpublished_generation(generations)
        os.fstat(parent_fd)
    finally:
        os.close(parent_fd)
