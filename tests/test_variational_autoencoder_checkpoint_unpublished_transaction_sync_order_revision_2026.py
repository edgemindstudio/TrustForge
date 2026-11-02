"""Synchronization-order regression for unpublished VAE generations.

Real disposable synthetic files; no authoritative publication or training.
"""

import os
import stat

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


def test_all_file_syncs_precede_generation_sync_and_verification(
    transaction_inputs,
    monkeypatch,
):
    generations, prepared = transaction_inputs
    parent_fd = open_parent(generations)

    original_fsync = os.fsync
    original_writer = transaction.write_exclusive_vae_artifact
    original_verifier = transaction.verify_precommit_vae_generation

    events = []
    generation = generations / GENERATION_ID

    def observe_fsync(fd):
        info = os.fstat(fd)

        if stat.S_ISREG(info.st_mode):
            events.append("file_sync")

        elif generation.is_dir():
            target = generation.stat()

            if (
                info.st_dev == target.st_dev
                and info.st_ino == target.st_ino
            ):
                events.append("generation_sync")

        return original_fsync(fd)

    def observe_write(fd, name, raw):
        result = original_writer(fd, name, raw)
        events.append("write:" + name)
        return result

    def observe_verify(fd, **kwargs):
        events.append("verify")
        return original_verifier(fd, **kwargs)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(
                transaction.os,
                "fsync",
                observe_fsync,
            )
            patch.setattr(
                transaction,
                "write_exclusive_vae_artifact",
                observe_write,
            )
            patch.setattr(
                transaction,
                "verify_precommit_vae_generation",
                observe_verify,
            )

            result = invoke(parent_fd, prepared)

        try:
            expected_names = [
                *sorted(transaction.STATE_FILENAMES),
                transaction.VAE_PROVENANCE_FILENAME,
                transaction.MANIFEST_FILENAME,
            ]

            expected_events = []

            for name in expected_names:
                expected_events.extend([
                    "file_sync",
                    "write:" + name,
                ])

            expected_events.extend([
                "generation_sync",
                "verify",
            ])

            assert events == expected_events

            assert result["publication_status"] == "UNPUBLISHED"

            assert set(result["written_artifacts"]) == (
                set(prepared["artifacts"])
            )

            assert not (generations / "commits").exists()

            os.fstat(result["generation_fd"])

        finally:
            os.close(result["generation_fd"])

        os.fstat(parent_fd)

    finally:
        os.close(parent_fd)
