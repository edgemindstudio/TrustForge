"""Additional synthetic VAE exclusive-writer failure regressions.

All filesystem operations use pytest temporary directories.
No generation commits, authoritative publication, or training.
"""

import os

import pytest

from trustforge.models.paper01_revision_2026 import (
    variational_autoencoder_checkpoint_publication_file as writer,
)


def test_zero_progress_write_rejected_and_file_preserved(
    tmp_path,
    monkeypatch,
):
    directory_fd = os.open(
        tmp_path,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    )

    try:
        with monkeypatch.context() as patch:
            patch.setattr(writer.os, "write", lambda fd, data: 0)

            with pytest.raises(
                OSError,
                match="write made no progress",
            ):
                writer.write_exclusive_vae_artifact(
                    directory_fd,
                    "zero-progress.bin",
                    b"synthetic-zero-progress-payload",
                )

        artifact = tmp_path / "zero-progress.bin"
        assert artifact.is_file()
        assert artifact.read_bytes() == b""

    finally:
        os.close(directory_fd)


def test_close_failure_propagates_without_removing_artifact(
    tmp_path,
    monkeypatch,
):
    directory_fd = os.open(
        tmp_path,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    )

    real_close = os.close
    close_events = []

    def close_then_report_failure(fd):
        real_close(fd)
        if fd != directory_fd:
            close_events.append("artifact-close")
            raise OSError("injected close failure")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(writer.os, "close", close_then_report_failure)

            with pytest.raises(
                OSError,
                match="injected close failure",
            ):
                writer.write_exclusive_vae_artifact(
                    directory_fd,
                    "close-failure.bin",
                    b"synthetic-close-failure-payload",
                )

        assert close_events == ["artifact-close"]
        assert (
            tmp_path / "close-failure.bin"
        ).read_bytes() == b"synthetic-close-failure-payload"

    finally:
        real_close(directory_fd)
