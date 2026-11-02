"""Synthetic same-UID generation substitution security regression."""

import os

import pytest

from trustforge.models.paper01_revision_2026 import (
    variational_autoencoder_checkpoint_publication_reservation as reservation,
)

GENERATION_ID = "d" * 32


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Reservation cannot prove that the opened directory "
        "is the inode originally created by mkdir"
    ),
)
def test_empty_private_same_uid_substitution_rejected(
    tmp_path,
    monkeypatch,
):
    generations = tmp_path / "generations"
    generations.mkdir(mode=0o700)
    generations.chmod(0o700)

    replacement = tmp_path / "replacement"
    replacement.mkdir(mode=0o700)
    replacement.chmod(0o700)

    assert list(replacement.iterdir()) == []

    parent_fd = os.open(
        generations,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    )

    actual_mkdir = os.mkdir
    evidence = {}

    def substitute_after_mkdir(name, *args, **kwargs):
        result = actual_mkdir(name, *args, **kwargs)

        if name == GENERATION_ID:
            created = generations / GENERATION_ID

            evidence["created_inode"] = created.stat().st_ino
            evidence["created_device"] = created.stat().st_dev

            created.rmdir()
            replacement.rename(created)

            evidence["replacement_inode"] = created.stat().st_ino
            evidence["replacement_device"] = created.stat().st_dev

        return result

    try:
        with monkeypatch.context() as patch:
            patch.setattr(
                reservation.os,
                "mkdir",
                substitute_after_mkdir,
            )

            child_fd = None

            try:
                child_fd = reservation.reserve_vae_generation_directory(
                    parent_fd,
                    GENERATION_ID,
                )
            except (ValueError, OSError):
                return
            finally:
                if child_fd is not None:
                    os.close(child_fd)

        assert evidence["created_inode"] != evidence["replacement_inode"]

        pytest.fail(
            "Reservation accepted an empty, private, "
            "same-UID substituted directory"
        )
    finally:
        os.close(parent_fd)
