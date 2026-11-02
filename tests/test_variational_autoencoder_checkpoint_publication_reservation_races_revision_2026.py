"""Synthetic post-mkdir VAE generation reservation race regressions.

All mutations occur in pytest-managed disposable directories.
The replacement-directory scenario documents an unresolved security risk.
"""

import os

import pytest

from trustforge.models.paper01_revision_2026 import (
    variational_autoencoder_checkpoint_publication_reservation as reservation,
)


GENERATION_ID = "b" * 32


@pytest.fixture
def reservation_root(tmp_path):
    generations = tmp_path / "generations"
    generations.mkdir(mode=0o700)
    generations.chmod(0o700)

    fd = os.open(
        generations,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    )

    try:
        yield fd, generations, tmp_path
    finally:
        os.close(fd)


def test_open_failure_after_mkdir_preserves_directory(
    reservation_root,
    monkeypatch,
):
    fd, generations, _ = reservation_root
    actual_open = os.open

    def reject_generation_open(path, flags, *args, **kwargs):
        if path == GENERATION_ID:
            raise OSError("injected post-mkdir open failure")
        return actual_open(path, flags, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(
            reservation.os,
            "open",
            reject_generation_open,
        )

        with pytest.raises(
            OSError,
            match="injected post-mkdir open failure",
        ):
            reservation.reserve_vae_generation_directory(
                fd, GENERATION_ID
            )

    assert (generations / GENERATION_ID).is_dir()


def test_post_mkdir_symlink_substitution_rejected(
    reservation_root,
    monkeypatch,
):
    fd, generations, tmp_path = reservation_root
    actual_mkdir = os.mkdir

    outside = tmp_path / "outside"
    outside.mkdir()

    def substitute_symlink(path, *args, **kwargs):
        result = actual_mkdir(path, *args, **kwargs)

        if path == GENERATION_ID:
            created = generations / GENERATION_ID
            created.rmdir()
            created.symlink_to(
                outside,
                target_is_directory=True,
            )

        return result

    with monkeypatch.context() as patch:
        patch.setattr(
            reservation.os,
            "mkdir",
            substitute_symlink,
        )

        with pytest.raises(OSError):
            reservation.reserve_vae_generation_directory(
                fd, GENERATION_ID
            )

    assert (generations / GENERATION_ID).is_symlink()


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Current reservation lacks proof that an opened directory "
        "is the directory created by mkdir"
    ),
)
def test_post_mkdir_real_directory_substitution_must_be_rejected(
    reservation_root,
    monkeypatch,
):
    fd, generations, tmp_path = reservation_root
    actual_mkdir = os.mkdir

    replacement = tmp_path / "replacement"
    replacement.mkdir()
    (replacement / "foreign.txt").write_text(
        "foreign-directory",
        encoding="utf-8",
    )

    def substitute_directory(path, *args, **kwargs):
        result = actual_mkdir(path, *args, **kwargs)

        if path == GENERATION_ID:
            created = generations / GENERATION_ID
            created.rmdir()
            replacement.rename(created)

        return result

    with monkeypatch.context() as patch:
        patch.setattr(
            reservation.os,
            "mkdir",
            substitute_directory,
        )

        child_fd = reservation.reserve_vae_generation_directory(
            fd, GENERATION_ID
        )

    try:
        assert not (generations / GENERATION_ID / "foreign.txt").exists(), (
            "Reservation accepted a substituted foreign directory"
        )
    finally:
        os.close(child_fd)
