"""Candidate descriptor-anchored VAE checkpoint artifact access.

A directory descriptor anchors member operations to one directory inode.
It does not establish immutable contents or authenticate provenance.
"""

from __future__ import annotations

import os
import stat
from contextlib import contextmanager
from pathlib import Path


def _require_basename(name):
    if (
        type(name) is not str
        or not name
        or name in (".", "..")
        or "/" in name
        or "\x00" in name
    ):
        raise ValueError("invalid checkpoint artifact basename")

    return name


@contextmanager
def open_checkpoint_directory(directory):
    """Open a real directory without following its final symlink."""
    root = Path(directory).expanduser()

    required = (
        "O_DIRECTORY",
        "O_NOFOLLOW",
        "O_CLOEXEC",
    )

    if any(not hasattr(os, name) for name in required):
        raise RuntimeError(
            "required directory-opening flags unavailable"
        )

    flags = (
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | os.O_CLOEXEC
    )

    try:
        descriptor = os.open(root, flags)
    except OSError as exc:
        raise ValueError(
            "invalid VAE checkpoint directory"
        ) from exc

    try:
        metadata = os.fstat(descriptor)

        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(
                "checkpoint anchor is not a directory"
            )

        yield descriptor

    finally:
        os.close(descriptor)


def anchored_member_names(directory_fd):
    """List names from the already-open checkpoint directory."""
    return tuple(os.listdir(directory_fd))


def anchored_member_exists(directory_fd, name):
    """Check a named entry without following a symlink."""
    name = _require_basename(name)

    try:
        os.stat(
            name,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return False

    return True


def read_anchored_regular_file(
    directory_fd,
    name,
    *,
    max_bytes,
):
    """Read one bounded regular file relative to an opened directory."""
    name = _require_basename(name)

    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError(
            "max_bytes must be a positive integer"
        )

    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError(
            "O_NOFOLLOW is required"
        )

    flags = (
        os.O_RDONLY
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | os.O_NOFOLLOW
    )

    try:
        descriptor = os.open(
            name,
            flags,
            dir_fd=directory_fd,
        )
    except OSError as exc:
        raise ValueError(
            "cannot securely open anchored checkpoint artifact"
        ) from exc

    try:
        before = os.fstat(descriptor)

        if not stat.S_ISREG(before.st_mode):
            raise ValueError(
                "anchored checkpoint artifact is not a regular file"
            )

        if not 0 < before.st_size <= max_bytes:
            raise ValueError(
                "anchored checkpoint artifact size is invalid"
            )

        chunks = []
        remaining = max_bytes + 1

        while remaining > 0:
            chunk = os.read(
                descriptor,
                min(65536, remaining),
            )

            if not chunk:
                break

            chunks.append(chunk)
            remaining -= len(chunk)

        raw = b"".join(chunks)
        after = os.fstat(descriptor)

        if len(raw) != before.st_size:
            raise ValueError(
                "anchored checkpoint artifact size changed"
            )

        identity = (
            "st_dev",
            "st_ino",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )

        if any(
            getattr(before, name) != getattr(after, name)
            for name in identity
        ):
            raise ValueError(
                "anchored checkpoint artifact changed during reading"
            )

        return raw

    finally:
        os.close(descriptor)


__all__ = [
    "open_checkpoint_directory",
    "anchored_member_names",
    "anchored_member_exists",
    "read_anchored_regular_file",
]
