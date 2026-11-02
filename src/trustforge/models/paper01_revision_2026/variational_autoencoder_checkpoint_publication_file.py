"""Exclusive descriptor-relative VAE publication artifact writes.

This primitive does not create generation directories, commit records,
or authoritative checkpoint publications. It does not perform rollback.
"""

from __future__ import annotations

import hashlib
import os
import stat


def write_exclusive_vae_artifact(directory_fd, name, raw):
    """Exclusively create, write, and fsync one regular artifact.

    On write or sync failure, the created file is retained for
    explicit recovery. The caller owns the directory descriptor.
    """
    if type(directory_fd) is not int:
        raise TypeError("directory_fd must be an integer")

    try:
        directory_info = os.fstat(directory_fd)
    except OSError as exc:
        raise ValueError("invalid publication directory descriptor") from exc

    if not stat.S_ISDIR(directory_info.st_mode):
        raise ValueError("publication descriptor is not a directory")

    if (
        type(name) is not str
        or not name
        or name in (".", "..")
        or "/" in name
        or "\x00" in name
    ):
        raise ValueError("invalid publication artifact basename")

    if type(raw) is not bytes or not raw:
        raise ValueError("publication artifact must be nonempty bytes")

    required = ("O_NOFOLLOW", "O_CLOEXEC", "O_EXCL")

    if any(not hasattr(os, flag) for flag in required):
        raise RuntimeError("required exclusive-write flags unavailable")

    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | os.O_NOFOLLOW
        | os.O_CLOEXEC
    )

    descriptor = os.open(
        name,
        flags,
        0o600,
        dir_fd=directory_fd,
    )

    try:
        created = os.fstat(descriptor)

        if not stat.S_ISREG(created.st_mode):
            raise ValueError("created artifact is not regular")

        if created.st_nlink != 1:
            raise ValueError("created artifact has unexpected link count")

        position = 0

        while position < len(raw):
            written = os.write(
                descriptor,
                memoryview(raw)[position:],
            )

            if written <= 0:
                raise OSError("publication artifact write made no progress")

            position += written

        os.fsync(descriptor)

        return {
            "filename": name,
            "size_bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }

    finally:
        os.close(descriptor)


__all__ = ["write_exclusive_vae_artifact"]
