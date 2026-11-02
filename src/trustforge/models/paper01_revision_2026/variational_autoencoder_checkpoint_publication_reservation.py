"""Exclusive generation-directory reservation for VAE publication.

The caller owns generations_fd and the returned generation descriptor.
A reserved generation is not a committed checkpoint. No rollback occurs.
"""

from __future__ import annotations

import os
import stat

from .variational_autoencoder_checkpoint_commit_record import (
    validate_vae_generation_id,
)
from .variational_autoencoder_checkpoint_publication_namespace_policy import (
    validate_private_vae_generation_parent,
)


def reserve_vae_generation_directory(generations_fd, generation_id):
    """Create and synchronize a new generation directory exclusively.

    The returned descriptor must be closed by the caller.
    On failure, do not remove a possibly created directory.
    """
    generation_id = validate_vae_generation_id(generation_id)

    if type(generations_fd) is not int:
        raise TypeError("generations_fd must be an integer")

    try:
        parent_info = os.fstat(generations_fd)
    except OSError as exc:
        raise ValueError("invalid generations directory descriptor") from exc

    if not stat.S_ISDIR(parent_info.st_mode):
        raise ValueError("generations descriptor is not a directory")

    validate_private_vae_generation_parent(generations_fd)

    required = ("O_DIRECTORY", "O_NOFOLLOW", "O_CLOEXEC")

    if any(not hasattr(os, flag) for flag in required):
        raise RuntimeError("required directory flags unavailable")

    os.mkdir(
        generation_id,
        mode=0o700,
        dir_fd=generations_fd,
    )

    flags = (
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | os.O_CLOEXEC
    )

    descriptor = os.open(
        generation_id,
        flags,
        dir_fd=generations_fd,
    )

    try:
        child_info = os.fstat(descriptor)

        if not stat.S_ISDIR(child_info.st_mode):
            raise ValueError("reserved generation is not a directory")

        os.fsync(generations_fd)

        return descriptor

    except BaseException:
        os.close(descriptor)
        raise


__all__ = ["reserve_vae_generation_directory"]
