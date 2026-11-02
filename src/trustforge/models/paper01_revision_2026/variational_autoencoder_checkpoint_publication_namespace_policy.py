"""Private namespace policy for VAE publication.

Checks basic POSIX ownership and mode of an opened directory.
Does not guarantee inode immutability, exclude same-UID actors,
inspect filesystem ACLs, or authorize authoritative publication.
"""

from __future__ import annotations

import os
import stat


def validate_private_vae_generation_parent(directory_fd):
    """Require an opened directory owned by this UID with private mode."""
    if type(directory_fd) is not int:
        raise TypeError("generation parent FD must be an integer")

    try:
        info = os.fstat(directory_fd)
    except OSError as exc:
        raise ValueError("invalid generation parent FD") from exc

    if not stat.S_ISDIR(info.st_mode):
        raise ValueError("generation parent is not a directory")

    if info.st_uid != os.geteuid():
        raise ValueError("generation parent ownership mismatch")

    if stat.S_IMODE(info.st_mode) & 0o077:
        raise ValueError("generation parent is not private")

    return {
        "device": info.st_dev,
        "inode": info.st_ino,
        "owner_uid": info.st_uid,
        "mode": stat.S_IMODE(info.st_mode),
    }


__all__ = ["validate_private_vae_generation_parent"]
