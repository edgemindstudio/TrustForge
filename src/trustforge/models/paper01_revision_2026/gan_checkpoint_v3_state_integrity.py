"""Descriptor-based checkpoint v3 state-file integrity checks.

Protects against final-component symlinks and several read-time races.
Does not guarantee immutability between verification and TF restore.
"""

import hashlib
import os
import stat
from pathlib import Path

from trustforge.models.paper01_revision_2026.gan_checkpoint_manifest import (
    STATE_FILE_PATTERN,
)

PREFIX = "model_state.ckpt."


def sha256_state_file_securely(path):
    """Hash a single opened regular file using its file descriptor."""
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("O_NOFOLLOW is required")

    flags = os.O_RDONLY | os.O_NOFOLLOW
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)

    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise ValueError("unsafe checkpoint state file") from exc

    try:
        before = os.fstat(fd)

        if not stat.S_ISREG(before.st_mode):
            raise ValueError("checkpoint state must be a regular file")

        if before.st_size <= 0:
            raise ValueError("empty checkpoint state file")

        digest = hashlib.sha256()
        length = 0

        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            length += len(chunk)

        after = os.fstat(fd)

        if length != before.st_size:
            raise ValueError("checkpoint state size changed during hashing")

        if (
            before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or before.st_ctime_ns != after.st_ctime_ns
        ):
            raise ValueError("checkpoint state changed during hashing")

        return digest.hexdigest()
    finally:
        os.close(fd)


def verify_gan_checkpoint_v3_state_files(directory, state_files):
    """Verify the complete state-file inventory and exact SHA-256s."""
    root = Path(directory)

    if not root.is_dir() or root.is_symlink():
        raise ValueError("invalid checkpoint state directory")

    if type(state_files) is not dict or not state_files:
        raise ValueError("invalid state-file inventory")

    if any(
        type(name) is not str
        or STATE_FILE_PATTERN.fullmatch(name) is None
        for name in state_files
    ):
        raise ValueError("invalid state-file name")

    actual = {
        path.name
        for path in root.iterdir()
        if path.name.startswith(PREFIX)
    }

    if actual != set(state_files):
        raise ValueError("checkpoint file inventory mismatch")

    for name, expected in state_files.items():
        if (
            type(expected) is not str
            or len(expected) != 64
            or any(c not in "0123456789abcdef" for c in expected)
        ):
            raise ValueError("invalid state-file SHA-256")

        actual_digest = sha256_state_file_securely(root / name)

        if actual_digest != expected:
            raise ValueError(f"checkpoint integrity failure: {name}")

    return True
