"""Bounded, descriptor-based reading of checkpoint v3 artifacts.

This protects against final-component symlink traversal and several
check/read races. It does not establish immutable filesystem snapshots.
"""

import os
import stat
from pathlib import Path


def read_regular_file_securely(path, *, max_bytes):
    """Read one opened regular file without following its final symlink."""
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("max_bytes must be a positive integer")

    path = Path(path)

    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
    flags |= getattr(os, "O_CLOEXEC", 0)

    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("O_NOFOLLOW is required")

    flags |= os.O_NOFOLLOW

    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise ValueError("cannot securely open checkpoint artifact") from exc

    try:
        before = os.fstat(fd)

        if not stat.S_ISREG(before.st_mode):
            raise ValueError("checkpoint artifact is not a regular file")

        if not 0 < before.st_size <= max_bytes:
            raise ValueError("checkpoint artifact size is invalid")

        chunks = []
        remaining = max_bytes + 1

        while remaining > 0:
            chunk = os.read(fd, min(65536, remaining))

            if not chunk:
                break

            chunks.append(chunk)
            remaining -= len(chunk)

        raw = b"".join(chunks)
        after = os.fstat(fd)

        if len(raw) != before.st_size:
            raise ValueError("checkpoint artifact size changed")

        if (
            before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or before.st_ctime_ns != after.st_ctime_ns
        ):
            raise ValueError("checkpoint artifact changed during reading")

        return raw
    finally:
        os.close(fd)
