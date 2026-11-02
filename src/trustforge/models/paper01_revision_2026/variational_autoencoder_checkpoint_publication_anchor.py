"""Read-only descriptor anchoring for VAE publication namespaces.

Anchors an existing publication root, its generation and commit
directories, and a specified generation. Does not establish a
filesystem snapshot, publication authenticity, or scientific approval.
"""

from __future__ import annotations

import os
import stat
from contextlib import contextmanager, ExitStack

from trustforge.models.paper01_revision_2026.variational_autoencoder_checkpoint_directory_anchor import (
    open_checkpoint_directory,
)
from trustforge.models.paper01_revision_2026.variational_autoencoder_checkpoint_commit_record import (
    validate_vae_generation_id,
)
from trustforge.models.paper01_revision_2026.variational_autoencoder_checkpoint_publication_layout import (
    GENERATIONS_DIRECTORY,
    COMMITS_DIRECTORY,
)


def _open_anchored_child_directory(parent_fd, name):
    """Open a real child directory relative to an anchored directory FD."""
    if (
        type(name) is not str
        or not name
        or name in (".", "..")
        or "/" in name
        or "\x00" in name
    ):
        raise ValueError("invalid publication directory basename")

    flags = (
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | os.O_CLOEXEC
    )

    try:
        fd = os.open(name, flags, dir_fd=parent_fd)
    except OSError as exc:
        raise ValueError(
            "cannot securely open publication directory"
        ) from exc

    try:
        if not stat.S_ISDIR(os.fstat(fd).st_mode):
            raise ValueError("publication entry is not a directory")
        return fd
    except BaseException:
        os.close(fd)
        raise


@contextmanager
def open_vae_publication_namespace(root, *, generation_id):
    """Yield anchored FDs for an existing publication namespace.

    Open descriptors preserve directory-inode identity even if
    directory entries change. They do not make contents immutable.
    """
    validated_id = validate_vae_generation_id(generation_id)

    with ExitStack() as stack:
        root_fd = stack.enter_context(
            open_checkpoint_directory(root)
        )

        generations_fd = _open_anchored_child_directory(
            root_fd, GENERATIONS_DIRECTORY
        )
        stack.callback(os.close, generations_fd)

        commits_fd = _open_anchored_child_directory(
            root_fd, COMMITS_DIRECTORY
        )
        stack.callback(os.close, commits_fd)

        generation_fd = _open_anchored_child_directory(
            generations_fd, validated_id
        )
        stack.callback(os.close, generation_fd)

        yield {
            "root_fd": root_fd,
            "generations_fd": generations_fd,
            "commits_fd": commits_fd,
            "generation_fd": generation_fd,
        }


__all__ = [
    "open_vae_publication_namespace",
]
