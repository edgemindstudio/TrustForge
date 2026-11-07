"""Exclusive AR/MAF bundle publication mechanics.

Requires a private caller-managed parent and cooperative same-UID operation.
No training authorization, origin authentication, automatic recovery or
filesystem crash-durability guarantee is established.
"""

import os

from .ar_maf_checkpoint_bundle import verify_checkpoint_bundle_bytes
from .variational_autoencoder_checkpoint_directory_anchor import (
    open_checkpoint_directory, read_anchored_regular_file,
)
from .variational_autoencoder_checkpoint_publication_file import (
    write_exclusive_vae_artifact,
)
from .variational_autoencoder_checkpoint_publication_namespace_policy import (
    validate_private_vae_generation_parent,
)


class IndeterminateCheckpointPublicationError(RuntimeError):
    """Manifest writing began; inspect the destination before any recovery."""


def publish_checkpoint_bundle(
    parent, generation_name, *, manifest_raw, best_raw, last_raw,
    recipe_raw, expected_identity,
    expected_source_commit, expected_source_files,
    expected_fitted_state_sha256=None,
):
    """Publish a new bundle without overwriting any existing generation.

    Exceptions retain created files. After manifest writing begins, failures
    are explicitly indeterminate: the bundle may already be readable.
    """
    if (
        type(generation_name) is not str or not generation_name
        or generation_name in (".", "..")
        or "/" in generation_name or "\\" in generation_name
        or "\x00" in generation_name
    ):
        raise ValueError("generation name must be a single basename")

    verified = verify_checkpoint_bundle_bytes(
        manifest_raw=manifest_raw, best_raw=best_raw, last_raw=last_raw,
        recipe_raw=recipe_raw, expected_identity=expected_identity,
        expected_source_commit=expected_source_commit,
        expected_source_files=expected_source_files,
        expected_fitted_state_sha256=expected_fitted_state_sha256,
    )

    with open_checkpoint_directory(parent) as parent_fd:
        validate_private_vae_generation_parent(parent_fd)
        os.mkdir(generation_name, mode=0o700, dir_fd=parent_fd)
        os.fsync(parent_fd)

        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        fd = os.open(generation_name, flags, dir_fd=parent_fd)
        try:
            opened = os.fstat(fd)

            def require_bound_directory():
                current = os.stat(
                    generation_name, dir_fd=parent_fd,
                    follow_symlinks=False,
                )
                if (
                    current.st_dev != opened.st_dev
                    or current.st_ino != opened.st_ino
                ):
                    raise RuntimeError("publication directory identity changed")

            require_bound_directory()
            for name, raw in (
                ("best_state.npz", best_raw),
                ("last_state.npz", last_raw),
            ):
                write_exclusive_vae_artifact(fd, name, raw)
                observed = read_anchored_regular_file(
                    fd, name, max_bytes=len(raw),
                )
                if observed != raw:
                    raise RuntimeError("published state readback mismatch")
            os.fsync(fd)
            require_bound_directory()

            try:
                write_exclusive_vae_artifact(fd, "manifest.json", manifest_raw)
                observed = read_anchored_regular_file(
                    fd, "manifest.json", max_bytes=len(manifest_raw),
                )
                if observed != manifest_raw:
                    raise RuntimeError("published manifest readback mismatch")
                os.fsync(fd)
                require_bound_directory()
                os.fsync(parent_fd)
            except Exception as exc:
                raise IndeterminateCheckpointPublicationError(
                    "manifest publication began; inspect retained generation"
                ) from exc
        finally:
            os.close(fd)

    return {
        "generation_name": generation_name,
        "manifest": verified["manifest"],
        "publication_status": "WRITTEN_VERIFIED_AND_SYNCED",
    }
