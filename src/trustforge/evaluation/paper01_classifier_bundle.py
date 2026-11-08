"""Exclusive selected-classifier publication and descriptor-anchored loading.

Requires a private caller-managed parent and cooperative same-UID operation.
No provenance authentication or general filesystem crash-durability guarantee.
Created artifacts are retained on failure; no automatic rollback or overwrite.
"""

import os

from .paper01_classifier_manifest import verify_classifier_manifest
from trustforge.models.paper01_revision_2026.ar_maf_checkpoint_npz import (
    MAX_ARCHIVE_BYTES,
)
from trustforge.models.paper01_revision_2026.variational_autoencoder_checkpoint_directory_anchor import (
    open_checkpoint_directory, anchored_member_names, read_anchored_regular_file,
)
from trustforge.models.paper01_revision_2026.variational_autoencoder_checkpoint_publication_file import (
    write_exclusive_vae_artifact,
)
from trustforge.models.paper01_revision_2026.variational_autoencoder_checkpoint_publication_namespace_policy import (
    validate_private_vae_generation_parent,
)


MEMBERS = frozenset({"manifest.json", "selected_state.npz"})
MAX_MANIFEST_BYTES = 2 * 1024 * 1024


class IndeterminateClassifierPublicationError(RuntimeError):
    """Manifest writing began; inspect retained artifacts before recovery."""


def _verify(manifest_raw, state_raw, expectations):
    if type(manifest_raw) is not bytes or not 0 < len(manifest_raw) <= MAX_MANIFEST_BYTES:
        raise ValueError("invalid bounded classifier manifest")
    return verify_classifier_manifest(
        manifest_raw=manifest_raw, state_raw=state_raw, **expectations,
    )


def read_classifier_bundle(
    directory, *, recipe_raw, expected_manifest_sha256,
    expected_context, expected_selected_weights_sha256,
):
    expectations = {
        "recipe_raw": recipe_raw,
        "expected_manifest_sha256": expected_manifest_sha256,
        "expected_context": expected_context,
        "expected_selected_weights_sha256": expected_selected_weights_sha256,
    }
    with open_checkpoint_directory(directory) as fd:
        if frozenset(anchored_member_names(fd)) != MEMBERS:
            raise ValueError("incomplete or unexpected classifier inventory")
        manifest_raw = read_anchored_regular_file(
            fd, "manifest.json", max_bytes=MAX_MANIFEST_BYTES,
        )
        state_raw = read_anchored_regular_file(
            fd, "selected_state.npz", max_bytes=MAX_ARCHIVE_BYTES,
        )
        manifest = _verify(manifest_raw, state_raw, expectations)
        if frozenset(anchored_member_names(fd)) != MEMBERS:
            raise ValueError("classifier inventory changed during reading")
    return {
        "manifest": manifest,
        "manifest_raw": manifest_raw,
        "state_raw": state_raw,
    }


def publish_classifier_bundle(
    parent, generation_name, *, manifest_raw, state_raw, recipe_raw,
    expected_manifest_sha256, expected_context, expected_selected_weights_sha256,
):
    if (
        type(generation_name) is not str or not generation_name
        or generation_name in (".", "..")
        or "/" in generation_name or "\\" in generation_name
        or "\x00" in generation_name
    ):
        raise ValueError("generation name must be one basename")
    expectations = {
        "recipe_raw": recipe_raw,
        "expected_manifest_sha256": expected_manifest_sha256,
        "expected_context": expected_context,
        "expected_selected_weights_sha256": expected_selected_weights_sha256,
    }
    manifest = _verify(manifest_raw, state_raw, expectations)
    with open_checkpoint_directory(parent) as parent_fd:
        validate_private_vae_generation_parent(parent_fd)
        os.mkdir(generation_name, mode=0o700, dir_fd=parent_fd)
        os.fsync(parent_fd)
        fd = os.open(
            generation_name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=parent_fd,
        )
        try:
            opened = os.fstat(fd)

            def bound():
                current = os.stat(
                    generation_name, dir_fd=parent_fd, follow_symlinks=False,
                )
                if (
                    current.st_dev != opened.st_dev
                    or current.st_ino != opened.st_ino
                ):
                    raise RuntimeError("classifier publication directory changed")

            bound()
            write_exclusive_vae_artifact(fd, "selected_state.npz", state_raw)
            if read_anchored_regular_file(
                fd, "selected_state.npz", max_bytes=len(state_raw),
            ) != state_raw:
                raise RuntimeError("classifier state readback mismatch")
            os.fsync(fd)
            bound()
            try:
                write_exclusive_vae_artifact(fd, "manifest.json", manifest_raw)
                if read_anchored_regular_file(
                    fd, "manifest.json", max_bytes=len(manifest_raw),
                ) != manifest_raw:
                    raise RuntimeError("classifier manifest readback mismatch")
                if frozenset(anchored_member_names(fd)) != MEMBERS:
                    raise RuntimeError("unexpected publication inventory")
                os.fsync(fd)
                bound()
                os.fsync(parent_fd)
            except Exception as exc:
                raise IndeterminateClassifierPublicationError(
                    "manifest publication began; inspect retained generation"
                ) from exc
        finally:
            os.close(fd)
    return {
        "generation_name": generation_name,
        "manifest": manifest,
        "publication_status": "WRITTEN_VERIFIED_AND_SYNCED",
    }
