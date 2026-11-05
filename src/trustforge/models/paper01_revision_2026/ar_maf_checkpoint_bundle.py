"""Read-only AR/MAF checkpoint bundle verification.

No model construction, weight assignment, publication or execution authority.
Descriptor anchoring does not establish immutable filesystem contents.
"""

import json

from .ar_maf_checkpoint_identity import canonical_json_bytes
from .ar_maf_checkpoint_manifest import prepare_checkpoint_manifest
from .ar_maf_checkpoint_npz import MAX_ARCHIVE_BYTES
from .variational_autoencoder_checkpoint_directory_anchor import (
    open_checkpoint_directory,
    anchored_member_names,
    read_anchored_regular_file,
)

MAX_MANIFEST_BYTES = 2 * 1024 * 1024
BUNDLE_MEMBERS = frozenset({
    "manifest.json", "best_state.npz", "last_state.npz",
})


def verify_checkpoint_bundle_bytes(
    *, manifest_raw, best_raw, last_raw, recipe_raw,
    expected_identity, expected_source_commit, expected_source_files,
):
    """Verify supplied bytes against independently requested identities."""
    if type(manifest_raw) is not bytes:
        raise TypeError("manifest must be bytes")
    if not 0 < len(manifest_raw) <= MAX_MANIFEST_BYTES:
        raise ValueError("invalid manifest size")
    manifest = json.loads(manifest_raw)
    if type(manifest) is not dict:
        raise ValueError("manifest must be a JSON object")
    if manifest_raw != canonical_json_bytes(manifest) + b"\n":
        raise ValueError("noncanonical checkpoint manifest")

    if canonical_json_bytes(manifest["identity"]) != canonical_json_bytes(
        expected_identity
    ):
        raise ValueError("requested scientific identity mismatch")
    if (
        manifest["source_commit"] != expected_source_commit
        or canonical_json_bytes(manifest["source_files"])
        != canonical_json_bytes(expected_source_files)
    ):
        raise ValueError("requested source identity mismatch")

    rebuilt = prepare_checkpoint_manifest(
        identity=manifest["identity"],
        recipe_raw=recipe_raw,
        history=manifest["history"],
        specification=manifest["tensor_specification"],
        best_raw=best_raw,
        last_raw=last_raw,
        source_commit=manifest["source_commit"],
        source_files=manifest["source_files"],
    )
    if canonical_json_bytes(manifest) != canonical_json_bytes(rebuilt):
        raise ValueError("manifest differs from verified artifact evidence")

    return {
        "manifest": rebuilt,
        "best_raw": best_raw,
        "last_raw": last_raw,
    }


def read_checkpoint_bundle(
    directory, *, recipe_raw, expected_identity,
    expected_source_commit, expected_source_files,
):
    """Read exactly one complete bundle through one directory descriptor.

    Returned state bytes are the bytes verified here; restoration must consume
    them directly rather than reopen the original artifact paths.
    """
    with open_checkpoint_directory(directory) as fd:
        if frozenset(anchored_member_names(fd)) != BUNDLE_MEMBERS:
            raise ValueError("incomplete or unexpected checkpoint inventory")
        manifest_raw = read_anchored_regular_file(
            fd, "manifest.json", max_bytes=MAX_MANIFEST_BYTES,
        )
        best_raw = read_anchored_regular_file(
            fd, "best_state.npz", max_bytes=MAX_ARCHIVE_BYTES,
        )
        last_raw = read_anchored_regular_file(
            fd, "last_state.npz", max_bytes=MAX_ARCHIVE_BYTES,
        )
        result = verify_checkpoint_bundle_bytes(
            manifest_raw=manifest_raw,
            best_raw=best_raw,
            last_raw=last_raw,
            recipe_raw=recipe_raw,
            expected_identity=expected_identity,
            expected_source_commit=expected_source_commit,
            expected_source_files=expected_source_files,
        )
        if frozenset(anchored_member_names(fd)) != BUNDLE_MEMBERS:
            raise ValueError("checkpoint inventory changed during reading")
    return result
