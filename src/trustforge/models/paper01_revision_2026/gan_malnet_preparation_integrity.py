"""Fail-closed MalNet preparation integrity verification.

Standard-library only. Full artifact hashing belongs on a compute node.
Does not authenticate original image bytes or authorize GAN training.
"""

import hashlib
import json
import re
from pathlib import Path


BASELINE_SHA256 = (
    "a66be1309058de641f120fa96909cf151a565db2912f465a84797db683655e7f"
)
MANIFEST_SHA256 = (
    "182eabd0c5beb36cda1f27c02a704a12996cb74d7122456d3ebe2a884e084cc8"
)
PROVIDER_ARCHIVE_SHA256 = (
    "74e3615aa848c27dc06bb9ea2ae53e415f4b842fdce8dc45ba50849db8851492"
)
SPLIT_COUNTS = {
    "train": 883416,
    "validation": 126202,
    "test": 252406,
}
SHA_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path):
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"missing or unsafe MalNet artifact: {path}")


def verify_malnet_preparation_integrity(
    *, preparation_dir, baseline_path
):
    """Verify fixed baseline, manifest, and six artifact hashes.

    A successful result establishes file consistency at verification
    time, not upstream archive authenticity or image-byte integrity.
    """
    root = Path(preparation_dir).expanduser()
    baseline_file = Path(baseline_path).expanduser()
    manifest_file = root / "manifest.json"

    if not root.is_dir() or root.is_symlink():
        raise ValueError("missing or unsafe preparation directory")

    _regular_file(baseline_file)
    _regular_file(manifest_file)

    if _sha256_file(baseline_file) != BASELINE_SHA256:
        raise ValueError("MalNet governed baseline SHA-256 mismatch")

    if _sha256_file(manifest_file) != MANIFEST_SHA256:
        raise ValueError("MalNet preparation manifest SHA-256 mismatch")

    baseline = json.loads(baseline_file.read_bytes())
    manifest = json.loads(manifest_file.read_bytes())

    if not isinstance(baseline, dict) or not isinstance(manifest, dict):
        raise TypeError("MalNet records must be JSON objects")

    if (
        baseline.get("dataset_id") != "malnet_image"
        or baseline.get("preparation") != "type-1.0-prepared"
        or manifest.get("dataset") != "MalNet-Image"
        or manifest.get("task") != "type-level classification"
    ):
        raise ValueError("MalNet preparation identity mismatch")

    if (
        baseline.get("provider_archive_sha256_declared")
        != PROVIDER_ARCHIVE_SHA256
        or manifest.get("provider_archive_sha256")
        != PROVIDER_ARCHIVE_SHA256
    ):
        raise ValueError("MalNet provider archive declaration mismatch")

    baseline_splits = baseline.get("splits")
    manifest_splits = manifest.get("splits")

    if (
        not isinstance(baseline_splits, dict)
        or not isinstance(manifest_splits, dict)
        or set(baseline_splits) != set(SPLIT_COUNTS)
        or set(manifest_splits) != set(SPLIT_COUNTS)
    ):
        raise ValueError("MalNet split inventory mismatch")

    verified = {}

    for split, expected_rows in SPLIT_COUNTS.items():
        b = baseline_splits[split]
        m = manifest_splits[split]

        if not isinstance(b, dict) or not isinstance(m, dict):
            raise TypeError("MalNet split records must be objects")

        for actual in (
            b.get("rows"),
            m.get("provider_rows"),
            m.get("prepared_rows"),
        ):
            if type(actual) is not int or actual != expected_rows:
                raise ValueError(f"MalNet {split} row-count mismatch")

        artifacts = (
            (
                f"{split}.txt",
                "prepared_text_file",
                "prepared_text_sha256",
            ),
            (
                f"{split}_offsets.npy",
                "offsets_file",
                "offsets_sha256",
            ),
        )

        for expected_name, filename_field, digest_field in artifacts:
            if m.get(filename_field) != expected_name:
                raise ValueError(
                    f"MalNet {split} unsafe artifact filename"
                )

            expected_digest = b.get(digest_field)

            if (
                type(expected_digest) is not str
                or SHA_PATTERN.fullmatch(expected_digest) is None
                or m.get(digest_field) != expected_digest
            ):
                raise ValueError(
                    f"MalNet {split} fingerprint contract mismatch"
                )

            artifact = root / expected_name
            _regular_file(artifact)

            if _sha256_file(artifact) != expected_digest:
                raise ValueError(
                    f"MalNet {split} artifact SHA-256 mismatch: "
                    f"{expected_name}"
                )

            verified[expected_name] = expected_digest

    if len(verified) != 6:
        raise RuntimeError("incomplete MalNet verification inventory")

    return {
        "dataset_id": "malnet_image",
        "baseline_sha256": BASELINE_SHA256,
        "manifest_sha256": MANIFEST_SHA256,
        "verified_artifacts": verified,
        "verification_scope": "six_prepared_artifacts_and_manifest",
        "original_image_bytes_verified": False,
        "provider_archive_rehashed": False,
    }
