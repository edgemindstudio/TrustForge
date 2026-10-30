"""Verify checkpoint-v3 dataset provenance artifact bytes.

Metadata and recorded evidence are validated without reading production
datasets, importing TensorFlow, or granting training authorization.
"""

import hashlib
from trustforge.models.paper01_revision_2026.gan_checkpoint_v3_secure_io import (
    read_regular_file_securely,
)
import json
from pathlib import Path

from trustforge.provenance import canonical_json_bytes
from trustforge.models.paper01_revision_2026.gan_checkpoint_v3_manifest import (
    PROVENANCE_FILENAME,
    validate_gan_checkpoint_v3_manifest,
)
from trustforge.models.paper01_revision_2026.gan_checkpoint_v3_provenance import (
    validate_gan_v3_provenance_evidence,
)

MAX_PROVENANCE_BYTES = 1024 * 1024


def verify_gan_checkpoint_v3_provenance_artifact(directory, metadata):
    """Verify exact file binding and return validated scientific evidence."""
    validated = validate_gan_checkpoint_v3_manifest(metadata)

    root = Path(directory)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("invalid checkpoint v3 directory")

    path = root / PROVENANCE_FILENAME
    if not path.is_file() or path.is_symlink():
        raise ValueError("missing or unsafe checkpoint v3 provenance")

    raw = read_regular_file_securely(
        path,
        max_bytes=MAX_PROVENANCE_BYTES,
    )

    digest = hashlib.sha256(raw).hexdigest()
    if digest != validated["dataset_provenance_sha256"]:
        raise ValueError("checkpoint v3 provenance SHA-256 mismatch")

    try:
        evidence = json.loads(raw)
    except (UnicodeError, ValueError) as exc:
        raise ValueError("invalid checkpoint v3 provenance JSON") from exc

    if type(evidence) is not dict:
        raise ValueError("checkpoint v3 provenance must be a JSON object")

    if raw != canonical_json_bytes(evidence):
        raise ValueError("checkpoint v3 provenance is not canonical JSON")

    return validate_gan_v3_provenance_evidence(
        evidence,
        expected_dataset_id=validated["dataset_id"],
    )
