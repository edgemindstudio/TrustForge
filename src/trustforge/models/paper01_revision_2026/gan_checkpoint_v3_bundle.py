"""Checkpoint v3 bundle integrity verification.

Verify state files, exposure evidence, and dataset provenance before
any TensorFlow model construction. This module never authorizes training.
"""

from pathlib import Path

from trustforge.models.paper01_revision_2026.gan_checkpoint_manifest import (
    REQUIRED_FIELDS as V2_REQUIRED_FIELDS,
)
from trustforge.models.paper01_revision_2026.gan_checkpoint_persistence import (
    _verify_artifacts as _verify_v2_artifacts,
)
from trustforge.models.paper01_revision_2026.gan_checkpoint_v3_artifact import (
    verify_gan_checkpoint_v3_provenance_artifact,
)
from trustforge.models.paper01_revision_2026.gan_checkpoint_v3_manifest import (
    validate_gan_checkpoint_v3_manifest,
)


from trustforge.models.paper01_revision_2026.gan_checkpoint_v3_state_integrity import (
    verify_gan_checkpoint_v3_state_files,
)

def verify_gan_checkpoint_v3_bundle(directory, metadata):
    """Return verified exposure and provenance evidence.

    The v2 view exists only to apply unchanged v2 artifact checks.
    It is never written to disk or treated as a v2 publication.
    """
    validated = validate_gan_checkpoint_v3_manifest(metadata)

    root = Path(directory)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("invalid checkpoint v3 directory")

    if (root / ".gan_checkpoint_publication.lock").exists():
        raise RuntimeError("checkpoint publication is incomplete")

    v2_view = {
        name: validated[name] for name in V2_REQUIRED_FIELDS
    }
    v2_view["schema_version"] = "2.0"

    verify_gan_checkpoint_v3_state_files(
        root, validated['state_files']
    )
    exposure = _verify_v2_artifacts(root, v2_view)

    provenance = verify_gan_checkpoint_v3_provenance_artifact(
        root,
        validated,
    )

    return {
        "exposure_evidence": exposure,
        "dataset_provenance": provenance,
    }
