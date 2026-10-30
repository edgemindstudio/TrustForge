"""Checkpoint v3 publication with mandatory dataset provenance.

Publication is distinct from training authorization. Checkpoint
provenance records are validated and bound to metadata, but this
function does not independently verify historical training inputs.
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

from trustforge.provenance import canonical_json_bytes, sha256_file

from trustforge.models.paper01_revision_2026.gan_checkpoint_manifest import (
    DATASET_CLASS_COUNTS,
    REQUIRED_FIELDS as V2_REQUIRED_FIELDS,
)

from trustforge.models.paper01_revision_2026.gan_checkpoint_persistence import (
    PREFIX,
    METADATA_FILENAME,
    _checkpoint_objects,
    _fsync_directory,
    _require_compute_node,
    _required_state_files,
    _sync_file,
    _verify_artifacts,
)

from trustforge.models.paper01_revision_2026.gan_checkpoint_evidence import (
    EVIDENCE_FILENAME,
)

from trustforge.models.paper01_revision_2026.gan_checkpoint_v3_manifest import (
    PROVENANCE_FILENAME,
    validate_gan_checkpoint_v3_manifest,
)

from trustforge.models.paper01_revision_2026.gan_checkpoint_v3_provenance import (
    validate_gan_v3_provenance_evidence,
)

from trustforge.models.paper01_revision_2026.gan_checkpoint_v3_artifact import (
    verify_gan_checkpoint_v3_provenance_artifact,
)


from trustforge.models.paper01_revision_2026.gan_checkpoint_v3_state_integrity import (
    verify_gan_checkpoint_v3_state_files,
)

def _write_new_synced_file(path: Path, data: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _verify_unpublished_v3_artifacts(root: Path, metadata: dict) -> None:
    """Verify all required artifacts while the publication lock is held."""
    v2_view = {
        name: metadata[name]
        for name in V2_REQUIRED_FIELDS
    }
    v2_view["schema_version"] = "2.0"

    verify_gan_checkpoint_v3_state_files(
        root, metadata['state_files']
    )
    _verify_artifacts(root, v2_view)
    verify_gan_checkpoint_v3_provenance_artifact(root, metadata)


def persist_terminal_gan_checkpoint_v3(
    directory,
    *,
    dataset_id,
    experimental_seed,
    class_mapping,
    planned_generator_iterations,
    exposure_evidence,
    dataset_provenance,
    generator,
    discriminator,
    generator_optimizer,
    discriminator_optimizer,
):
    """Publish v3 artifacts atomically, without overwriting a checkpoint."""
    _require_compute_node()

    requested_root = Path(directory).expanduser()
    if requested_root.is_symlink() or not requested_root.is_dir():
        raise ValueError("checkpoint directory must be a real directory")

    root = requested_root.resolve()

    if dataset_id not in DATASET_CLASS_COUNTS:
        raise ValueError("unsupported governed dataset")

    if len(class_mapping) != DATASET_CLASS_COUNTS[dataset_id]:
        raise ValueError("class mapping length mismatch")

    validated_provenance = validate_gan_v3_provenance_evidence(
        dataset_provenance,
        expected_dataset_id=dataset_id,
    )

    if type(exposure_evidence) is not dict:
        raise TypeError("exposure evidence must be a dictionary")

    g_steps = int(generator_optimizer.iterations.numpy())
    d_steps = int(discriminator_optimizer.iterations.numpy())

    if g_steps != planned_generator_iterations:
        raise ValueError("checkpoint is not at the terminal iteration")

    if d_steps != 2 * g_steps:
        raise ValueError("discriminator update accounting mismatch")

    if int(generator.inputs[1].shape[-1]) != len(class_mapping):
        raise ValueError("generator class count mismatch")

    if int(discriminator.inputs[1].shape[-1]) != len(class_mapping):
        raise ValueError("discriminator class count mismatch")

    fields = {
        "schema_version": "3.0",
        "schema_type": "paper01_revision2026_conditional_gan",
        "paper_revision": "paper01_revision_2026",
        "model_family": "conditional_gan",
        "dataset_id": dataset_id,
        "experimental_seed": experimental_seed,
        "class_mapping": list(class_mapping),
        "class_count": len(class_mapping),
        "generator_iteration": g_steps,
        "discriminator_updates": d_steps,
        "planned_generator_iterations":
            planned_generator_iterations,
        "checkpoint_selection": "terminal_governed_iteration",
        "image_shape": [40, 40, 3],
        "latent_dim": 100,
        "learning_rate": 0.0002,
        "beta_1": 0.5,
        "class_sampling_alpha": 0.85,
        "batch_size": 256,
    }

    lock = root / ".gan_checkpoint_publication.lock"
    lock.mkdir()

    published = []
    metadata_published = False

    try:
        protected_names = (
            METADATA_FILENAME,
            EVIDENCE_FILENAME,
            PROVENANCE_FILENAME,
        )

        if any((root / name).exists() for name in protected_names):
            raise FileExistsError("checkpoint artifact already exists")

        if list(root.glob(PREFIX + ".*")):
            raise FileExistsError("checkpoint state already exists")

        with tempfile.TemporaryDirectory(
            prefix=".gan_checkpoint_v3_stage.",
            dir=root,
        ) as temporary:
            stage = Path(temporary)

            checkpoint = _checkpoint_objects(
                generator,
                discriminator,
                generator_optimizer,
                discriminator_optimizer,
            )
            checkpoint.write(str(stage / PREFIX))

            state_files = _required_state_files(stage)

            for item in state_files:
                _sync_file(item)

            fields["state_files"] = {
                item.name: sha256_file(item)
                for item in state_files
            }

            exposure_path = stage / EVIDENCE_FILENAME
            _write_new_synced_file(
                exposure_path,
                canonical_json_bytes(exposure_evidence),
            )
            fields["exposure_evidence_file"] = EVIDENCE_FILENAME
            fields["exposure_evidence_sha256"] = (
                sha256_file(exposure_path)
            )

            provenance_path = stage / PROVENANCE_FILENAME
            _write_new_synced_file(
                provenance_path,
                canonical_json_bytes(validated_provenance),
            )
            fields["dataset_provenance_file"] = PROVENANCE_FILENAME
            fields["dataset_provenance_sha256"] = (
                sha256_file(provenance_path)
            )

            metadata = validate_gan_checkpoint_v3_manifest(fields)

            # Verify both scientific evidence artifacts and state files
            # before publishing anything to the authoritative directory.
            _verify_unpublished_v3_artifacts(stage, metadata)

            metadata_path = stage / METADATA_FILENAME
            _write_new_synced_file(
                metadata_path,
                canonical_json_bytes(metadata) + b"\n",
            )

            # Publish the data first. Metadata remains unpublished.
            for item in (
                *state_files,
                exposure_path,
                provenance_path,
            ):
                destination = root / item.name
                os.link(item, destination)
                published.append(destination)

            _fsync_directory(root)

            # Complete integrity verification under the lock.
            _verify_unpublished_v3_artifacts(root, metadata)

            # Authoritative publication marker is always last.
            metadata_destination = root / METADATA_FILENAME
            os.link(metadata_path, metadata_destination)
            published.append(metadata_destination)
            _fsync_directory(root)
            metadata_published = True

            return metadata

    except Exception:
        if not metadata_published:
            for item in reversed(published):
                item.unlink(missing_ok=True)
            _fsync_directory(root)
        raise

    finally:
        lock.rmdir()
        _fsync_directory(root)
