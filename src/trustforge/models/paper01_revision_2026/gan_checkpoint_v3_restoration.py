"""Strict restoration of verified GAN checkpoint v3 bundles.

All metadata, identity, and artifact checks occur before model
construction. Restoration does not authorize training.
"""

import json
from pathlib import Path

from trustforge.provenance import canonical_json_bytes
from trustforge.models.paper01_revision_2026.gan_checkpoint_v3_secure_io import (
    read_regular_file_securely,
)

from trustforge.models.paper01_revision_2026.gan_checkpoint_persistence import (
    PREFIX,
    METADATA_FILENAME,
    _require_compute_node,
    _validate_identity,
    _checkpoint_objects,
)

from trustforge.models.paper01_revision_2026.gan_checkpoint_v3_manifest import (
    validate_gan_checkpoint_v3_manifest,
)

from trustforge.models.paper01_revision_2026.gan_checkpoint_v3_bundle import (
    verify_gan_checkpoint_v3_bundle,
)


MAX_METADATA_BYTES = 1024 * 1024


def load_terminal_gan_checkpoint_v3(
    directory,
    *,
    requested_dataset_id,
    requested_experimental_seed,
    requested_class_mapping,
):
    """Verify the complete v3 bundle before restoring TensorFlow state."""
    _require_compute_node()

    requested_root = Path(directory).expanduser()
    if requested_root.is_symlink() or not requested_root.is_dir():
        raise ValueError("invalid checkpoint directory")

    root = requested_root.resolve()
    metadata_path = root / METADATA_FILENAME

    if not metadata_path.is_file() or metadata_path.is_symlink():
        raise FileNotFoundError("authoritative GAN v3 metadata is missing")

    if (root / ".gan_checkpoint_publication.lock").exists():
        raise RuntimeError("checkpoint publication is incomplete")

    raw = read_regular_file_securely(
        metadata_path,
        max_bytes=MAX_METADATA_BYTES,
    )

    try:
        loaded = json.loads(raw)
    except (UnicodeError, ValueError) as exc:
        raise ValueError("invalid checkpoint metadata JSON") from exc

    metadata = validate_gan_checkpoint_v3_manifest(loaded)

    if raw != canonical_json_bytes(metadata) + b"\n":
        raise ValueError("checkpoint metadata is not canonical JSON")

    _validate_identity(
        metadata,
        requested_dataset_id,
        requested_experimental_seed,
        requested_class_mapping,
    )

    verified = verify_gan_checkpoint_v3_bundle(root, metadata)

    # Only now may TensorFlow models and optimizers be constructed.
    from trustforge.models.paper01_revision_2026.gan_model_factory import (
        GANModelConfig,
        build_revision_gan,
    )

    models = build_revision_gan(
        GANModelConfig(num_classes=metadata["class_count"])
    )

    generator = models["generator"]
    discriminator = models["discriminator"]
    g_optimizer = models["generator_optimizer"]
    d_optimizer = models["discriminator_optimizer"]

    g_optimizer.build(generator.trainable_variables)
    d_optimizer.build(discriminator.trainable_variables)

    checkpoint = _checkpoint_objects(
        generator,
        discriminator,
        g_optimizer,
        d_optimizer,
    )

    status = checkpoint.restore(str(root / PREFIX))
    status.assert_consumed()

    if int(g_optimizer.iterations.numpy()) != metadata["generator_iteration"]:
        raise ValueError("restored generator iteration mismatch")

    if int(d_optimizer.iterations.numpy()) != metadata["discriminator_updates"]:
        raise ValueError("restored discriminator update count mismatch")

    return {
        "metadata": metadata,
        "exposure_evidence": verified["exposure_evidence"],
        "dataset_provenance": verified["dataset_provenance"],
        "generator": generator,
        "discriminator": discriminator,
        "generator_optimizer": g_optimizer,
        "discriminator_optimizer": d_optimizer,
    }
