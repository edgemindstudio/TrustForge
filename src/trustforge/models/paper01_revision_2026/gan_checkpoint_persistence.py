"""Verified terminal checkpoint persistence for revision-2026 GAN.

The metadata file is the final publication marker.
Checkpoints are evaluation artifacts, not deterministic resume points.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

from trustforge.provenance import canonical_json_bytes, sha256_file
from trustforge.models.paper01_revision_2026.gan_checkpoint_evidence import (
    EVIDENCE_FILENAME,
    verify_gan_checkpoint_evidence,
)
from trustforge.models.paper01_revision_2026.gan_checkpoint_manifest import (
    validate_gan_checkpoint_manifest,
    DATASET_CLASS_COUNTS,
)
from trustforge.models.paper01_revision_2026.gan_model_factory import (
    GANModelConfig,
    build_revision_gan,
)


PREFIX = "model_state.ckpt"
METADATA_FILENAME = "model_metadata.json"


def _require_compute_node():
    if not os.environ.get("SLURM_JOB_ID", "").strip():
        raise RuntimeError("GAN checkpoint operations require Slurm")


def _fsync_directory(directory):
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sync_file(path):
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _required_state_files(directory):
    files = sorted(directory.glob(PREFIX + ".*"))
    names = [p.name for p in files]

    if not files or len(names) != len(set(names)):
        raise ValueError("missing or duplicate checkpoint files")

    for item in files:
        if not item.is_file() or item.is_symlink():
            raise ValueError("invalid checkpoint state file")
        if item.stat().st_size <= 0:
            raise ValueError("empty checkpoint state file")

    return files


def _verify_artifacts(directory, metadata):
    expected = metadata["state_files"]
    actual = {p.name for p in _required_state_files(directory)}

    if actual != set(expected):
        raise ValueError("checkpoint file inventory mismatch")

    for name, expected_digest in expected.items():
        item = directory / name
        if sha256_file(item) != expected_digest:
            raise ValueError(f"checkpoint integrity failure: {name}")

    return verify_gan_checkpoint_evidence(directory, metadata)


def _validate_identity(metadata, dataset_id, seed, class_mapping):
    if metadata["dataset_id"] != dataset_id:
        raise ValueError("checkpoint dataset identity mismatch")
    if metadata["experimental_seed"] != seed:
        raise ValueError("checkpoint experimental seed mismatch")
    if metadata["class_mapping"] != list(class_mapping):
        raise ValueError("checkpoint ordered class mapping mismatch")


def _checkpoint_objects(generator, discriminator, g_optimizer, d_optimizer):
    import tensorflow as tf

    tracked = {
        "generator": generator,
        "discriminator": discriminator,
        "generator_optimizer": g_optimizer,
        "discriminator_optimizer": d_optimizer,
    }

    # Keras 3 stochastic-layer seed state can be absent from the
    # TensorFlow model checkpoint object graph despite appearing in
    # model.variables. Register its underlying tf.Variable explicitly.
    for model_name, model in (
        ("generator", generator),
        ("discriminator", discriminator),
    ):
        rng_states = [
            variable
            for variable in model.variables
            if variable.name == "seed_generator_state"
        ]

        for index, variable in enumerate(rng_states):
            tracked[f"{model_name}_rng_state_{index}"] = variable.value

    checkpoint = tf.train.Checkpoint(**tracked)

    # Include TensorFlow's checkpoint save counter in both write()
    # and restore() object graphs for strict assert_consumed().
    _ = checkpoint.save_counter

    return checkpoint


def persist_terminal_gan_checkpoint(
    directory,
    *,
    dataset_id,
    experimental_seed,
    class_mapping,
    planned_generator_iterations,
    exposure_evidence,
    generator,
    discriminator,
    generator_optimizer,
    discriminator_optimizer,
):
    """Publish one complete terminal feasibility checkpoint, without overwrite."""
    _require_compute_node()

    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("checkpoint directory must already exist")

    if dataset_id not in DATASET_CLASS_COUNTS:
        raise ValueError("unsupported governed dataset")

    if len(class_mapping) != DATASET_CLASS_COUNTS[dataset_id]:
        raise ValueError("class mapping length mismatch")

    g_steps = int(generator_optimizer.iterations.numpy())
    d_steps = int(discriminator_optimizer.iterations.numpy())

    if g_steps != planned_generator_iterations:
        raise ValueError("checkpoint is not at the declared terminal iteration")
    if d_steps != 2 * g_steps:
        raise ValueError("discriminator update accounting mismatch")

    if int(generator.inputs[1].shape[-1]) != len(class_mapping):
        raise ValueError("generator class count mismatch")
    if int(discriminator.inputs[1].shape[-1]) != len(class_mapping):
        raise ValueError("discriminator class count mismatch")

    # Validate all fixed metadata fields before publishing.
    fields = {
        "schema_version": "2.0",
        "schema_type": "paper01_revision2026_conditional_gan",
        "paper_revision": "paper01_revision_2026",
        "model_family": "conditional_gan",
        "dataset_id": dataset_id,
        "experimental_seed": experimental_seed,
        "class_mapping": list(class_mapping),
        "class_count": len(class_mapping),
        "generator_iteration": g_steps,
        "discriminator_updates": d_steps,
        "planned_generator_iterations": planned_generator_iterations,
        "checkpoint_selection": "terminal_governed_iteration",
        "image_shape": [40, 40, 3],
        "latent_dim": 100,
        "learning_rate": 0.0002,
        "beta_1": 0.5,
        "class_sampling_alpha": 0.85,
        "batch_size": 256,
    }

    lock = root / ".gan_checkpoint_publication.lock"
    lock.mkdir()  # Fail if another publication owns this directory.

    published = []
    metadata_published = False

    try:
        if (root / METADATA_FILENAME).exists():
            raise FileExistsError("authoritative GAN metadata already exists")
        if list(root.glob(PREFIX + ".*")):
            raise FileExistsError("authoritative GAN state already exists")
        if (root / EVIDENCE_FILENAME).exists():
            raise FileExistsError("authoritative GAN evidence already exists")

        with tempfile.TemporaryDirectory(
            prefix=".gan_checkpoint_stage.", dir=root
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

            evidence_bytes = canonical_json_bytes(exposure_evidence)
            evidence_temp = stage / EVIDENCE_FILENAME

            with evidence_temp.open("xb") as handle:
                handle.write(evidence_bytes)
                handle.flush()
                os.fsync(handle.fileno())

            fields["exposure_evidence_file"] = EVIDENCE_FILENAME
            fields["exposure_evidence_sha256"] = sha256_file(
                evidence_temp
            )

            metadata = validate_gan_checkpoint_manifest(fields)
            _verify_artifacts(stage, metadata)
            metadata_bytes = canonical_json_bytes(metadata) + b"\n"

            metadata_temp = stage / METADATA_FILENAME
            with metadata_temp.open("xb") as handle:
                handle.write(metadata_bytes)
                handle.flush()
                os.fsync(handle.fileno())

            for item in state_files:
                destination = root / item.name
                os.link(item, destination)
                published.append(destination)

            evidence_destination = root / EVIDENCE_FILENAME
            os.link(evidence_temp, evidence_destination)
            published.append(evidence_destination)

            _fsync_directory(root)
            _verify_artifacts(root, metadata)

            # Metadata is published last and marks completion.
            os.link(metadata_temp, root / METADATA_FILENAME)
            metadata_published = True
            _fsync_directory(root)

        return metadata

    except Exception:
        if not metadata_published:
            for item in published:
                item.unlink(missing_ok=True)
            _fsync_directory(root)
        raise
    finally:
        lock.rmdir()
        _fsync_directory(root)


def load_terminal_gan_checkpoint(
    directory,
    *,
    requested_dataset_id,
    requested_experimental_seed,
    requested_class_mapping,
):
    """Verify identity and hashes before constructing/restoring any models."""
    _require_compute_node()

    root = Path(directory).expanduser().resolve()
    metadata_path = root / METADATA_FILENAME

    if not metadata_path.is_file() or metadata_path.is_symlink():
        raise FileNotFoundError("authoritative GAN metadata is missing")

    if (root / ".gan_checkpoint_publication.lock").exists():
        raise RuntimeError("checkpoint publication is incomplete")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata = validate_gan_checkpoint_manifest(metadata)

    _validate_identity(
        metadata,
        requested_dataset_id,
        requested_experimental_seed,
        requested_class_mapping,
    )
    exposure_evidence = _verify_artifacts(root, metadata)

    models = build_revision_gan(
        GANModelConfig(num_classes=metadata["class_count"])
    )

    generator = models["generator"]
    discriminator = models["discriminator"]
    g_optimizer = models["generator_optimizer"]
    d_optimizer = models["discriminator_optimizer"]

    # Materialize Adam slots before requiring complete restoration.
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
        "exposure_evidence": exposure_evidence,
        "generator": generator,
        "discriminator": discriminator,
        "generator_optimizer": g_optimizer,
        "discriminator_optimizer": d_optimizer,
    }
