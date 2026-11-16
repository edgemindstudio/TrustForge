"""Governed CIC GAN production synthetic-pool publication.

Loads one verified terminal GAN v3 checkpoint, generates the approved
2,000 samples per class through the accepted latent stream, writes
non-overwriting batch artifacts and publishes a complete manifest only
after every batch is valid.

This module does not train, restore an unverified checkpoint, authorize
synthesis execution, publish a partial pool or select samples by quality.

The manifest's executable source commit is captured from the clean publication
checkout and records the code that generated the pool. It is distinct from the
training checkpoint run identity and is not read from or claimed to exist inside
the original checkpoint metadata.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np

from trustforge.provenance import canonical_json_bytes
from trustforge.provenance.git import capture_git_state

from .gan_checkpoint_v3_restoration import (
    load_terminal_gan_checkpoint_v3,
)
from .gan_class_mapping import resolve_gan_class_mapping
from .synthesis_linkage import (
    CANONICAL_OUTPUT,
    LATENT_STREAM_CONTRACT,
    PRIMARY_QUOTA,
    SynthesisRequest,
    governed_batch_size,
    load_synthesis_recipe,
    rng_metadata,
    synthesize_gan_batch,
)


GAN_POOL_SCHEMA = "trustforge.paper01.gan-cic-pool-manifest.v1"
GAN_POOL_FAILURE_SCHEMA = "trustforge.paper01.gan-cic-pool-failure.v1"
GAN_POOL_DATASET = "cicmaldroid2020"
GAN_POOL_FAMILY = "gan"
GAN_POOL_BATCH_SIZE = 16
GAN_POOL_BATCHES_PER_CLASS = PRIMARY_QUOTA // GAN_POOL_BATCH_SIZE
GAN_POOL_TOTAL = 5 * PRIMARY_QUOTA
GAN_POOL_CHECKPOINT_ROLE = "terminal"
GAN_POOL_STATUS = "COMPLETE"
MANIFEST_FILENAME = "manifest.json"
PENDING_MANIFEST_FILENAME = ".manifest.pending"
FAILURE_FILENAME = "FAILURE.json"


def _sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def _checkpoint_identity(metadata, checkpoint_directory):
    """Derive the checkpoint identity from verified canonical metadata.

    The v3 loader verifies the canonical metadata, every state-file SHA-256
    and the dataset-provenance artifact. Hashing the canonical metadata
    therefore commits to the verified state inventory and provenance digest
    without accepting a caller-supplied checkpoint digest.
    """
    if type(metadata) is not dict:
        raise TypeError("verified checkpoint metadata must be a mapping")

    canonical = canonical_json_bytes(metadata)

    run_identity = Path(checkpoint_directory).expanduser().name

    if (
        type(run_identity) is not str
        or not run_identity
        or run_identity in {".", ".."}
    ):
        raise ValueError("checkpoint run identity is invalid")

    return {
        "dataset_id": metadata["dataset_id"],
        "experimental_seed": metadata["experimental_seed"],
        "class_mapping": list(metadata["class_mapping"]),
        "checkpoint_role": GAN_POOL_CHECKPOINT_ROLE,
        "checkpoint_sha256": _sha256(canonical),
        "checkpoint_run_identity": run_identity,
    }


def _write_batch(destination, *, class_id, batch_index, images):
    filename = f"c{class_id}-b{batch_index:03d}.npy"
    buffer = io.BytesIO()
    np.save(buffer, images, allow_pickle=False)
    raw = buffer.getvalue()

    with (destination / filename).open("xb") as stream:
        stream.write(raw)

    return {
        "class_id": class_id,
        "batch_index": batch_index,
        "sample_indices": list(
            range(
                batch_index * GAN_POOL_BATCH_SIZE,
                (batch_index + 1) * GAN_POOL_BATCH_SIZE,
            )
        ),
        "filename": filename,
        "sha256": _sha256(raw),
        "shape": list(images.shape),
        "dtype": str(images.dtype),
    }


def _write_failure(destination, exc):
    """Record an explicit incomplete disposition; never publish a manifest."""
    try:
        with (destination / FAILURE_FILENAME).open(
            "x", encoding="utf-8"
        ) as stream:
            json.dump(
                {
                    "schema": GAN_POOL_FAILURE_SCHEMA,
                    "dataset_id": GAN_POOL_DATASET,
                    "family": GAN_POOL_FAMILY,
                    "status": "INCOMPLETE",
                    "manifest_published": False,
                    "error_type": type(exc).__name__,
                },
                stream,
                sort_keys=True,
                allow_nan=False,
            )
            stream.write("\n")
    except OSError:
        pass


def publish_gan_cic_pool(
    destination,
    *,
    checkpoint_directory,
    dataset_id,
    experimental_seed,
    repository_root,
):
    """Generate and publish one complete CIC GAN synthetic pool.

    The checkpoint is always loaded through the strict v3 restoration path;
    no generator or checkpoint digest may be supplied by the caller.
    """
    if dataset_id != GAN_POOL_DATASET:
        raise ValueError("unsupported GAN pool dataset")

    if type(experimental_seed) is not int or not 42 <= experimental_seed <= 48:
        raise ValueError("unsupported GAN pool experimental seed")

    recipe = load_synthesis_recipe(repository_root)
    class_mapping = resolve_gan_class_mapping(
        dataset_id=dataset_id, adapter=None,
    )

    loaded = load_terminal_gan_checkpoint_v3(
        checkpoint_directory,
        requested_dataset_id=dataset_id,
        requested_experimental_seed=experimental_seed,
        requested_class_mapping=list(class_mapping),
    )

    metadata = loaded["metadata"]
    generator = loaded["generator"]

    if (
        metadata.get("dataset_id") != dataset_id
        or metadata.get("experimental_seed") != experimental_seed
        or metadata.get("class_mapping") != list(class_mapping)
        or metadata.get("class_count") != len(class_mapping)
    ):
        raise ValueError("verified checkpoint identity mismatch")

    checkpoint = _checkpoint_identity(metadata, checkpoint_directory)
    checkpoint_identity = {
        "dataset_id": checkpoint["dataset_id"],
        "experimental_seed": checkpoint["experimental_seed"],
        "class_mapping": checkpoint["class_mapping"],
        "checkpoint_role": checkpoint["checkpoint_role"],
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
    }

    state = capture_git_state(repository_root)

    if (
        type(state.commit) is not str
        or len(state.commit) != 40
        or any(
            character not in "0123456789abcdef"
            for character in state.commit
        )
    ):
        raise RuntimeError("publication source commit is malformed")

    if state.dirty:
        raise RuntimeError("publication source checkout is dirty")

    executable_source_commit = state.commit

    batch_size = governed_batch_size(recipe, GAN_POOL_FAMILY)

    if batch_size != GAN_POOL_BATCH_SIZE:
        raise RuntimeError("GAN synthesis batch-size contract changed")

    if PRIMARY_QUOTA % batch_size != 0:
        raise RuntimeError("GAN synthesis quota is not batch-aligned")

    destination = Path(destination).expanduser()
    destination.mkdir(mode=0o700)

    try:
        batch_records = []

        for class_id in range(len(class_mapping)):
            for batch_index in range(GAN_POOL_BATCHES_PER_CLASS):
                sample_indices = tuple(
                    range(
                        batch_index * batch_size,
                        (batch_index + 1) * batch_size,
                    )
                )
                request = SynthesisRequest(
                    dataset_id=dataset_id,
                    family=GAN_POOL_FAMILY,
                    experimental_seed=experimental_seed,
                    class_id=class_id,
                    sample_indices=sample_indices,
                    class_mapping=class_mapping,
                    scope="production_batch",
                )
                result = synthesize_gan_batch(
                    generator=generator,
                    request=request,
                    recipe=recipe,
                    checkpoint_identity=checkpoint_identity,
                )
                images = result["images"]

                if (
                    result["record"]["observed_sample_count"]
                    != batch_size
                    or result["record"]["scope"] != "production_batch"
                ):
                    raise RuntimeError("GAN batch accounting mismatch")

                batch_records.append(
                    _write_batch(
                        destination,
                        class_id=class_id,
                        batch_index=batch_index,
                        images=images,
                    )
                )

        if len(batch_records) != 5 * GAN_POOL_BATCHES_PER_CLASS:
            raise RuntimeError("GAN pool batch inventory is incomplete")

        manifest = {
            "schema": GAN_POOL_SCHEMA,
            "dataset_id": dataset_id,
            "family": GAN_POOL_FAMILY,
            "experimental_seed": experimental_seed,
            "class_mapping": list(class_mapping),
            "class_count": len(class_mapping),
            "checkpoint_role": checkpoint["checkpoint_role"],
            "checkpoint_sha256": checkpoint["checkpoint_sha256"],
            "checkpoint_run_identity": checkpoint[
                "checkpoint_run_identity"
            ],
            "executable_source_commit": executable_source_commit,
            "recipe_sha256": recipe.sha256,
            "latent_stream_contract": LATENT_STREAM_CONTRACT,
            "rng": rng_metadata(),
            "model_call_batch_size": batch_size,
            "quota_per_class": PRIMARY_QUOTA,
            "batches_per_class": GAN_POOL_BATCHES_PER_CLASS,
            "complete_pool": True,
            "expected_total": GAN_POOL_TOTAL,
            "observed_total": GAN_POOL_TOTAL,
            "status": GAN_POOL_STATUS,
            "authoritative_artifact": True,
            "canonical_output": dict(CANONICAL_OUTPUT),
            "batches": batch_records,
        }

        pending = destination / PENDING_MANIFEST_FILENAME
        final = destination / MANIFEST_FILENAME

        with pending.open("x", encoding="utf-8") as stream:
            json.dump(
                manifest,
                stream,
                sort_keys=True,
                allow_nan=False,
            )
            stream.write("\n")

        final.hardlink_to(pending)
        pending.unlink()

        return final
    except Exception as exc:
        _write_failure(destination, exc)
        raise


__all__ = [
    "FAILURE_FILENAME",
    "GAN_POOL_BATCHES_PER_CLASS",
    "GAN_POOL_BATCH_SIZE",
    "GAN_POOL_CHECKPOINT_ROLE",
    "GAN_POOL_DATASET",
    "GAN_POOL_FAILURE_SCHEMA",
    "GAN_POOL_FAMILY",
    "GAN_POOL_SCHEMA",
    "GAN_POOL_STATUS",
    "GAN_POOL_TOTAL",
    "MANIFEST_FILENAME",
    "PENDING_MANIFEST_FILENAME",
    "publish_gan_cic_pool",
]
