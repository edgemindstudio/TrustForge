"""Governed VAE production synthetic-pool publication.

Generates and publishes the accepted complete per-class VAE allocation from
one independently verified committed BEST checkpoint. The module reuses the
accepted synthesis linkage and governed VAE loader; it does not train,
restore an arbitrary checkpoint, authorize synthesis, publish partial pools
or filter generated samples by quality.

Production synthesis remains default-deny through the source-level gate below.
The approved recipe authorization flags remain false.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np

from trustforge.provenance.git import capture_git_state

from .gan_checkpoint_manifest import DATASET_CLASS_COUNTS
from .synthesis_linkage import (
    LATENT_STREAM_CONTRACT,
    PRIMARY_QUOTA,
    CANONICAL_OUTPUT,
    SynthesisRequest,
    governed_batch_size,
    load_synthesis_recipe,
    load_vae_decoder_for_synthesis,
    rng_metadata,
    synthesize_vae_batch,
)
from .variational_autoencoder_checkpoint_commit_record import (
    validate_vae_generation_id,
)
from .variational_autoencoder_checkpoint_committed_generation import (
    verify_committed_vae_generation,
)


VAE_POOL_SCHEMA = "trustforge.paper01.vae-cic-pool-manifest.v1"
VAE_POOL_FAILURE_SCHEMA = "trustforge.paper01.vae-cic-pool-failure.v1"
VAE_POOL_FAMILY = "vae"
VAE_POOL_BATCH_SIZE = 32
VAE_POOL_BATCHES_PER_CLASS = (
    PRIMARY_QUOTA + VAE_POOL_BATCH_SIZE - 1
) // VAE_POOL_BATCH_SIZE
VAE_POOL_CHECKPOINT_ROLE = "best"
VAE_POOL_STATUS = "COMPLETE"
MANIFEST_FILENAME = "manifest.json"
PENDING_MANIFEST_FILENAME = ".manifest.pending"
FAILURE_FILENAME = "FAILURE.json"

# Source-level activation only. The accepted recipe authorization fields and
# the execution-specific researcher decision remain separate.
VAE_PRODUCTION_SYNTHESIS_AUTHORIZED = True


class VAEPoolPublicationError(RuntimeError):
    """Raised when VAE production pool publication is unavailable."""


def require_vae_production_synthesis_authorization():
    """Fail closed unless production synthesis has been separately enabled."""
    if VAE_PRODUCTION_SYNTHESIS_AUTHORIZED is not True:
        raise PermissionError(
            "VAE production synthesis is not authorized"
        )


def _sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def _require_commit(value, name):
    if (
        type(value) is not str
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(name + " must be a full lowercase SHA-1")
    return value


def _require_digest(value, name):
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(name + " must be a lowercase SHA-256")
    return value


def _checked_class_mapping(dataset_id, class_mapping):
    if type(dataset_id) is not str or dataset_id not in DATASET_CLASS_COUNTS:
        raise ValueError("unsupported governed dataset")
    if not isinstance(class_mapping, (tuple, list)):
        raise TypeError("class_mapping must be a sequence")
    mapping = tuple(class_mapping)
    count = DATASET_CLASS_COUNTS[dataset_id]
    if len(mapping) != count:
        raise ValueError("class_mapping length mismatch")
    if any(
        type(label) is not str
        or not label
        or label != label.strip()
        for label in mapping
    ):
        raise ValueError("invalid class_mapping label")
    if len(set(mapping)) != count:
        raise ValueError("duplicate class_mapping label")
    return mapping


def _checkpoint_identity(metadata, *, expected_manifest, generation_id):
    if type(metadata) is not dict:
        raise TypeError("verified checkpoint metadata must be a mapping")
    if metadata.get("checkpoint_role") != VAE_POOL_CHECKPOINT_ROLE:
        raise ValueError("VAE production synthesis requires the best checkpoint")
    config = metadata.get("effective_configuration")
    if type(config) is not dict:
        raise ValueError("verified checkpoint configuration is missing")
    source_commit = _require_commit(
        config.get("source_commit"),
        "checkpoint source commit",
    )
    run_identity = config.get("run_identity")
    if type(run_identity) is not str or not run_identity:
        raise ValueError("checkpoint run identity is invalid")
    return {
        "checkpoint_generation_id": generation_id,
        "checkpoint_manifest_sha256": expected_manifest,
        "checkpoint_role": VAE_POOL_CHECKPOINT_ROLE,
        "checkpoint_run_identity": run_identity,
        "checkpoint_source_commit": source_commit,
    }


def _write_batch(destination, *, class_id, batch_index, images, sample_indices):
    filename = f"c{class_id}-b{batch_index:03d}.npy"
    buffer = io.BytesIO()
    np.save(buffer, images, allow_pickle=False)
    raw = buffer.getvalue()
    with (destination / filename).open("xb") as stream:
        stream.write(raw)
    return {
        "class_id": class_id,
        "batch_index": batch_index,
        "sample_indices": list(sample_indices),
        "filename": filename,
        "sha256": _sha256(raw),
        "size_bytes": len(raw),
        "shape": list(images.shape),
        "dtype": str(images.dtype),
    }


def _write_failure(destination, exc):
    try:
        with (destination / FAILURE_FILENAME).open(
            "x", encoding="utf-8"
        ) as stream:
            json.dump(
                {
                    "schema": VAE_POOL_FAILURE_SCHEMA,
                    "family": VAE_POOL_FAMILY,
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


def publish_vae_pool(
    destination,
    *,
    publication_root,
    generation_id,
    expected_manifest_sha256,
    dataset_id,
    experimental_seed,
    class_mapping,
    repository_root,
    expected_synthesis_source_commit,
):
    """Generate and publish one complete VAE synthetic pool.

    Publication is default-deny and requires a separately verified committed
    BEST checkpoint. The checkpoint and synthesis source commits are recorded
    independently and need not be identical.
    """
    require_vae_production_synthesis_authorization()

    mapping = _checked_class_mapping(dataset_id, class_mapping)

    if type(experimental_seed) is not int or not 42 <= experimental_seed <= 48:
        raise ValueError("unsupported experimental seed")

    validate_vae_generation_id(generation_id)
    _require_digest(expected_manifest_sha256, "expected manifest")
    synthesis_source = _require_commit(
        expected_synthesis_source_commit,
        "expected synthesis source commit",
    )

    recipe = load_synthesis_recipe(repository_root)

    if governed_batch_size(recipe, VAE_POOL_FAMILY) != VAE_POOL_BATCH_SIZE:
        raise RuntimeError("VAE synthesis batch-size contract changed")
    if PRIMARY_QUOTA != 2000:
        raise RuntimeError("VAE primary quota contract changed")

    publication_root = Path(publication_root).expanduser().resolve()
    generation_directory = publication_root / "generations" / generation_id

    verified = verify_committed_vae_generation(
        publication_root,
        generation_id=generation_id,
        requested_dataset_id=dataset_id,
        requested_experimental_seed=experimental_seed,
        requested_class_mapping=mapping,
    )
    metadata = verified["metadata"]
    commit_record = verified["commit_record"]

    if commit_record.get("manifest_sha256") != expected_manifest_sha256:
        raise ValueError("checkpoint manifest digest mismatch")
    if metadata.get("dataset_id") != dataset_id:
        raise ValueError("checkpoint dataset mismatch")
    if metadata.get("experimental_seed") != experimental_seed:
        raise ValueError("checkpoint experimental seed mismatch")
    if metadata.get("class_mapping") != list(mapping):
        raise ValueError("checkpoint class mapping mismatch")
    if metadata.get("class_count") != len(mapping):
        raise ValueError("checkpoint class count mismatch")
    if metadata.get("architecture_version") != (
        "paper01_revision2026_conditional_vae_arch_v1"
    ):
        raise ValueError("checkpoint architecture mismatch")
    if metadata.get("observation_version") != (
        "gaussian_fixed_variance_1p0_v1"
    ):
        raise ValueError("checkpoint observation version mismatch")

    checkpoint = _checkpoint_identity(
        metadata,
        expected_manifest=expected_manifest_sha256,
        generation_id=generation_id,
    )

    state = capture_git_state(repository_root)
    if state.dirty:
        raise RuntimeError("synthesis source checkout is dirty")
    synthesis_commit = _require_commit(
        state.commit,
        "synthesis source commit",
    )
    if synthesis_commit != synthesis_source:
        raise PermissionError(
            "synthesis source does not match the expected execution source"
        )

    checkpoint_identity = {
        "dataset_id": dataset_id,
        "experimental_seed": experimental_seed,
        "class_mapping": list(mapping),
        "checkpoint_role": VAE_POOL_CHECKPOINT_ROLE,
        "checkpoint_sha256": expected_manifest_sha256,
        "source_commit": checkpoint["checkpoint_source_commit"],
    }

    load_request = SynthesisRequest(
        dataset_id=dataset_id,
        family=VAE_POOL_FAMILY,
        experimental_seed=experimental_seed,
        class_id=0,
        sample_indices=(0,),
        class_mapping=mapping,
        scope="production_batch",
    )
    decoder = load_vae_decoder_for_synthesis(
        checkpoint_directory=generation_directory,
        request=load_request,
        recipe=recipe,
        checkpoint_identity=checkpoint_identity,
    )

    destination = Path(destination).expanduser()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("synthetic pool destination already exists")

    destination.mkdir(mode=0o700)

    try:
        batch_records = []
        expected_total = len(mapping) * PRIMARY_QUOTA

        for class_id in range(len(mapping)):
            for batch_index in range(VAE_POOL_BATCHES_PER_CLASS):
                start = batch_index * VAE_POOL_BATCH_SIZE
                stop = min(start + VAE_POOL_BATCH_SIZE, PRIMARY_QUOTA)
                sample_indices = tuple(range(start, stop))
                request = SynthesisRequest(
                    dataset_id=dataset_id,
                    family=VAE_POOL_FAMILY,
                    experimental_seed=experimental_seed,
                    class_id=class_id,
                    sample_indices=sample_indices,
                    class_mapping=mapping,
                    scope="production_batch",
                )
                result = synthesize_vae_batch(
                    decoder=decoder,
                    request=request,
                    recipe=recipe,
                    checkpoint_identity=checkpoint_identity,
                )
                images = result["images"]
                if (
                    result["record"]["observed_sample_count"]
                    != len(sample_indices)
                    or result["record"]["scope"] != "production_batch"
                ):
                    raise RuntimeError("VAE batch accounting mismatch")
                batch_records.append(
                    _write_batch(
                        destination,
                        class_id=class_id,
                        batch_index=batch_index,
                        images=images,
                        sample_indices=sample_indices,
                    )
                )

        if len(batch_records) != (
            len(mapping) * VAE_POOL_BATCHES_PER_CLASS
        ):
            raise RuntimeError("VAE pool batch inventory is incomplete")

        observed_total = sum(
            len(record["sample_indices"]) for record in batch_records
        )
        if observed_total != expected_total:
            raise RuntimeError("VAE pool sample coverage is incomplete")

        manifest = {
            "schema": VAE_POOL_SCHEMA,
            "dataset_id": dataset_id,
            "family": VAE_POOL_FAMILY,
            "experimental_seed": experimental_seed,
            "class_mapping": list(mapping),
            "class_count": len(mapping),
            "checkpoint_generation_id": generation_id,
            "checkpoint_manifest_sha256": expected_manifest_sha256,
            "checkpoint_role": VAE_POOL_CHECKPOINT_ROLE,
            "checkpoint_run_identity": checkpoint["checkpoint_run_identity"],
            "checkpoint_source_commit": checkpoint["checkpoint_source_commit"],
            "synthesis_source_commit": synthesis_commit,
            "recipe_sha256": recipe.sha256,
            "latent_stream_contract": LATENT_STREAM_CONTRACT,
            "rng": rng_metadata(),
            "model_call_batch_size": VAE_POOL_BATCH_SIZE,
            "quota_per_class": PRIMARY_QUOTA,
            "batches_per_class": VAE_POOL_BATCHES_PER_CLASS,
            "complete_pool": True,
            "expected_total": expected_total,
            "observed_total": observed_total,
            "status": VAE_POOL_STATUS,
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
    "MANIFEST_FILENAME",
    "PENDING_MANIFEST_FILENAME",
    "VAE_POOL_BATCHES_PER_CLASS",
    "VAE_POOL_BATCH_SIZE",
    "VAE_POOL_CHECKPOINT_ROLE",
    "VAE_POOL_FAILURE_SCHEMA",
    "VAE_POOL_FAMILY",
    "VAE_POOL_SCHEMA",
    "VAE_POOL_STATUS",
    "VAE_PRODUCTION_SYNTHESIS_AUTHORIZED",
    "VAEPoolPublicationError",
    "publish_vae_pool",
    "require_vae_production_synthesis_authorization",
]
