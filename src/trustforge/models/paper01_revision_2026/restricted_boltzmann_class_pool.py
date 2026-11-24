"""Durable publication and independent verification of RBM class pools.

This module consumes a validated
``RBMProductionSamplingExecution`` produced by the governed production entry
point. It never invokes the sampling primitives directly and never creates,
broadens or bypasses an execution authorization.

The storage batch size of eight is a publication-only convention chosen for
consistency with the accepted GMM class-pool writer. It does not alter the
governed sampling policy, the deterministic per-sample random stream or the
2,000-sample class quota.

Complete dataset coverage is composed from independently authorized class
executions: the existing authorization record still binds exactly one class
and one sample range, so each class receives its own attempt-specific
authorization and Slurm job. Dataset-level verification requires one shared
checkpoint and sampling policy but deliberately does not require the classes
to share an authorization ID, attempt or job ID.

Nothing here claims convergence, mixing sufficiency, stationarity or semantic
class fidelity.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import re

import numpy as np

from trustforge.models.paper01_revision_2026.experimental_seed_policy import (
    require_governed_experimental_seed,
)
from trustforge.models.paper01_revision_2026.gan_checkpoint_manifest import (
    DATASET_CLASS_COUNTS,
)
from trustforge.models.paper01_revision_2026.restricted_boltzmann_production_entry import (
    RBMProductionSamplingExecution,
    RBMProductionSamplingGrant,
    authoritative_class_mapping,
)
from trustforge.models.paper01_revision_2026.restricted_boltzmann_production_sampling import (
    CANONICAL_RANGE,
    CANONICAL_SHAPE,
    PRIMARY_SAMPLES_PER_CLASS,
    PRODUCTION_BURN_IN,
    PRODUCTION_TRANSITION_COUNT,
    RBMProductionSamplingPolicy,
)
from trustforge.provenance import canonical_json_bytes


SCHEMA = "trustforge.paper01.rbm-class-pool.v1"
PAPER_REVISION = "paper01_revision_2026"
FAMILY = "restricted_boltzmann"
BATCH_SIZE = 8
SAMPLE_COUNT = PRIMARY_SAMPLES_PER_CLASS
BATCH_COUNT = SAMPLE_COUNT // BATCH_SIZE
MANIFEST_FILENAME = "manifest.json"
PENDING_FILENAME = ".manifest.pending"
PUBLICATION_STATUS = "WRITTEN_VERIFIED_AND_SYNCED"

HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
AUTHORIZATION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")

CANONICAL_OUTPUT = {
    "dtype": "float32",
    "image_shape": list(CANONICAL_SHAPE),
    "pool_shape": [SAMPLE_COUNT, *CANONICAL_SHAPE],
    "range": list(CANONICAL_RANGE),
}

SAMPLING_PROVENANCE = {
    "transition_count": PRODUCTION_TRANSITION_COUNT,
    "burn_in": PRODUCTION_BURN_IN,
    "initialization": "h0_zero_v0_bounded_exponential_p_v_given_h0",
    "class_conditioning": "fixed_requested_class",
    "chain_semantics": "independent_chain_per_sample",
    "internal_dtype": "float64",
    "canonical_shape": list(CANONICAL_SHAPE),
    "canonical_dtype": "float32",
}


def _digest(value, *, name):
    if type(value) is not str or HEX64.fullmatch(value) is None:
        raise ValueError(name + " must be a lowercase SHA-256")
    return value


def _commit(value, *, name):
    if type(value) is not str or HEX40.fullmatch(value) is None:
        raise ValueError(name + " must be a full lowercase SHA-1")
    return value


def _require_regular_file(path, *, name):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(name + " must be a regular file")
    return path


def _validate_images(images):
    array = np.asarray(images)
    if array.shape != (SAMPLE_COUNT, *CANONICAL_SHAPE):
        raise ValueError(
            "RBM class pool must contain exactly 2,000 canonical images"
        )
    if array.dtype != np.float32:
        raise TypeError("RBM class-pool images must be float32")
    if not np.isfinite(array).all():
        raise ValueError("RBM class-pool images contain non-finite values")
    if np.any(array < CANONICAL_RANGE[0]) or np.any(
        array > CANONICAL_RANGE[1]
    ):
        raise ValueError("RBM class-pool images escaped [0,1]")
    return array


def _validate_batch(batch, *, name):
    array = np.asarray(batch)
    if array.shape != (BATCH_SIZE, *CANONICAL_SHAPE):
        raise ValueError(name + " has invalid canonical batch shape")
    if array.dtype != np.float32:
        raise TypeError(name + " must be float32")
    if not np.isfinite(array).all():
        raise ValueError(name + " contains non-finite values")
    if np.any(array < CANONICAL_RANGE[0]) or np.any(
        array > CANONICAL_RANGE[1]
    ):
        raise ValueError(name + " escaped [0,1]")
    return array


def _fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _execution_identity(
    execution,
    *,
    expected_checkpoint_metadata_sha256,
    expected_checkpoint_state_sha256,
    expected_policy_digest,
):
    if not isinstance(execution, RBMProductionSamplingExecution):
        raise TypeError(
            "execution must be RBMProductionSamplingExecution"
        )
    if not isinstance(execution.grant, RBMProductionSamplingGrant):
        raise TypeError("execution grant is invalid")

    request = execution.request
    grant = execution.grant
    result = execution.result
    provenance = result.provenance

    expected_indices = tuple(range(SAMPLE_COUNT))

    if (
        request.sample_start != 0
        or request.sample_stop != SAMPLE_COUNT
        or request.sample_indices != expected_indices
    ):
        raise ValueError(
            "RBM class pool requires the complete 0..1999 sample range"
        )

    mapping = authoritative_class_mapping(
        dataset_id=request.dataset_id,
        repository_root=request.repository_root,
    )

    if request.class_id >= len(mapping):
        raise ValueError(
            "requested class exceeds the governed class universe"
        )

    class_label = mapping[request.class_id]

    if (
        grant.dataset_id != request.dataset_id
        or grant.experimental_seed != request.experimental_seed
        or grant.attempt != request.attempt
        or grant.class_id != request.class_id
        or grant.sample_start != request.sample_start
        or grant.sample_stop != request.sample_stop
        or grant.source_commit != request.expected_source_commit
        or grant.checkpoint_state_sha256
        != request.expected_checkpoint_state_sha256
    ):
        raise ValueError("execution grant identity differs from request")

    if Path(grant.checkpoint_directory) != Path(request.checkpoint_directory):
        raise ValueError(
            "execution grant checkpoint directory differs from request"
        )

    policy = RBMProductionSamplingPolicy(
        dataset_id=request.dataset_id,
        experimental_seed=request.experimental_seed,
        class_mapping=mapping,
        checkpoint_state_sha256=request.expected_checkpoint_state_sha256,
    )

    if policy.digest() != grant.policy_digest:
        raise ValueError("execution policy digest is invalid")

    if (
        provenance.dataset_id != request.dataset_id
        or provenance.experimental_seed != request.experimental_seed
        or provenance.class_id != request.class_id
        or provenance.class_label != class_label
        or provenance.sample_indices != expected_indices
        or provenance.sample_count != SAMPLE_COUNT
        or provenance.transition_count != PRODUCTION_TRANSITION_COUNT
        or provenance.burn_in != PRODUCTION_BURN_IN
        or provenance.initialization != policy.initialization
        or provenance.class_conditioning != policy.class_conditioning
        or provenance.chain_semantics != policy.chain_semantics
        or provenance.internal_dtype != policy.internal_dtype
        or tuple(provenance.canonical_shape) != CANONICAL_SHAPE
        or provenance.canonical_dtype != policy.canonical_dtype
        or provenance.checkpoint_state_sha256
        != request.expected_checkpoint_state_sha256
        or provenance.policy_digest != policy.digest()
    ):
        raise ValueError("execution sampling provenance is inconsistent")

    if not isinstance(execution.source_binding, dict):
        raise TypeError("execution source binding is invalid")

    if (
        execution.source_binding.get("requested_source_commit")
        != request.expected_source_commit
        or execution.source_binding.get("resolved_source_commit")
        != request.expected_source_commit
    ):
        raise ValueError("execution source identity is inconsistent")

    module_paths = execution.source_binding.get("module_paths")
    if not isinstance(module_paths, dict) or not module_paths:
        raise ValueError("execution module binding is missing")

    if AUTHORIZATION_ID.fullmatch(grant.authorization_id) is None:
        raise ValueError("execution authorization identity is invalid")

    if (
        not isinstance(execution.slurm_job_id, str)
        or not execution.slurm_job_id
    ):
        raise ValueError("execution Slurm job identity is invalid")

    if not isinstance(grant.resources, dict) or not grant.resources:
        raise ValueError("execution resource envelope is missing")

    metadata_sha256 = _digest(
        execution.checkpoint_metadata_sha256,
        name="checkpoint metadata sha256",
    )
    state_sha256 = _digest(
        request.expected_checkpoint_state_sha256,
        name="checkpoint state sha256",
    )
    policy_digest = policy.digest()

    for value, expected, name in (
        (
            metadata_sha256,
            expected_checkpoint_metadata_sha256,
            "checkpoint metadata",
        ),
        (
            state_sha256,
            expected_checkpoint_state_sha256,
            "checkpoint state",
        ),
        (policy_digest, expected_policy_digest, "sampling policy"),
    ):
        if expected is not None and value != _digest(expected, name=name):
            raise ValueError(name + " expectation mismatch")

    return {
        "dataset_id": request.dataset_id,
        "family": FAMILY,
        "experimental_seed": request.experimental_seed,
        "attempt": request.attempt,
        "run_identity": request.run_identity,
        "class_id": request.class_id,
        "class_label": class_label,
        "sample_start": request.sample_start,
        "sample_stop": request.sample_stop,
        "sample_count": SAMPLE_COUNT,
        "images": _validate_images(result.images),
        "checkpoint_metadata_sha256": metadata_sha256,
        "checkpoint_state_sha256": state_sha256,
        "policy_digest": policy_digest,
        "authorization_id": grant.authorization_id,
        "requested_source_commit": (
            execution.source_binding["requested_source_commit"]
        ),
        "resolved_source_commit": (
            execution.source_binding["resolved_source_commit"]
        ),
        "module_paths": dict(module_paths),
        "slurm_job_id": execution.slurm_job_id,
        "resources": dict(grant.resources),
    }


def publish_rbm_class_pool(
    *,
    execution,
    destination,
    expected_checkpoint_metadata_sha256=None,
    expected_checkpoint_state_sha256=None,
    expected_policy_digest=None,
):
    """Publish and verify one complete authorized RBM class pool.

    Refuses to overwrite an existing destination and only represents a pool
    as complete after every batch has been read back and checked.
    """
    identity = _execution_identity(
        execution,
        expected_checkpoint_metadata_sha256=(
            expected_checkpoint_metadata_sha256
        ),
        expected_checkpoint_state_sha256=(
            expected_checkpoint_state_sha256
        ),
        expected_policy_digest=expected_policy_digest,
    )

    destination = Path(destination).expanduser()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("RBM class-pool destination already exists")
    destination.mkdir(mode=0o700)

    images = identity["images"]
    class_id = identity["class_id"]

    batches = []
    for index in range(BATCH_COUNT):
        start = index * BATCH_SIZE
        stop = start + BATCH_SIZE
        batch = _validate_batch(images[start:stop], name="RBM batch")
        filename = f"class{class_id}-b{index:03d}.npy"
        buffer = io.BytesIO()
        np.save(buffer, batch, allow_pickle=False)
        raw = buffer.getvalue()
        with (destination / filename).open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        batches.append({
            "batch_index": index,
            "filename": filename,
            "start_sample": start,
            "stop_sample": stop,
            "sample_count": BATCH_SIZE,
            "sha256": hashlib.sha256(raw).hexdigest(),
        })

    for record in batches:
        path = _require_regular_file(
            destination / record["filename"], name="RBM batch file",
        )
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != record["sha256"]:
            raise RuntimeError("RBM batch read-back digest mismatch")
        observed = np.load(io.BytesIO(raw), allow_pickle=False)
        if not isinstance(observed, np.ndarray):
            raise RuntimeError("RBM batch read-back is not an ndarray")
        _validate_batch(observed, name="RBM batch read-back")
        start = record["start_sample"]
        if not np.array_equal(
            observed,
            images[start:start + BATCH_SIZE],
        ):
            raise RuntimeError("RBM batch read-back order mismatch")

    manifest = {
        "schema": SCHEMA,
        "paper_revision": PAPER_REVISION,
        "family": FAMILY,
        "dataset_id": identity["dataset_id"],
        "experimental_seed": identity["experimental_seed"],
        "attempt": identity["attempt"],
        "run_identity": identity["run_identity"],
        "class_id": class_id,
        "class_label": identity["class_label"],
        "sample_start": identity["sample_start"],
        "sample_stop": identity["sample_stop"],
        "sample_count": identity["sample_count"],
        "batch_size": BATCH_SIZE,
        "batch_count": BATCH_COUNT,
        "canonical_output": dict(CANONICAL_OUTPUT),
        "checkpoint": {
            "metadata_file_sha256": identity[
                "checkpoint_metadata_sha256"
            ],
            "state_file_sha256": identity["checkpoint_state_sha256"],
        },
        "sampling_policy_digest": identity["policy_digest"],
        "sampling_provenance": dict(SAMPLING_PROVENANCE),
        "execution": {
            "authorization_id": identity["authorization_id"],
            "requested_source_commit": (
                identity["requested_source_commit"]
            ),
            "resolved_source_commit": identity["resolved_source_commit"],
            "module_paths": identity["module_paths"],
            "slurm_job_id": identity["slurm_job_id"],
        },
        "resources": identity["resources"],
        "batches": batches,
        "publication_status": PUBLICATION_STATUS,
    }
    manifest_raw = canonical_json_bytes(manifest) + b"\n"
    pending = destination / PENDING_FILENAME
    final = destination / MANIFEST_FILENAME

    with pending.open("xb") as handle:
        handle.write(manifest_raw)
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(destination)
    if final.exists() or final.is_symlink():
        raise FileExistsError("RBM class-pool manifest already exists")
    os.link(pending, final)
    pending.unlink()
    _fsync_directory(destination)

    return verify_rbm_class_pool(
        final,
        expected_checkpoint_metadata_sha256=(
            expected_checkpoint_metadata_sha256
        ),
        expected_checkpoint_state_sha256=(
            expected_checkpoint_state_sha256
        ),
        expected_policy_digest=expected_policy_digest,
    )


def _source_repository_root():
    """Return the checked-out source tree this module belongs to.

    The governed MalNet class mapping is resolved from the source tree that
    contains the verifying code, never from a path supplied by a manifest.
    """
    return Path(__file__).resolve().parents[4]


def _verify_manifest(manifest, *, directory, manifest_sha256, repository_root):
    if (
        type(manifest) is not dict
        or manifest.get("schema") != SCHEMA
        or manifest.get("paper_revision") != PAPER_REVISION
        or manifest.get("family") != FAMILY
    ):
        raise ValueError("RBM class-pool manifest identity mismatch")

    dataset_id = manifest.get("dataset_id")
    class_id = manifest.get("class_id")
    seed = manifest.get("experimental_seed")

    if dataset_id not in DATASET_CLASS_COUNTS:
        raise ValueError("RBM class-pool dataset is unsupported")
    if (
        type(class_id) is not int
        or not 0 <= class_id < DATASET_CLASS_COUNTS[dataset_id]
    ):
        raise ValueError("RBM class-pool class is invalid")
    seed = require_governed_experimental_seed(seed)

    if (
        manifest.get("sample_start") != 0
        or manifest.get("sample_stop") != SAMPLE_COUNT
        or manifest.get("sample_count") != SAMPLE_COUNT
        or manifest.get("batch_size") != BATCH_SIZE
        or manifest.get("batch_count") != BATCH_COUNT
        or manifest.get("canonical_output") != CANONICAL_OUTPUT
    ):
        raise ValueError("RBM class-pool quota or output contract mismatch")

    class_label = manifest.get("class_label")
    if type(class_label) is not str or not class_label:
        raise ValueError("RBM class-pool class label is invalid")

    class_mapping = authoritative_class_mapping(
        dataset_id=dataset_id,
        repository_root=repository_root,
    )

    if class_label != class_mapping[class_id]:
        raise ValueError(
            "RBM class-pool class label does not match the authoritative "
            "governed mapping"
        )

    attempt = manifest.get("attempt")
    if type(attempt) is not int or attempt < 1:
        raise ValueError("RBM class-pool attempt is invalid")

    run_identity = manifest.get("run_identity")
    if type(run_identity) is not str or not run_identity:
        raise ValueError("RBM class-pool run identity is invalid")

    checkpoint = manifest.get("checkpoint")
    if (
        type(checkpoint) is not dict
        or set(checkpoint) != {
            "metadata_file_sha256",
            "state_file_sha256",
        }
    ):
        raise ValueError("RBM class-pool checkpoint schema mismatch")
    metadata_sha256 = _digest(
        checkpoint["metadata_file_sha256"],
        name="checkpoint metadata sha256",
    )
    state_sha256 = _digest(
        checkpoint["state_file_sha256"],
        name="checkpoint state sha256",
    )
    policy_digest = _digest(
        manifest.get("sampling_policy_digest"),
        name="sampling policy digest",
    )
    policy = RBMProductionSamplingPolicy(
        dataset_id=dataset_id,
        experimental_seed=seed,
        class_mapping=class_mapping,
        checkpoint_state_sha256=state_sha256,
    )

    if policy.digest() != policy_digest:
        raise ValueError(
            "RBM class-pool sampling policy digest does not match the "
            "reconstructed governed policy"
        )

    provenance = manifest.get("sampling_provenance")
    if provenance != SAMPLING_PROVENANCE:
        raise ValueError("RBM class-pool sampling policy mismatch")

    execution = manifest.get("execution")
    if (
        type(execution) is not dict
        or set(execution) != {
            "authorization_id",
            "requested_source_commit",
            "resolved_source_commit",
            "module_paths",
            "slurm_job_id",
        }
    ):
        raise ValueError("RBM class-pool execution schema mismatch")
    if AUTHORIZATION_ID.fullmatch(execution["authorization_id"]) is None:
        raise ValueError("RBM class-pool authorization identity is invalid")
    _commit(
        execution["requested_source_commit"],
        name="requested source commit",
    )
    _commit(
        execution["resolved_source_commit"],
        name="resolved source commit",
    )
    if (
        execution["requested_source_commit"]
        != execution["resolved_source_commit"]
    ):
        raise ValueError("RBM class-pool source identity is inconsistent")

    expected_run_identity = (
        f"commit{execution['requested_source_commit'][:7]}"
        f"_attempt{attempt:02d}"
    )

    if run_identity != expected_run_identity:
        raise ValueError(
            "RBM class-pool run identity does not match source commit and "
            "attempt"
        )

    if (
        type(execution["module_paths"]) is not dict
        or not execution["module_paths"]
        or type(execution["slurm_job_id"]) is not str
        or not execution["slurm_job_id"]
    ):
        raise ValueError("RBM class-pool execution lineage is incomplete")

    resources = manifest.get("resources")
    if type(resources) is not dict or not resources:
        raise ValueError("RBM class-pool resource envelope is missing")

    if manifest.get("publication_status") != PUBLICATION_STATUS:
        raise ValueError("RBM class-pool publication status mismatch")

    batches = manifest.get("batches")
    if type(batches) is not list or len(batches) != BATCH_COUNT:
        raise ValueError("RBM class-pool requires 250 ordered batches")

    filenames = set()
    total = 0
    for index, record in enumerate(batches):
        expected_filename = f"class{class_id}-b{index:03d}.npy"
        if (
            type(record) is not dict
            or set(record) != {
                "batch_index",
                "filename",
                "start_sample",
                "stop_sample",
                "sample_count",
                "sha256",
            }
            or record["batch_index"] != index
            or record["start_sample"] != index * BATCH_SIZE
            or record["stop_sample"] != (index + 1) * BATCH_SIZE
            or record["sample_count"] != BATCH_SIZE
            or record["filename"] != expected_filename
            or record["filename"] in filenames
        ):
            raise ValueError("invalid RBM batch record")
        _digest(record["sha256"], name="RBM batch sha256")
        filenames.add(record["filename"])
        path = _require_regular_file(
            directory / record["filename"], name="RBM batch file",
        )
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != record["sha256"]:
            raise ValueError("RBM batch digest mismatch")
        batch = np.load(io.BytesIO(raw), allow_pickle=False)
        if not isinstance(batch, np.ndarray):
            raise ValueError("RBM batch archive must be an ndarray")
        _validate_batch(batch, name="RBM batch")
        total += int(batch.shape[0])

    if total != SAMPLE_COUNT:
        raise ValueError("RBM class-pool total image count mismatch")

    return {
        "schema": SCHEMA,
        "manifest_sha256": manifest_sha256,
        "dataset_id": dataset_id,
        "family": FAMILY,
        "experimental_seed": seed,
        "attempt": attempt,
        "run_identity": run_identity,
        "class_id": class_id,
        "class_label": class_label,
        "sample_count": total,
        "batch_count": BATCH_COUNT,
        "checkpoint_metadata_file_sha256": metadata_sha256,
        "checkpoint_state_file_sha256": state_sha256,
        "sampling_policy_digest": policy_digest,
        "authorization_id": execution["authorization_id"],
        "requested_source_commit": execution["requested_source_commit"],
        "resolved_source_commit": execution["resolved_source_commit"],
        "slurm_job_id": execution["slurm_job_id"],
        "batch_filenames": sorted(filenames),
    }


def verify_rbm_class_pool(
    manifest_path,
    *,
    repository_root=None,
    expected_dataset_id=None,
    expected_seed=None,
    expected_class_id=None,
    expected_class_label=None,
    expected_checkpoint_metadata_sha256=None,
    expected_checkpoint_state_sha256=None,
    expected_policy_digest=None,
):
    """Independently verify one persisted complete RBM class pool.

    The class label is checked against the governed authoritative mapping and
    the sampling-policy digest is reconstructed from the recorded scientific
    identity. Every referenced batch is reread from disk and checked against
    its recorded digest, canonical shape, dtype and value range.
    """
    manifest_path = _require_regular_file(
        manifest_path, name="RBM class-pool manifest",
    )
    if repository_root is None:
        repository_root = _source_repository_root()
    else:
        repository_root = Path(repository_root).expanduser()

    raw = manifest_path.read_bytes()
    try:
        manifest = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid RBM class-pool manifest JSON") from exc
    if raw != canonical_json_bytes(manifest) + b"\n":
        raise ValueError("noncanonical RBM class-pool manifest")

    report = _verify_manifest(
        manifest,
        directory=manifest_path.parent,
        manifest_sha256=hashlib.sha256(raw).hexdigest(),
        repository_root=repository_root,
    )

    for value, expected, name in (
        (report["dataset_id"], expected_dataset_id, "dataset"),
        (report["experimental_seed"], expected_seed, "seed"),
        (report["class_id"], expected_class_id, "class"),
        (report["class_label"], expected_class_label, "class label"),
        (
            report["checkpoint_metadata_file_sha256"],
            expected_checkpoint_metadata_sha256,
            "checkpoint metadata",
        ),
        (
            report["checkpoint_state_file_sha256"],
            expected_checkpoint_state_sha256,
            "checkpoint state",
        ),
        (
            report["sampling_policy_digest"],
            expected_policy_digest,
            "sampling policy",
        ),
    ):
        if expected is not None and value != expected:
            raise ValueError(
                "RBM class-pool " + name + " expectation mismatch"
            )

    return report


def verify_rbm_dataset_pools(
    manifest_paths,
    *,
    repository_root=None,
    expected_dataset_id=None,
    expected_seed=None,
    expected_class_mapping=None,
    expected_checkpoint_metadata_sha256=None,
    expected_checkpoint_state_sha256=None,
    expected_policy_digest=None,
):
    """Verify complete class coverage for one dataset/seed RBM pool.

    The two MaleX class pools may be produced by different authorizations,
    attempts and Slurm jobs; only the shared checkpoint and sampling policy
    must agree. Every class label and policy digest is independently checked
    by the class-level verifier.
    """
    if isinstance(manifest_paths, (str, bytes)):
        raise TypeError("manifest_paths must be a path sequence")
    try:
        paths = tuple(manifest_paths)
    except TypeError as exc:
        raise TypeError("manifest_paths must be a path sequence") from exc
    if not paths:
        raise ValueError("at least one RBM class-pool manifest is required")

    reports = [
        verify_rbm_class_pool(
            path,
            repository_root=repository_root,
            expected_dataset_id=expected_dataset_id,
            expected_seed=expected_seed,
            expected_checkpoint_metadata_sha256=(
                expected_checkpoint_metadata_sha256
            ),
            expected_checkpoint_state_sha256=(
                expected_checkpoint_state_sha256
            ),
            expected_policy_digest=expected_policy_digest,
        )
        for path in paths
    ]

    dataset_id = reports[0]["dataset_id"]
    seed = reports[0]["experimental_seed"]

    if any(
        report["dataset_id"] != dataset_id
        or report["experimental_seed"] != seed
        for report in reports
    ):
        raise ValueError("RBM class-pool dataset or seed identity mismatch")

    class_ids = [report["class_id"] for report in reports]
    count = DATASET_CLASS_COUNTS[dataset_id]

    if (
        len(set(class_ids)) != len(class_ids)
        or set(class_ids) != set(range(count))
    ):
        raise ValueError(
            "RBM class-pool coverage is incomplete or duplicated"
        )

    if expected_class_mapping is not None:
        mapping = tuple(expected_class_mapping)
        if (
            len(mapping) != count
            or any(type(label) is not str or not label for label in mapping)
        ):
            raise ValueError("RBM expected class mapping is invalid")
        for report in reports:
            if report["class_label"] != mapping[report["class_id"]]:
                raise ValueError(
                    "RBM class-pool authoritative label mismatch"
                )

    checkpoint_pairs = {
        (
            report["checkpoint_metadata_file_sha256"],
            report["checkpoint_state_file_sha256"],
        )
        for report in reports
    }
    policies = {report["sampling_policy_digest"] for report in reports}

    if len(checkpoint_pairs) != 1:
        raise ValueError("RBM class pools do not share one checkpoint")
    if len(policies) != 1:
        raise ValueError("RBM class pools do not share one sampling policy")

    total_samples = sum(report["sample_count"] for report in reports)
    if total_samples != count * SAMPLE_COUNT:
        raise ValueError("RBM dataset-pool total sample count mismatch")

    ordered = sorted(reports, key=lambda report: report["class_id"])

    return {
        "dataset_id": dataset_id,
        "family": FAMILY,
        "experimental_seed": seed,
        "class_count": count,
        "class_ids": [report["class_id"] for report in ordered],
        "class_labels": [report["class_label"] for report in ordered],
        "sample_count_per_class": SAMPLE_COUNT,
        "total_samples": total_samples,
        "checkpoint_metadata_file_sha256": ordered[0][
            "checkpoint_metadata_file_sha256"
        ],
        "checkpoint_state_file_sha256": ordered[0][
            "checkpoint_state_file_sha256"
        ],
        "sampling_policy_digest": ordered[0]["sampling_policy_digest"],
        "manifests": ordered,
    }


__all__ = [
    "BATCH_COUNT",
    "BATCH_SIZE",
    "CANONICAL_OUTPUT",
    "FAMILY",
    "MANIFEST_FILENAME",
    "PENDING_FILENAME",
    "PUBLICATION_STATUS",
    "SAMPLE_COUNT",
    "SCHEMA",
    "publish_rbm_class_pool",
    "verify_rbm_class_pool",
    "verify_rbm_dataset_pools",
]
