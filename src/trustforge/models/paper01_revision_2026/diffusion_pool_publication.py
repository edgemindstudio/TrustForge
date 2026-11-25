"""Durable range publication and independent verification for Diffusion.

This module persists one authorized contiguous per-class sample range and
verifies it from disk without invoking the sampler, restoring a model or
training. Complete dataset-pool verification composes explicit range
manifests from any number of independently authorized jobs.

Storage grouping is an artifact-layout convention only. It never changes the
accepted Diffusion model-call batch size of one, the per-sample noise identity,
or the published sample values.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import math
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
    authoritative_class_mapping,
)
from trustforge.provenance import canonical_json_bytes


SCHEMA = "trustforge.paper01.diffusion-production-range.v1"
PAPER = "paper01"
PAPER_REVISION = "paper01_revision_2026"
FAMILY = "diffusion"
STORAGE_BATCH_SIZE = 8
PRIMARY_SAMPLES_PER_CLASS = 2000
CANONICAL_SHAPE = (40, 40, 3)
CANONICAL_DTYPE = "<f4"
CANONICAL_OUTPUT = {
    "layout": "NHWC",
    "dtype": "float32",
    "shape": list(CANONICAL_SHAPE),
    "range": [0.0, 1.0],
}
INDEX_CONTRACT = "contiguous_half_open_sample_index"
SCOPES = ("production", "fixture")
CHECKPOINT_ROLES = ("best",)
MANIFEST_FILENAME = "manifest.json"
PENDING_FILENAME = ".range.pending"
PUBLICATION_STATUS = "WRITTEN_VERIFIED_AND_SYNCED"
VERIFICATION_STATUS = "PUBLISHED_READBACK_VERIFIED"

ACCEPTED_SAMPLING_POLICY = {
    "generation": "conditional_ddpm_reverse",
    "timesteps": 1000,
    "beta_start": 0.0001,
    "beta_end": 0.02,
    "noise_identity": [
        "experimental_seed",
        "class_id",
        "sample_index",
        "step",
        "stream",
    ],
    "model_call_batch_size": 1,
    "class_conditioning": "one_hot_requested_class",
    "reverse_variance": "ddpm_posterior_variance",
}

HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
AUTHORIZATION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")

_MANIFEST_FIELDS = frozenset({
    "schema",
    "paper",
    "paper_revision",
    "family",
    "scope",
    "dataset_id",
    "experimental_seed",
    "attempt",
    "run_identity",
    "class_id",
    "class_label",
    "sample_start",
    "sample_stop",
    "sample_count",
    "expected_sample_count",
    "observed_sample_count",
    "index_contract",
    "storage_batch_size",
    "storage_batch_count",
    "canonical_output",
    "checkpoint",
    "synthesis_source_commit",
    "recipe_sha256",
    "sampling_policy",
    "execution",
    "batches",
    "publication_status",
    "verification_status",
})
_CHECKPOINT_FIELDS = frozenset({
    "manifest_sha256",
    "state_sha256",
    "role",
    "training_source_commit",
})
_EXECUTION_FIELDS = frozenset({
    "authorization_id",
    "slurm_job_id",
    "resources",
})
_BATCH_FIELDS = frozenset({
    "batch_index",
    "filename",
    "start_sample",
    "stop_sample",
    "sample_count",
    "dtype",
    "shape",
    "sha256",
})
_BATCH_NAME = re.compile(
    r"^class(?P<class_id>[0-9]+)-s(?P<start>[0-9]{4})-"
    r"b(?P<batch>[0-9]{3})\.npy$"
)


def _source_repository_root():
    """Return the checked-out source tree containing this verifier."""
    return Path(__file__).resolve().parents[4]


def _digest(value, *, name):
    if type(value) is not str or HEX64.fullmatch(value) is None:
        raise ValueError(name + " must be a lowercase SHA-256")
    return value


def _commit(value, *, name):
    if type(value) is not str or HEX40.fullmatch(value) is None:
        raise ValueError(name + " must be a full lowercase SHA-1")
    return value


def _exact_int(value, *, name, minimum=None, maximum=None):
    if isinstance(value, bool) or type(value) is not int:
        raise TypeError(name + " must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(name + " must be >= " + str(minimum))
    if maximum is not None and value > maximum:
        raise ValueError(name + " must be <= " + str(maximum))
    return value


def _require_regular_file(path, *, name):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(name + " must be a regular file")
    return path


def _fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_policy(policy):
    if policy != ACCEPTED_SAMPLING_POLICY:
        raise ValueError(
            "Diffusion sampling policy does not match the accepted recipe "
            "contract"
        )
    return dict(policy)


@dataclass(frozen=True)
class DiffusionRangeIdentity:
    """Explicit identity of one published contiguous sample range."""

    dataset_id: str
    experimental_seed: int
    attempt: int
    class_id: int
    class_label: str
    sample_start: int
    sample_stop: int
    checkpoint_manifest_sha256: str
    checkpoint_state_sha256: str
    checkpoint_training_source_commit: str
    synthesis_source_commit: str
    recipe_sha256: str
    sampling_policy: dict
    authorization_id: str
    slurm_job_id: str
    resources: dict
    scope: str = "production"
    checkpoint_role: str = "best"

    def __post_init__(self):
        if self.dataset_id not in DATASET_CLASS_COUNTS:
            raise ValueError("unsupported Diffusion dataset")
        seed = require_governed_experimental_seed(self.experimental_seed)
        attempt = _exact_int(self.attempt, name="attempt", minimum=1)
        class_id = _exact_int(self.class_id, name="class_id", minimum=0)
        if class_id >= DATASET_CLASS_COUNTS[self.dataset_id]:
            raise ValueError("class_id exceeds the governed class universe")
        start = _exact_int(
            self.sample_start, name="sample_start", minimum=0,
            maximum=PRIMARY_SAMPLES_PER_CLASS - 1,
        )
        stop = _exact_int(
            self.sample_stop, name="sample_stop", minimum=1,
            maximum=PRIMARY_SAMPLES_PER_CLASS,
        )
        if start >= stop:
            raise ValueError("sample_start must be less than sample_stop")
        if type(self.class_label) is not str or not self.class_label:
            raise ValueError("class_label must be a nonempty string")
        if self.checkpoint_role not in CHECKPOINT_ROLES:
            raise ValueError("Diffusion publication requires the best role")
        if self.scope not in SCOPES:
            raise ValueError("unsupported Diffusion publication scope")
        if (
            AUTHORIZATION_ID.fullmatch(self.authorization_id) is None
        ):
            raise ValueError("invalid authorization identity")
        if type(self.slurm_job_id) is not str or not self.slurm_job_id:
            raise ValueError("slurm_job_id must be a nonempty string")
        if type(self.resources) is not dict or not self.resources:
            raise ValueError("resource envelope is required")

        object.__setattr__(self, "experimental_seed", seed)
        object.__setattr__(self, "attempt", attempt)
        object.__setattr__(self, "class_id", class_id)
        object.__setattr__(self, "sample_start", start)
        object.__setattr__(self, "sample_stop", stop)
        _digest(
            self.checkpoint_manifest_sha256,
            name="checkpoint manifest sha256",
        )
        _digest(
            self.checkpoint_state_sha256,
            name="checkpoint state sha256",
        )
        _commit(
            self.checkpoint_training_source_commit,
            name="checkpoint training source commit",
        )
        _commit(
            self.synthesis_source_commit,
            name="synthesis source commit",
        )
        _digest(self.recipe_sha256, name="recipe sha256")
        _validate_policy(self.sampling_policy)
        object.__setattr__(
            self, "sampling_policy", dict(self.sampling_policy),
        )
        object.__setattr__(self, "resources", dict(self.resources))

    @property
    def run_identity(self):
        return (
            f"commit{self.synthesis_source_commit[:7]}"
            f"_attempt{self.attempt:02d}"
        )

    @property
    def sample_count(self):
        return self.sample_stop - self.sample_start

    @property
    def storage_batch_count(self):
        return math.ceil(self.sample_count / STORAGE_BATCH_SIZE)


def validate_range_images(images, *, sample_count):
    array = np.asarray(images)
    if array.shape != (sample_count, *CANONICAL_SHAPE):
        raise ValueError(
            "Diffusion range must contain exactly the declared sample count"
        )
    if array.dtype != np.float32:
        raise TypeError("Diffusion range images must be float32")
    if not np.isfinite(array).all():
        raise ValueError("Diffusion range contains non-finite values")
    if np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError("Diffusion range escaped [0,1]")
    return array


def _validate_batch(batch, *, name):
    array = np.asarray(batch)
    if (
        array.ndim != 4
        or array.shape[1:] != CANONICAL_SHAPE
        or not 1 <= array.shape[0] <= STORAGE_BATCH_SIZE
    ):
        raise ValueError(name + " has invalid canonical shape")
    if array.dtype != np.float32:
        raise TypeError(name + " must be float32")
    if not np.isfinite(array).all():
        raise ValueError(name + " contains non-finite values")
    if np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError(name + " escaped [0,1]")
    return array


def publish_diffusion_range(
    *,
    identity,
    images,
    destination,
    expected_checkpoint_manifest_sha256=None,
    expected_checkpoint_state_sha256=None,
    expected_recipe_sha256=None,
    repository_root=None,
):
    """Persist and independently verify one complete contiguous range."""
    if not isinstance(identity, DiffusionRangeIdentity):
        raise TypeError("identity must be DiffusionRangeIdentity")
    mapping = authoritative_class_mapping(
        dataset_id=identity.dataset_id,
        repository_root=(
            _source_repository_root()
            if repository_root is None
            else Path(repository_root).expanduser()
        ),
    )
    if identity.class_label != mapping[identity.class_id]:
        raise ValueError(
            "class_label does not match the authoritative governed mapping"
        )

    for value, expected, name in (
        (
            identity.checkpoint_manifest_sha256,
            expected_checkpoint_manifest_sha256,
            "checkpoint manifest",
        ),
        (
            identity.checkpoint_state_sha256,
            expected_checkpoint_state_sha256,
            "checkpoint state",
        ),
        (identity.recipe_sha256, expected_recipe_sha256, "recipe"),
    ):
        if expected is not None and value != _digest(expected, name=name):
            raise ValueError(name + " expectation mismatch")

    array = validate_range_images(
        images, sample_count=identity.sample_count,
    )

    destination = Path(destination).expanduser()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Diffusion range destination already exists")
    destination.mkdir(mode=0o700)

    records = []
    for index in range(identity.storage_batch_count):
        offset = index * STORAGE_BATCH_SIZE
        start = identity.sample_start + offset
        stop = min(
            identity.sample_start + offset + STORAGE_BATCH_SIZE,
            identity.sample_stop,
        )
        batch = _validate_batch(
            array[offset:offset + (stop - start)],
            name="Diffusion storage batch",
        )
        filename = (
            f"class{identity.class_id}-s{identity.sample_start:04d}-"
            f"b{index:03d}.npy"
        )
        buffer = io.BytesIO()
        np.save(buffer, batch, allow_pickle=False)
        raw = buffer.getvalue()
        with (destination / filename).open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        records.append({
            "batch_index": index,
            "filename": filename,
            "start_sample": start,
            "stop_sample": stop,
            "sample_count": stop - start,
            "dtype": batch.dtype.str,
            "shape": list(batch.shape),
            "sha256": hashlib.sha256(raw).hexdigest(),
        })

    for record in records:
        path = _require_regular_file(
            destination / record["filename"], name="Diffusion batch",
        )
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != record["sha256"]:
            raise RuntimeError("Diffusion batch read-back digest mismatch")
        observed = np.load(io.BytesIO(raw), allow_pickle=False)
        if not isinstance(observed, np.ndarray):
            raise RuntimeError("Diffusion batch read-back is not an ndarray")
        _validate_batch(observed, name="Diffusion batch read-back")
        offset = record["start_sample"] - identity.sample_start
        if not np.array_equal(
            observed,
            array[offset:offset + record["sample_count"]],
        ):
            raise RuntimeError("Diffusion batch read-back order mismatch")

    manifest = {
        "schema": SCHEMA,
        "paper": PAPER,
        "paper_revision": PAPER_REVISION,
        "family": FAMILY,
        "scope": identity.scope,
        "dataset_id": identity.dataset_id,
        "experimental_seed": identity.experimental_seed,
        "attempt": identity.attempt,
        "run_identity": identity.run_identity,
        "class_id": identity.class_id,
        "class_label": identity.class_label,
        "sample_start": identity.sample_start,
        "sample_stop": identity.sample_stop,
        "sample_count": identity.sample_count,
        "expected_sample_count": identity.sample_count,
        "observed_sample_count": identity.sample_count,
        "index_contract": INDEX_CONTRACT,
        "storage_batch_size": STORAGE_BATCH_SIZE,
        "storage_batch_count": identity.storage_batch_count,
        "canonical_output": dict(CANONICAL_OUTPUT),
        "checkpoint": {
            "manifest_sha256": identity.checkpoint_manifest_sha256,
            "state_sha256": identity.checkpoint_state_sha256,
            "role": identity.checkpoint_role,
            "training_source_commit":
                identity.checkpoint_training_source_commit,
        },
        "synthesis_source_commit": identity.synthesis_source_commit,
        "recipe_sha256": identity.recipe_sha256,
        "sampling_policy": dict(identity.sampling_policy),
        "execution": {
            "authorization_id": identity.authorization_id,
            "slurm_job_id": identity.slurm_job_id,
            "resources": dict(identity.resources),
        },
        "batches": records,
        "publication_status": PUBLICATION_STATUS,
        "verification_status": VERIFICATION_STATUS,
    }
    raw = canonical_json_bytes(manifest) + b"\n"
    pending = destination / PENDING_FILENAME
    final = destination / MANIFEST_FILENAME

    with pending.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(destination)
    if final.exists() or final.is_symlink():
        raise FileExistsError("Diffusion range manifest already exists")
    os.link(pending, final)
    pending.unlink()
    _fsync_directory(destination)

    return verify_diffusion_range(
        final,
        expected_checkpoint_manifest_sha256=(
            expected_checkpoint_manifest_sha256
        ),
        expected_checkpoint_state_sha256=(
            expected_checkpoint_state_sha256
        ),
        expected_recipe_sha256=expected_recipe_sha256,
        repository_root=repository_root,
    )


def _verify_manifest(manifest, *, directory, manifest_sha256, repository_root):
    if (
        type(manifest) is not dict
        or set(manifest) != _MANIFEST_FIELDS
        or manifest.get("schema") != SCHEMA
        or manifest.get("paper") != PAPER
        or manifest.get("paper_revision") != PAPER_REVISION
        or manifest.get("family") != FAMILY
        or manifest.get("scope") not in SCOPES
    ):
        raise ValueError("Diffusion range manifest identity mismatch")

    dataset_id = manifest.get("dataset_id")
    class_id = manifest.get("class_id")
    seed = manifest.get("experimental_seed")
    if dataset_id not in DATASET_CLASS_COUNTS:
        raise ValueError("Diffusion range dataset is unsupported")
    if (
        type(class_id) is not int
        or not 0 <= class_id < DATASET_CLASS_COUNTS[dataset_id]
    ):
        raise ValueError("Diffusion range class is invalid")
    seed = require_governed_experimental_seed(seed)

    mapping = authoritative_class_mapping(
        dataset_id=dataset_id,
        repository_root=repository_root,
    )
    class_label = manifest.get("class_label")
    if type(class_label) is not str or class_label != mapping[class_id]:
        raise ValueError(
            "Diffusion range class label does not match the authoritative "
            "governed mapping"
        )

    attempt = _exact_int(
        manifest.get("attempt"), name="attempt", minimum=1,
    )
    run_identity = manifest.get("run_identity")
    source_commit = _commit(
        manifest.get("synthesis_source_commit"),
        name="synthesis source commit",
    )
    expected_run = f"commit{source_commit[:7]}_attempt{attempt:02d}"
    if run_identity != expected_run:
        raise ValueError(
            "Diffusion range run identity does not match source and attempt"
        )

    start = _exact_int(
        manifest.get("sample_start"), name="sample_start", minimum=0,
        maximum=PRIMARY_SAMPLES_PER_CLASS - 1,
    )
    stop = _exact_int(
        manifest.get("sample_stop"), name="sample_stop", minimum=1,
        maximum=PRIMARY_SAMPLES_PER_CLASS,
    )
    if start >= stop:
        raise ValueError("Diffusion range sample interval is invalid")
    if (
        manifest.get("sample_count") != stop - start
        or manifest.get("expected_sample_count") != stop - start
        or manifest.get("observed_sample_count") != stop - start
        or manifest.get("index_contract") != INDEX_CONTRACT
        or manifest.get("storage_batch_size") != STORAGE_BATCH_SIZE
        or manifest.get("storage_batch_count")
        != math.ceil((stop - start) / STORAGE_BATCH_SIZE)
        or manifest.get("canonical_output") != CANONICAL_OUTPUT
    ):
        raise ValueError("Diffusion range quota or output contract mismatch")

    checkpoint = manifest.get("checkpoint")
    if (
        type(checkpoint) is not dict
        or set(checkpoint) != _CHECKPOINT_FIELDS
        or checkpoint.get("role") != "best"
    ):
        raise ValueError("Diffusion range checkpoint schema mismatch")
    manifest_digest = _digest(
        checkpoint["manifest_sha256"], name="checkpoint manifest sha256",
    )
    state_digest = _digest(
        checkpoint["state_sha256"], name="checkpoint state sha256",
    )
    training_commit = _commit(
        checkpoint["training_source_commit"],
        name="checkpoint training source commit",
    )
    recipe_sha256 = _digest(
        manifest.get("recipe_sha256"), name="recipe sha256",
    )
    if manifest.get("sampling_policy") != ACCEPTED_SAMPLING_POLICY:
        raise ValueError("Diffusion range sampling policy mismatch")

    execution = manifest.get("execution")
    if (
        type(execution) is not dict
        or set(execution) != _EXECUTION_FIELDS
    ):
        raise ValueError("Diffusion range execution schema mismatch")
    if AUTHORIZATION_ID.fullmatch(execution["authorization_id"]) is None:
        raise ValueError("Diffusion range authorization identity is invalid")
    if (
        type(execution["slurm_job_id"]) is not str
        or not execution["slurm_job_id"]
        or type(execution["resources"]) is not dict
        or not execution["resources"]
    ):
        raise ValueError("Diffusion range execution lineage is incomplete")

    if manifest.get("publication_status") != PUBLICATION_STATUS:
        raise ValueError("Diffusion range publication status mismatch")
    if manifest.get("verification_status") != VERIFICATION_STATUS:
        raise ValueError("Diffusion range verification status mismatch")

    batches = manifest.get("batches")
    expected_batches = math.ceil((stop - start) / STORAGE_BATCH_SIZE)
    if type(batches) is not list or len(batches) != expected_batches:
        raise ValueError("Diffusion range batch inventory is invalid")

    filenames = set()
    total = 0
    previous_stop = start
    for index, record in enumerate(batches):
        batch_start = start + index * STORAGE_BATCH_SIZE
        batch_stop = min(batch_start + STORAGE_BATCH_SIZE, stop)
        expected_filename = (
            f"class{class_id}-s{start:04d}-b{index:03d}.npy"
        )
        if (
            type(record) is not dict
            or set(record) != _BATCH_FIELDS
            or record["batch_index"] != index
            or record["filename"] != expected_filename
            or record["filename"] in filenames
            or record["start_sample"] != batch_start
            or record["stop_sample"] != batch_stop
            or record["sample_count"] != batch_stop - batch_start
            or record["dtype"] != CANONICAL_DTYPE
            or record["shape"] != [
                batch_stop - batch_start, *CANONICAL_SHAPE,
            ]
            or batch_start != previous_stop
        ):
            raise ValueError("invalid Diffusion batch record")
        _digest(record["sha256"], name="Diffusion batch sha256")
        filenames.add(record["filename"])
        match = _BATCH_NAME.fullmatch(record["filename"])
        if (
            match is None
            or int(match.group("class_id")) != class_id
            or int(match.group("start")) != start
            or int(match.group("batch")) != index
        ):
            raise ValueError("invalid Diffusion batch filename")
        path = _require_regular_file(
            directory / record["filename"], name="Diffusion batch",
        )
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != record["sha256"]:
            raise ValueError("Diffusion batch digest mismatch")
        batch = np.load(io.BytesIO(raw), allow_pickle=False)
        if not isinstance(batch, np.ndarray):
            raise ValueError("Diffusion batch archive must be an ndarray")
        _validate_batch(batch, name="Diffusion batch")
        if (
            batch.dtype.str != record["dtype"]
            or list(batch.shape) != record["shape"]
        ):
            raise ValueError("Diffusion batch metadata mismatch")
        total += int(batch.shape[0])
        previous_stop = batch_stop

    if total != stop - start:
        raise ValueError("Diffusion range total sample count mismatch")

    return {
        "schema": SCHEMA,
        "manifest_sha256": manifest_sha256,
        "scope": manifest["scope"],
        "dataset_id": dataset_id,
        "family": FAMILY,
        "experimental_seed": seed,
        "attempt": attempt,
        "run_identity": run_identity,
        "class_id": class_id,
        "class_label": class_label,
        "sample_start": start,
        "sample_stop": stop,
        "sample_count": total,
        "storage_batch_count": expected_batches,
        "checkpoint_manifest_sha256": manifest_digest,
        "checkpoint_state_sha256": state_digest,
        "checkpoint_training_source_commit": training_commit,
        "synthesis_source_commit": source_commit,
        "recipe_sha256": recipe_sha256,
        "authorization_id": execution["authorization_id"],
        "slurm_job_id": execution["slurm_job_id"],
        "publication_status": manifest["publication_status"],
        "authoritative_artifact": manifest["scope"] == "production",
    }


def verify_diffusion_range(
    manifest_path,
    *,
    repository_root=None,
    expected_dataset_id=None,
    expected_seed=None,
    expected_class_id=None,
    expected_class_label=None,
    expected_checkpoint_manifest_sha256=None,
    expected_checkpoint_state_sha256=None,
    expected_recipe_sha256=None,
):
    """Independently verify one persisted Diffusion range manifest."""
    manifest_path = _require_regular_file(
        manifest_path, name="Diffusion range manifest",
    )
    pending = manifest_path.parent / PENDING_FILENAME
    if pending.exists() or pending.is_symlink():
        raise ValueError(
            "Diffusion range publication is interrupted: pending marker "
            "is present"
        )
    root = (
        _source_repository_root()
        if repository_root is None
        else Path(repository_root).expanduser()
    )
    raw = manifest_path.read_bytes()
    try:
        manifest = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid Diffusion range manifest JSON") from exc
    if raw != canonical_json_bytes(manifest) + b"\n":
        raise ValueError("noncanonical Diffusion range manifest")

    report = _verify_manifest(
        manifest,
        directory=manifest_path.parent,
        manifest_sha256=hashlib.sha256(raw).hexdigest(),
        repository_root=root,
    )

    for value, expected, name in (
        (report["dataset_id"], expected_dataset_id, "dataset"),
        (report["experimental_seed"], expected_seed, "seed"),
        (report["class_id"], expected_class_id, "class"),
        (report["class_label"], expected_class_label, "class label"),
        (
            report["checkpoint_manifest_sha256"],
            expected_checkpoint_manifest_sha256,
            "checkpoint manifest",
        ),
        (
            report["checkpoint_state_sha256"],
            expected_checkpoint_state_sha256,
            "checkpoint state",
        ),
        (report["recipe_sha256"], expected_recipe_sha256, "recipe"),
    ):
        if expected is not None and value != expected:
            raise ValueError(
                "Diffusion range " + name + " expectation mismatch"
            )
    return report


def verify_diffusion_dataset_pools(
    manifest_paths,
    *,
    repository_root=None,
    expected_dataset_id=None,
    expected_seed=None,
    expected_class_mapping=None,
    expected_checkpoint_manifest_sha256=None,
    expected_checkpoint_state_sha256=None,
    expected_recipe_sha256=None,
):
    """Verify complete five-class coverage across explicit range manifests."""
    if isinstance(manifest_paths, (str, bytes)):
        raise TypeError("manifest_paths must be a path sequence")
    try:
        paths = tuple(manifest_paths)
    except TypeError as exc:
        raise TypeError("manifest_paths must be a path sequence") from exc
    if not paths:
        raise ValueError("at least one Diffusion range manifest is required")

    reports = [
        verify_diffusion_range(
            path,
            repository_root=repository_root,
            expected_dataset_id=expected_dataset_id,
            expected_seed=expected_seed,
            expected_checkpoint_manifest_sha256=(
                expected_checkpoint_manifest_sha256
            ),
            expected_checkpoint_state_sha256=(
                expected_checkpoint_state_sha256
            ),
            expected_recipe_sha256=expected_recipe_sha256,
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
        raise ValueError("Diffusion range dataset or seed identity mismatch")
    if len({report["scope"] for report in reports}) != 1:
        raise ValueError("Diffusion ranges mix artifact scopes")

    count = DATASET_CLASS_COUNTS[dataset_id]
    class_ids = {report["class_id"] for report in reports}
    if class_ids != set(range(count)):
        raise ValueError(
            "Diffusion class coverage is incomplete or duplicated"
        )

    if expected_class_mapping is not None:
        mapping = tuple(expected_class_mapping)
        if (
            len(mapping) != count
            or any(type(label) is not str or not label for label in mapping)
        ):
            raise ValueError("Diffusion expected class mapping is invalid")
        for report in reports:
            if report["class_label"] != mapping[report["class_id"]]:
                raise ValueError(
                    "Diffusion authoritative class label mismatch"
                )

    checkpoint_identities = {
        (
            report["checkpoint_manifest_sha256"],
            report["checkpoint_state_sha256"],
            report["checkpoint_training_source_commit"],
        )
        for report in reports
    }
    recipes = {report["recipe_sha256"] for report in reports}
    if len(checkpoint_identities) != 1:
        raise ValueError(
            "Diffusion ranges do not share one checkpoint identity"
        )
    if len(recipes) != 1:
        raise ValueError("Diffusion ranges do not share one recipe")

    by_class = {}
    for report in reports:
        by_class.setdefault(report["class_id"], []).append(report)

    class_reports = []
    for class_id in range(count):
        ranges = sorted(
            by_class[class_id], key=lambda item: item["sample_start"],
        )
        covered = []
        for item in ranges:
            if covered and item["sample_start"] < covered[-1][1]:
                raise ValueError(
                    "Diffusion class ranges overlap or duplicate sample "
                    "identities"
                )
            covered.append((item["sample_start"], item["sample_stop"]))
        if covered[0][0] != 0 or covered[-1][1] != PRIMARY_SAMPLES_PER_CLASS:
            raise ValueError(
                "Diffusion class does not cover sample indices 0-1999"
            )
        if any(
            right[0] != left[1]
            for left, right in zip(covered, covered[1:])
        ):
            raise ValueError(
                "Diffusion class ranges do not form a contiguous partition"
            )
        class_reports.append({
            "class_id": class_id,
            "class_label": ranges[0]["class_label"],
            "sample_count": sum(item["sample_count"] for item in ranges),
            "range_count": len(ranges),
            "manifests": [
                {
                    "manifest_sha256": item["manifest_sha256"],
                    "sample_start": item["sample_start"],
                    "sample_stop": item["sample_stop"],
                    "attempt": item["attempt"],
                    "run_identity": item["run_identity"],
                    "authorization_id": item["authorization_id"],
                    "slurm_job_id": item["slurm_job_id"],
                }
                for item in ranges
            ],
        })

    total = sum(item["sample_count"] for item in class_reports)
    if total != count * PRIMARY_SAMPLES_PER_CLASS:
        raise ValueError("Diffusion dataset-pool total sample count mismatch")

    return {
        "dataset_id": dataset_id,
        "family": FAMILY,
        "experimental_seed": seed,
        "class_count": count,
        "class_ids": list(range(count)),
        "class_labels": [item["class_label"] for item in class_reports],
        "sample_count_per_class": PRIMARY_SAMPLES_PER_CLASS,
        "total_samples": total,
        "checkpoint_manifest_sha256": next(
            iter(checkpoint_identities)
        )[0],
        "checkpoint_state_sha256": next(iter(checkpoint_identities))[1],
        "checkpoint_training_source_commit": next(
            iter(checkpoint_identities)
        )[2],
        "recipe_sha256": next(iter(recipes)),
        "authoritative_artifact": all(
            report["authoritative_artifact"] for report in reports
        ),
        "classes": class_reports,
    }


__all__ = [
    "ACCEPTED_SAMPLING_POLICY",
    "CANONICAL_OUTPUT",
    "CANONICAL_SHAPE",
    "DiffusionRangeIdentity",
    "INDEX_CONTRACT",
    "MANIFEST_FILENAME",
    "PENDING_FILENAME",
    "PRIMARY_SAMPLES_PER_CLASS",
    "PUBLICATION_STATUS",
    "SCHEMA",
    "STORAGE_BATCH_SIZE",
    "publish_diffusion_range",
    "validate_range_images",
    "verify_diffusion_dataset_pools",
    "verify_diffusion_range",
]
