"""Independent verification and bounded access for a published VAE pool.

Reads and verifies the authoritative VAE pool manifest and every batch byte
without invoking the generator. Exposes the same size/identity/preflight/batch
shape used by the existing synthetic-pool consumers.

This module does not modify CombinedTrainingView or admit the pool to the
downstream evaluator. That integration is deferred to Milestone B.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat

import numpy as np

from trustforge.models.paper01_revision_2026.gan_checkpoint_manifest import (
    DATASET_CLASS_COUNTS,
)
from trustforge.models.paper01_revision_2026.synthesis_linkage import (
    CANONICAL_OUTPUT,
    LATENT_STREAM_CONTRACT,
    PRIMARY_QUOTA,
)
from trustforge.models.paper01_revision_2026.vae_cic_pool_publication import (
    FAILURE_FILENAME,
    VAE_POOL_BATCHES_PER_CLASS,
    VAE_POOL_BATCH_SIZE,
    VAE_POOL_CHECKPOINT_ROLE,
    VAE_POOL_FAMILY,
    VAE_POOL_SCHEMA,
    VAE_POOL_STATUS,
)
from trustforge.models.paper01_revision_2026.variational_autoencoder_checkpoint_commit_record import (
    validate_vae_generation_id,
)

from .paper01_inputs import MAX_BATCH_SIZE, validate_batch


HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_BATCH_FILE_BYTES = 32 * 40 * 40 * 3 * 4 + 4096

MANIFEST_FIELDS = frozenset({
    "schema",
    "dataset_id",
    "family",
    "experimental_seed",
    "class_mapping",
    "class_count",
    "checkpoint_generation_id",
    "checkpoint_manifest_sha256",
    "checkpoint_role",
    "checkpoint_run_identity",
    "checkpoint_source_commit",
    "synthesis_source_commit",
    "recipe_sha256",
    "latent_stream_contract",
    "rng",
    "model_call_batch_size",
    "quota_per_class",
    "batches_per_class",
    "complete_pool",
    "expected_total",
    "observed_total",
    "status",
    "authoritative_artifact",
    "canonical_output",
    "batches",
})
BATCH_FIELDS = frozenset({
    "class_id",
    "batch_index",
    "sample_indices",
    "filename",
    "sha256",
    "size_bytes",
    "shape",
    "dtype",
})


def _digest(value):
    if type(value) is not str or HEX64.fullmatch(value) is None:
        raise ValueError("expected SHA-256 digest")
    return value


def _commit(value):
    if type(value) is not str or HEX40.fullmatch(value) is None:
        raise ValueError("expected 40-character commit")
    return value


def _exact_int(value, name):
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    return value


def _read_file(path, limit):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= limit:
            raise ValueError("invalid bounded regular file")
        with os.fdopen(fd, "rb", closefd=False) as handle:
            raw = handle.read(limit + 1)
        if not 0 < len(raw) <= limit:
            raise ValueError("file exceeds size limit")
        return raw
    finally:
        os.close(fd)


def _reject_constant(value):
    raise ValueError(f"non-finite JSON constant: {value}")


def _reject_duplicate_keys(pairs):
    record = {}
    for key, value in pairs:
        if key in record:
            raise ValueError(f"duplicate JSON field: {key}")
        record[key] = value
    return record


def _expected_sample_indices(batch_index):
    start = batch_index * VAE_POOL_BATCH_SIZE
    stop = min(start + VAE_POOL_BATCH_SIZE, PRIMARY_QUOTA)
    return list(range(start, stop))


def _batch_filename(class_id, batch_index):
    return f"c{class_id}-b{batch_index:03d}.npy"


class VAEPoolView:
    """Verified complete VAE pool with bounded per-position access."""

    def __init__(
        self,
        manifest_path,
        *,
        manifest_sha256,
        family,
        experimental_seed,
        recipe_sha256,
        checkpoint_sha256,
    ):
        if family != VAE_POOL_FAMILY:
            raise ValueError("unsupported VAE pool family")
        if (
            type(experimental_seed) is not int
            or experimental_seed not in range(42, 49)
        ):
            raise ValueError("unsupported experimental seed")

        self._manifest_path = Path(manifest_path)
        raw = _read_file(self._manifest_path, MAX_MANIFEST_BYTES)
        if hashlib.sha256(raw).hexdigest() != _digest(manifest_sha256):
            raise ValueError("VAE pool manifest digest mismatch")

        manifest = json.loads(
            raw,
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
        if type(manifest) is not dict or set(manifest) != MANIFEST_FIELDS:
            raise ValueError("VAE pool manifest fields are invalid")

        dataset_id = manifest["dataset_id"]
        if (
            type(dataset_id) is not str
            or dataset_id not in DATASET_CLASS_COUNTS
        ):
            raise ValueError("unsupported VAE pool dataset")
        class_count = DATASET_CLASS_COUNTS[dataset_id]
        mapping = manifest["class_mapping"]
        if (
            type(mapping) is not list
            or len(mapping) != class_count
            or any(type(label) is not str or not label for label in mapping)
            or len(set(mapping)) != class_count
        ):
            raise ValueError("invalid VAE pool class mapping")

        if (
            manifest["schema"] != VAE_POOL_SCHEMA
            or manifest["family"] != VAE_POOL_FAMILY
            or _exact_int(
                manifest["experimental_seed"], "experimental_seed"
            ) != experimental_seed
            or _exact_int(
                manifest["class_count"], "class_count"
            ) != class_count
            or manifest["checkpoint_role"] != VAE_POOL_CHECKPOINT_ROLE
            or _digest(manifest["checkpoint_manifest_sha256"])
                != _digest(checkpoint_sha256)
            or _digest(manifest["recipe_sha256"]) != _digest(recipe_sha256)
            or manifest["latent_stream_contract"] != LATENT_STREAM_CONTRACT
            or _exact_int(
                manifest["model_call_batch_size"],
                "model_call_batch_size",
            ) != VAE_POOL_BATCH_SIZE
            or _exact_int(
                manifest["quota_per_class"], "quota_per_class"
            ) != PRIMARY_QUOTA
            or _exact_int(
                manifest["batches_per_class"], "batches_per_class"
            ) != VAE_POOL_BATCHES_PER_CLASS
            or manifest["complete_pool"] is not True
            or manifest["expected_total"] != class_count * PRIMARY_QUOTA
            or manifest["observed_total"] != class_count * PRIMARY_QUOTA
            or manifest["status"] != VAE_POOL_STATUS
            or manifest["authoritative_artifact"] is not True
            or manifest["canonical_output"] != CANONICAL_OUTPUT
        ):
            raise ValueError("VAE pool identity or completeness mismatch")

        validate_vae_generation_id(manifest["checkpoint_generation_id"])
        _commit(manifest["checkpoint_source_commit"])
        _commit(manifest["synthesis_source_commit"])
        if (
            type(manifest["checkpoint_run_identity"]) is not str
            or not manifest["checkpoint_run_identity"]
        ):
            raise ValueError("invalid checkpoint run identity")

        rng = manifest["rng"]
        if (
            type(rng) is not dict
            or rng.get("contract") != LATENT_STREAM_CONTRACT
            or rng.get("bit_generator") != "PCG64"
            or rng.get("latent_dimension") != 100
        ):
            raise ValueError("VAE pool RNG identity mismatch")

        records = manifest["batches"]
        expected_batches = class_count * VAE_POOL_BATCHES_PER_CLASS
        if type(records) is not list or len(records) != expected_batches:
            raise ValueError(
                f"VAE pool requires exactly {expected_batches} batch records"
            )

        indexed = {}
        filenames = set()
        for record in records:
            if type(record) is not dict or set(record) != BATCH_FIELDS:
                raise ValueError("invalid VAE batch record")
            class_id = _exact_int(record["class_id"], "class_id")
            batch_index = _exact_int(record["batch_index"], "batch_index")
            if not 0 <= class_id < class_count:
                raise ValueError("VAE batch class outside mapping")
            if not 0 <= batch_index < VAE_POOL_BATCHES_PER_CLASS:
                raise ValueError("VAE batch index outside contract")
            key = (class_id, batch_index)
            if key in indexed:
                raise ValueError("duplicate VAE batch identity")
            expected_indices = _expected_sample_indices(batch_index)
            if record["sample_indices"] != expected_indices:
                raise ValueError("VAE batch sample identity mismatch")
            filename = record["filename"]
            expected_filename = _batch_filename(class_id, batch_index)
            if filename != expected_filename:
                raise ValueError("VAE batch filename mismatch")
            if filename in filenames:
                raise ValueError("duplicate VAE batch filename")
            _digest(record["sha256"])
            _exact_int(record["size_bytes"], "size_bytes")
            expected_shape = [
                len(expected_indices),
                40,
                40,
                3,
            ]
            if record["shape"] != expected_shape:
                raise ValueError("VAE batch shape mismatch")
            if record["dtype"] != "float32":
                raise ValueError("VAE batch dtype mismatch")
            indexed[key] = record
            filenames.add(filename)

        expected_keys = {
            (class_id, batch_index)
            for class_id in range(class_count)
            for batch_index in range(VAE_POOL_BATCHES_PER_CLASS)
        }
        if set(indexed) != expected_keys:
            raise ValueError("VAE pool class/batch coverage is incomplete")

        self._dataset_id = dataset_id
        self._class_count = class_count
        self._class_mapping = tuple(mapping)
        self._experimental_seed = experimental_seed
        self._records = indexed
        self._preflight_done = False
        self._identity = {
            "schema": VAE_POOL_SCHEMA,
            "dataset_id": dataset_id,
            "family": VAE_POOL_FAMILY,
            "experimental_seed": experimental_seed,
            "class_mapping": list(mapping),
            "class_count": class_count,
            "checkpoint_generation_id": manifest["checkpoint_generation_id"],
            "checkpoint_manifest_sha256": manifest[
                "checkpoint_manifest_sha256"
            ],
            "checkpoint_role": manifest["checkpoint_role"],
            "checkpoint_run_identity": manifest["checkpoint_run_identity"],
            "checkpoint_source_commit": manifest["checkpoint_source_commit"],
            "synthesis_source_commit": manifest["synthesis_source_commit"],
            "recipe_sha256": manifest["recipe_sha256"],
            "latent_stream_contract": manifest["latent_stream_contract"],
            "model_call_batch_size": VAE_POOL_BATCH_SIZE,
            "quota_per_class": PRIMARY_QUOTA,
            "batches_per_class": VAE_POOL_BATCHES_PER_CLASS,
            "manifest_sha256": manifest_sha256,
        }

    @property
    def size(self):
        return self._class_count * PRIMARY_QUOTA

    @property
    def identity(self):
        return dict(self._identity)

    def _images(self, class_id, batch_index):
        record = self._records[(class_id, batch_index)]
        path = self._manifest_path.parent / record["filename"]
        raw = _read_file(path, MAX_BATCH_FILE_BYTES)
        if hashlib.sha256(raw).hexdigest() != record["sha256"]:
            raise ValueError("VAE batch digest mismatch")
        images = np.load(io.BytesIO(raw), allow_pickle=False)
        if not isinstance(images, np.ndarray):
            raise ValueError("VAE batch must be a numpy array")
        if (
            list(images.shape) != record["shape"]
            or str(images.dtype) != record["dtype"]
            or record["size_bytes"] != len(raw)
        ):
            raise ValueError("VAE batch metadata mismatch")
        labels = np.zeros(
            (len(images), self._class_count), dtype=np.float32
        )
        labels[:, class_id] = 1.0
        validate_batch(
            images,
            labels,
            count=len(images),
            class_count=self._class_count,
        )
        return images

    def preflight(self):
        """Stream every batch before any evaluator batch may be requested."""
        failure = self._manifest_path.parent / FAILURE_FILENAME
        if failure.exists() or failure.is_symlink():
            raise RuntimeError("VAE pool publication is incomplete")

        self._preflight_done = False
        for class_id in range(self._class_count):
            for batch_index in range(VAE_POOL_BATCHES_PER_CLASS):
                self._images(class_id, batch_index)
        self._preflight_done = True
        return {
            "identity": self.identity,
            "verified_batches": (
                self._class_count * VAE_POOL_BATCHES_PER_CLASS
            ),
            "verified_images": self.size,
        }

    def batch(self, positions):
        if not self._preflight_done:
            raise RuntimeError("complete streaming preflight required")
        if (
            not isinstance(positions, np.ndarray)
            or positions.dtype != np.int64
            or positions.ndim != 1
            or not 1 <= positions.size <= MAX_BATCH_SIZE
            or np.any(positions < 0)
            or np.any(positions >= self.size)
            or np.unique(positions).size != positions.size
        ):
            raise ValueError("invalid bounded synthetic positions")

        requested = positions.copy()
        classes = requested // PRIMARY_QUOTA
        local = requested % PRIMARY_QUOTA
        batch_indices = local // VAE_POOL_BATCH_SIZE
        offsets = local % VAE_POOL_BATCH_SIZE
        images = np.empty(
            (len(requested), 40, 40, 3), dtype=np.float32
        )
        slots = sorted(set(zip(classes.tolist(), batch_indices.tolist())))
        for class_id, batch_index in slots:
            batch_images = self._images(class_id, batch_index)
            selected = np.flatnonzero(
                (classes == class_id) & (batch_indices == batch_index)
            )
            images[selected] = batch_images[offsets[selected]]
        labels = np.eye(
            self._class_count, dtype=np.float32
        )[classes]
        validate_batch(
            images,
            labels,
            count=len(requested),
            class_count=self._class_count,
        )
        return requested, images, labels


__all__ = [
    "BATCH_FIELDS",
    "MANIFEST_FIELDS",
    "VAEPoolView",
]
