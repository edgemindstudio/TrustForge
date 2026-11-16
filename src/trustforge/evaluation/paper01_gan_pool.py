"""Bounded CIC GAN synthetic-pool view for the accepted evaluator.

Verifies a complete GAN pool manifest and every batch byte before the
combined real-plus-synthetic training view may request images. It does not
authenticate generator lineage beyond the recorded manifest/checkpoint
identity, authorize evaluation or alter classifier/metric policy.
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

from .paper01_inputs import MAX_BATCH_SIZE, validate_batch
from trustforge.models.paper01_revision_2026.gan_cic_pool_publication import (
    FAILURE_FILENAME,
    GAN_POOL_BATCHES_PER_CLASS,
    GAN_POOL_BATCH_SIZE,
    GAN_POOL_CHECKPOINT_ROLE,
    GAN_POOL_DATASET,
    GAN_POOL_FAMILY,
    GAN_POOL_SCHEMA,
    GAN_POOL_TOTAL,
    MANIFEST_FILENAME,
    PENDING_MANIFEST_FILENAME,
)
from trustforge.models.paper01_revision_2026.synthesis_linkage import (
    CANONICAL_OUTPUT,
    LATENT_STREAM_CONTRACT,
    PRIMARY_QUOTA,
)


HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
MAX_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_BATCH_FILE_BYTES = 16 * 40 * 40 * 3 * 4 + 4096
EXPECTED_CLASS_MAPPING = ("0", "1", "2", "3", "4")
MANIFEST_FIELDS = frozenset({
    "schema",
    "dataset_id",
    "family",
    "experimental_seed",
    "class_mapping",
    "class_count",
    "checkpoint_role",
    "checkpoint_sha256",
    "checkpoint_run_identity",
    "executable_source_commit",
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
    "shape",
    "dtype",
})
EXPECTED_CANONICAL = {
    "shape": [40, 40, 3],
    "layout": "NHWC",
    "dtype": "float32",
    "range": [0.0, 1.0],
}


def _digest(value):
    if type(value) is not str or HEX64.fullmatch(value) is None:
        raise ValueError("expected SHA-256 digest")
    return value


def _read_file(path, limit):
    """Bound reads and reject symlinked final members/nonregular files."""
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


def _exact_int(value, name):
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    return value


def _reject_constant(value):
    raise ValueError(f"non-finite JSON constant: {value}")


def _reject_duplicate_keys(pairs):
    record = {}

    for key, value in pairs:
        if key in record:
            raise ValueError(f"duplicate JSON field: {key}")
        record[key] = value

    return record


class GANPoolView:
    """Complete ordered 10,000-image GAN pool for combined training."""

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
        if family != GAN_POOL_FAMILY:
            raise ValueError("unsupported GAN pool family")
        if (
            type(experimental_seed) is not int
            or experimental_seed not in range(42, 49)
        ):
            raise ValueError("unsupported experimental seed")

        self._manifest_path = Path(manifest_path)
        raw = _read_file(self._manifest_path, MAX_MANIFEST_BYTES)

        if hashlib.sha256(raw).hexdigest() != _digest(manifest_sha256):
            raise ValueError("GAN pool manifest digest mismatch")

        manifest = json.loads(
            raw,
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )

        if type(manifest) is not dict or set(manifest) != MANIFEST_FIELDS:
            raise ValueError("GAN pool manifest fields are invalid")

        if (
            manifest["schema"] != GAN_POOL_SCHEMA
            or manifest["dataset_id"] != GAN_POOL_DATASET
            or manifest["family"] != GAN_POOL_FAMILY
            or _exact_int(
                manifest["experimental_seed"], "experimental_seed"
            ) != experimental_seed
            or manifest["class_mapping"] != list(EXPECTED_CLASS_MAPPING)
            or manifest["class_count"] != len(EXPECTED_CLASS_MAPPING)
            or manifest["checkpoint_role"] != GAN_POOL_CHECKPOINT_ROLE
            or manifest["checkpoint_sha256"] != _digest(
                checkpoint_sha256
            )
            or manifest["recipe_sha256"] != _digest(recipe_sha256)
            or manifest["latent_stream_contract"]
            != LATENT_STREAM_CONTRACT
            or manifest["model_call_batch_size"]
            != GAN_POOL_BATCH_SIZE
            or manifest["quota_per_class"] != PRIMARY_QUOTA
            or manifest["batches_per_class"]
            != GAN_POOL_BATCHES_PER_CLASS
            or manifest["complete_pool"] is not True
            or manifest["expected_total"] != GAN_POOL_TOTAL
            or manifest["observed_total"] != GAN_POOL_TOTAL
            or manifest["status"] != "COMPLETE"
            or manifest["authoritative_artifact"] is not True
            or manifest["canonical_output"] != EXPECTED_CANONICAL
        ):
            raise ValueError("GAN pool identity or completeness mismatch")

        if (
            type(manifest["checkpoint_run_identity"]) is not str
            or not manifest["checkpoint_run_identity"]
        ):
            raise ValueError("invalid checkpoint run identity")

        source_commit = manifest["executable_source_commit"]
        if (
            type(source_commit) is not str
            or HEX40.fullmatch(source_commit) is None
        ):
            raise ValueError("invalid executable source identity")

        rng = manifest["rng"]
        if (
            type(rng) is not dict
            or rng.get("contract") != LATENT_STREAM_CONTRACT
            or rng.get("bit_generator") != "PCG64"
            or rng.get("latent_dimension") != 100
        ):
            raise ValueError("GAN pool RNG identity mismatch")

        records = manifest["batches"]
        if type(records) is not list or len(records) != (
            5 * GAN_POOL_BATCHES_PER_CLASS
        ):
            raise ValueError("GAN pool requires 625 batch records")

        indexed = {}
        filenames = set()

        for record in records:
            if type(record) is not dict or set(record) != BATCH_FIELDS:
                raise ValueError("invalid GAN batch record")

            cls = _exact_int(record["class_id"], "class_id")
            batch = _exact_int(record["batch_index"], "batch_index")

            if not 0 <= cls < 5 or not 0 <= batch < (
                GAN_POOL_BATCHES_PER_CLASS
            ):
                raise ValueError("GAN batch index outside complete pool")

            key = (cls, batch)
            if key in indexed:
                raise ValueError("duplicate GAN batch coverage")

            expected_name = f"c{cls}-b{batch:03d}.npy"
            if (
                record["filename"] != expected_name
                or expected_name in filenames
            ):
                raise ValueError("invalid or repeated GAN batch filename")

            expected_indices = list(
                range(
                    batch * GAN_POOL_BATCH_SIZE,
                    (batch + 1) * GAN_POOL_BATCH_SIZE,
                )
            )
            if record["sample_indices"] != expected_indices:
                raise ValueError("GAN batch sample identity mismatch")

            if (
                record["shape"] != [GAN_POOL_BATCH_SIZE, 40, 40, 3]
                or record["dtype"] != "float32"
            ):
                raise ValueError("GAN batch shape or dtype mismatch")

            indexed[key] = (
                expected_name,
                _digest(record["sha256"]),
            )
            filenames.add(expected_name)

        if set(indexed) != {
            (cls, batch)
            for cls in range(5)
            for batch in range(GAN_POOL_BATCHES_PER_CLASS)
        }:
            raise ValueError("incomplete GAN batch coverage")

        self._records = tuple(
            indexed[(cls, batch)]
            for cls in range(5)
            for batch in range(GAN_POOL_BATCHES_PER_CLASS)
        )
        self._identity = {
            "schema": GAN_POOL_SCHEMA,
            "dataset_id": GAN_POOL_DATASET,
            "family": GAN_POOL_FAMILY,
            "experimental_seed": experimental_seed,
            "class_mapping": list(EXPECTED_CLASS_MAPPING),
            "class_count": 5,
            "checkpoint_role": GAN_POOL_CHECKPOINT_ROLE,
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_run_identity": manifest[
                "checkpoint_run_identity"
            ],
            "executable_source_commit": source_commit,
            "recipe_sha256": recipe_sha256,
            "manifest_sha256": manifest_sha256,
            "model_call_batch_size": GAN_POOL_BATCH_SIZE,
            "quota_per_class": PRIMARY_QUOTA,
        }
        self._preflight_done = False

    @property
    def size(self):
        return GAN_POOL_TOTAL

    @property
    def identity(self):
        return dict(self._identity)

    def _images(self, slot):
        filename, expected = self._records[slot]
        raw = _read_file(
            self._manifest_path.parent / filename,
            MAX_BATCH_FILE_BYTES,
        )

        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("GAN batch digest mismatch: " + filename)

        images = np.load(io.BytesIO(raw), allow_pickle=False)

        if not isinstance(images, np.ndarray):
            raise ValueError("expected NPY image array")

        class_id = slot // GAN_POOL_BATCHES_PER_CLASS
        labels = np.zeros((GAN_POOL_BATCH_SIZE, 5), dtype=np.float32)
        labels[:, class_id] = 1.0
        validate_batch(
            images,
            labels,
            count=GAN_POOL_BATCH_SIZE,
            class_count=5,
        )
        return images

    def preflight(self):
        """Stream every GAN batch before any evaluator batch is available."""
        root = self._manifest_path.parent

        if (root / FAILURE_FILENAME).exists():
            raise RuntimeError("GAN pool is marked incomplete")

        if (root / PENDING_MANIFEST_FILENAME).exists():
            raise RuntimeError("GAN pool publication is incomplete")

        self._preflight_done = False

        for slot in range(5 * GAN_POOL_BATCHES_PER_CLASS):
            self._images(slot)

        self._preflight_done = True
        return {
            "identity": self.identity,
            "verified_batches": 5 * GAN_POOL_BATCHES_PER_CLASS,
            "verified_images": self.size,
        }

    def batch(self, positions):
        if not self._preflight_done:
            raise RuntimeError("complete GAN preflight required")

        if (
            not isinstance(positions, np.ndarray)
            or positions.dtype != np.int64
            or positions.ndim != 1
            or not 1 <= positions.size <= MAX_BATCH_SIZE
            or np.any(positions < 0)
            or np.any(positions >= self.size)
            or np.unique(positions).size != positions.size
        ):
            raise ValueError("invalid bounded GAN positions")

        requested = positions.copy()
        images = np.empty((len(requested), 40, 40, 3), dtype=np.float32)

        slots = requested // GAN_POOL_BATCH_SIZE

        for slot in np.unique(slots):
            selected = np.flatnonzero(slots == slot)
            images[selected] = self._images(int(slot))[
                requested[selected] % GAN_POOL_BATCH_SIZE
            ]

        classes = requested // PRIMARY_QUOTA
        labels = np.eye(5, dtype=np.float32)[classes]

        validate_batch(images, labels, count=len(requested), class_count=5)
        return requested, images, labels

    def iter_ordered(self, *, batch_size=MAX_BATCH_SIZE):
        if type(batch_size) is not int or not 1 <= batch_size <= MAX_BATCH_SIZE:
            raise ValueError("invalid bounded batch size")

        for start in range(0, self.size, batch_size):
            yield self.batch(
                np.arange(
                    start,
                    min(start + batch_size, self.size),
                    dtype=np.int64,
                )
            )


__all__ = [
    "EXPECTED_CLASS_MAPPING",
    "GANPoolView",
    "MANIFEST_FIELDS",
]
