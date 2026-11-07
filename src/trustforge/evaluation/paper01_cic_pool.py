"""Bounded CIC AR/MAF synthetic-pool view.

Uses the existing 1250-record pool contract. Verifies captured manifest bytes
and current batch bytes; does not authenticate checkpoint lineage or authorize
evaluation. Parent-directory immutability is not established here.
"""

import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat

import numpy as np

from .paper01_inputs import MAX_BATCH_SIZE, validate_batch
from trustforge.models.paper01_revision_2026.ar_maf_execution_inputs import (
    synthesis_batch_seed,
)


HEX64 = re.compile(r"^[0-9a-f]{64}$")
MAX_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_BATCH_FILE_BYTES = 8 * 40 * 40 * 3 * 4 + 4096


def _digest(value):
    if type(value) is not str or HEX64.fullmatch(value) is None:
        raise ValueError("expected SHA-256 digest")
    return value


def _read_file(path, limit):
    """Bound file reads and reject symlinked final members/nonregular files."""
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


class CICPoolView:
    def __init__(
        self, manifest_path, *, manifest_sha256, family, experimental_seed,
        recipe_sha256, checkpoint_sha256,
    ):
        if family not in ("ar", "maf"):
            raise ValueError("unsupported CIC pool family")
        if type(experimental_seed) is not int or experimental_seed not in range(42, 49):
            raise ValueError("unsupported experimental seed")
        self._manifest_path = Path(manifest_path)
        raw = _read_file(self._manifest_path, MAX_MANIFEST_BYTES)
        if hashlib.sha256(raw).hexdigest() != _digest(manifest_sha256):
            raise ValueError("manifest digest mismatch")
        manifest = json.loads(raw)
        expected = {
            "schema_version": 1,
            "dataset_id": "cicmaldroid2020",
            "family": family,
            "experimental_seed": experimental_seed,
            "recipe_sha256": _digest(recipe_sha256),
            "checkpoint_sha256": _digest(checkpoint_sha256),
        }
        if type(manifest) is not dict or any(
            type(manifest.get(key)) is not type(value)
            or manifest.get(key) != value
            for key, value in expected.items()
        ):
            raise ValueError("CIC pool identity mismatch")
        records = manifest.get("batches")
        if type(records) is not list or len(records) != 1250:
            raise ValueError("complete pool requires 1250 batch records")

        indexed = {}
        filenames = set()
        for record in records:
            if type(record) is not dict:
                raise ValueError("invalid batch record")
            cls = record.get("class_id")
            batch = record.get("batch_index")
            if (
                type(cls) is not int or not 0 <= cls < 5
                or type(batch) is not int or not 0 <= batch < 250
            ):
                raise ValueError("invalid class or batch index")
            key = (cls, batch)
            if key in indexed:
                raise ValueError("duplicate batch coverage")
            filename = record.get("filename")
            if (
                type(filename) is not str
                or not filename.endswith(".npy")
                or Path(filename).name != filename
                or "/" in filename or "\\" in filename or "\x00" in filename
                or filename in filenames
            ):
                raise ValueError("invalid or repeated batch filename")
            seed = synthesis_batch_seed(
                dataset_id="cicmaldroid2020", family=family,
                experimental_seed=experimental_seed,
                class_id=cls, batch_index=batch,
            )
            if type(record.get("sampling_seed")) is not int or record["sampling_seed"] != seed:
                raise ValueError("batch sampling seed mismatch")
            indexed[key] = (filename, _digest(record.get("sha256")))
            filenames.add(filename)

        self._records = tuple(
            indexed[(cls, batch)] for cls in range(5) for batch in range(250)
        )
        self._identity = {**expected, "manifest_sha256": manifest_sha256}
        self._preflight_done = False

    @property
    def size(self):
        return 10000

    @property
    def identity(self):
        return dict(self._identity)

    def _images(self, slot):
        filename, expected = self._records[slot]
        raw = _read_file(
            self._manifest_path.parent / filename, MAX_BATCH_FILE_BYTES,
        )
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("batch digest mismatch: " + filename)
        images = np.load(io.BytesIO(raw), allow_pickle=False)
        if not isinstance(images, np.ndarray):
            raise ValueError("expected NPY image array")
        labels = np.zeros((8, 5), dtype=np.float32)
        labels[:, slot // 250] = 1
        validate_batch(images, labels, count=8, class_count=5)
        return images

    def preflight(self):
        """Stream every batch before any evaluator batch may be requested."""
        self._preflight_done = False
        for slot in range(1250):
            self._images(slot)
        self._preflight_done = True
        return {
            "identity": self.identity,
            "verified_batches": 1250,
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
            or np.any(positions < 0) or np.any(positions >= self.size)
            or np.unique(positions).size != positions.size
        ):
            raise ValueError("invalid bounded synthetic positions")
        requested = positions.copy()
        images = np.empty((len(requested), 40, 40, 3), dtype=np.float32)
        # Class-major sample order: each class has 250 consecutive batches.
        slots = requested // 8
        for slot in np.unique(slots):
            selected = np.flatnonzero(slots == slot)
            images[selected] = self._images(int(slot))[requested[selected] % 8]
        classes = requested // 2000
        labels = np.eye(5, dtype=np.float32)[classes]
        validate_batch(images, labels, count=len(requested), class_count=5)
        return requested, images, labels

    def iter_ordered(self, *, batch_size=MAX_BATCH_SIZE):
        if type(batch_size) is not int or not 1 <= batch_size <= MAX_BATCH_SIZE:
            raise ValueError("invalid bounded batch size")
        for start in range(0, self.size, batch_size):
            yield self.batch(np.arange(
                start, min(start + batch_size, self.size), dtype=np.int64,
            ))
