"""Validate one CIC AR/MAF benchmark pool before evaluator training."""

import hashlib
import io
import json
from pathlib import Path

import numpy as np

from .ar_maf_execution_inputs import FAMILIES, SEEDS, synthesis_batch_seed


def _sha256(value):
    return hashlib.sha256(value).hexdigest()


def _is_digest(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def load_cic_benchmark_pool(
    manifest_path, *, family, experimental_seed, recipe_sha256, checkpoint_sha256
):
    """Return verified float32 images and int32 labels in class/batch order."""
    if (
        family not in FAMILIES
        or type(experimental_seed) is not int
        or experimental_seed not in SEEDS
        or not _is_digest(recipe_sha256)
        or not _is_digest(checkpoint_sha256)
    ):
        raise ValueError("invalid expected CIC run identity")

    manifest_path = Path(manifest_path)
    if manifest_path.suffix != ".json":
        raise ValueError("benchmark mode requires a JSON pool manifest")
    manifest = json.loads(manifest_path.read_bytes())
    expected = {
        "schema_version": 1,
        "dataset_id": "cicmaldroid2020",
        "family": family,
        "experimental_seed": experimental_seed,
        "recipe_sha256": recipe_sha256,
        "checkpoint_sha256": checkpoint_sha256,
    }
    if not isinstance(manifest, dict) or any(
        type(manifest.get(key)) is not type(value) or manifest.get(key) != value
        for key, value in expected.items()
    ):
        raise ValueError("CIC pool identity mismatch")

    records = manifest.get("batches")
    if not isinstance(records, list) or len(records) != 1250:
        raise ValueError("expected exactly 1,250 CIC batch records")

    indexed = {}
    filenames = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("invalid CIC batch record")
        class_id = record.get("class_id")
        batch_index = record.get("batch_index")
        if (
            type(class_id) is not int or not 0 <= class_id < 5
            or type(batch_index) is not int or not 0 <= batch_index < 250
        ):
            raise ValueError("invalid CIC class ID or batch index")
        key = (class_id, batch_index)
        if key in indexed:
            raise ValueError(f"duplicate CIC batch {key}")

        seed = synthesis_batch_seed(
            dataset_id=expected["dataset_id"],
            family=family,
            experimental_seed=experimental_seed,
            class_id=class_id,
            batch_index=batch_index,
        )
        filename = record.get("filename")
        digest = record.get("sha256")
        if (
            type(record.get("sampling_seed")) is not int
            or record["sampling_seed"] != seed
            or not isinstance(filename, str)
            or Path(filename).name != filename
            or Path(filename).suffix != ".npy"
            or filename in filenames
            or not _is_digest(digest)
        ):
            raise ValueError(f"invalid seed, filename, or digest for batch {key}")
        indexed[key] = (filename, digest)
        filenames.add(filename)

    images = np.empty((10000, 40, 40, 3), dtype=np.float32)
    labels = np.repeat(np.arange(5, dtype=np.int32), 2000)
    for (class_id, batch_index), (filename, digest) in sorted(indexed.items()):
        path = manifest_path.parent / filename
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"expected a regular batch file: {filename}")
        raw = path.read_bytes()
        if _sha256(raw) != digest:
            raise ValueError(f"batch digest mismatch: {filename}")
        batch = np.load(io.BytesIO(raw), allow_pickle=False)
        if (
            not isinstance(batch, np.ndarray)
            or batch.shape != (8, 40, 40, 3)
            or batch.dtype != np.float32
            or not np.isfinite(batch).all()
            or np.any(batch < 0)
            or np.any(batch > 1)
        ):
            raise ValueError(f"invalid canonical images: {filename}")
        start = 2000 * class_id + 8 * batch_index
        images[start:start + 8] = batch

    return images, labels
