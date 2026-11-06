"""Publish one complete CIC AR/MAF pool using the benchmark manifest format."""

import hashlib
import io
import json
from pathlib import Path

import numpy as np

from .ar_maf_execution_inputs import FAMILIES, SEEDS, synthesis_batch_seed


def publish_cic_benchmark_pool(
    destination, *, family, experimental_seed, recipe_sha256,
    checkpoint_sha256, sample_batch,
):
    """Call sample_batch(class_id, batch_index, sampling_seed) for each batch.

    Return the manifest path. An interrupted attempt may leave batch files,
    but consumers can discover the final manifest only after publication.
    """
    if family not in FAMILIES or type(experimental_seed) is not int or experimental_seed not in SEEDS:
        raise ValueError("invalid CIC family or experimental seed")
    for digest in (recipe_sha256, checkpoint_sha256):
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("invalid recipe or checkpoint SHA-256")

    destination = Path(destination)
    destination.mkdir(mode=0o700)
    dataset_id = "cicmaldroid2020"
    records = []

    for class_id in range(5):
        for batch_index in range(250):
            sampling_seed = synthesis_batch_seed(
                dataset_id=dataset_id,
                family=family,
                experimental_seed=experimental_seed,
                class_id=class_id,
                batch_index=batch_index,
            )
            batch = sample_batch(class_id, batch_index, sampling_seed)
            if (
                not isinstance(batch, np.ndarray)
                or batch.shape != (8, 40, 40, 3)
                or batch.dtype != np.float32
                or not np.isfinite(batch).all()
                or np.any(batch < 0)
                or np.any(batch > 1)
            ):
                raise ValueError(f"invalid canonical batch {class_id}/{batch_index}")

            filename = f"c{class_id}-b{batch_index:03d}.npy"
            buffer = io.BytesIO()
            np.save(buffer, batch, allow_pickle=False)
            raw = buffer.getvalue()
            with (destination / filename).open("xb") as stream:
                stream.write(raw)
            records.append({
                "class_id": class_id,
                "batch_index": batch_index,
                "sampling_seed": sampling_seed,
                "filename": filename,
                "sha256": hashlib.sha256(raw).hexdigest(),
            })

    manifest = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "family": family,
        "experimental_seed": experimental_seed,
        "recipe_sha256": recipe_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "batches": records,
    }
    pending = destination / ".manifest.pending"
    final = destination / "manifest.json"
    with pending.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
    final.hardlink_to(pending)
    pending.unlink()
    return final
