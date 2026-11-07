"""Finalize a complete checkpoint-backed CIC pool from worker batch outputs."""

import hashlib
import io
import json
from pathlib import Path

import numpy as np

from .ar_maf_cic_pool_publication import publish_cic_benchmark_pool
from .ar_maf_execution_inputs import synthesis_batch_seed


def finalize_cic_partitioned_pool(
    destination, *, batch_directories, family, experimental_seed,
    recipe_sha256, checkpoint_sha256, synthesis_commit, config_sha256,
):
    """Publish a benchmark pool from batch directories supplied in any order.

    Each directory contains report.json and samples.npz. The caller supplies
    expected provenance from the verified run configuration and checkpoint.
    """
    expected = {
        "dataset_id": "cicmaldroid2020",
        "family": family,
        "experimental_seed": experimental_seed,
        "recipe_sha256": recipe_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "synthesis_commit": synthesis_commit,
        "config_sha256": config_sha256,
    }
    indexed = {}
    for directory in batch_directories:
        directory = Path(directory)
        report = json.loads((directory / "report.json").read_bytes())
        if not isinstance(report, dict) or any(
            type(report.get(key)) is not type(value) or report.get(key) != value
            for key, value in expected.items()
        ):
            raise ValueError(f"worker batch identity mismatch: {directory}")

        class_id = report.get("class_id")
        batch_index = report.get("batch_index")
        seed = synthesis_batch_seed(
            dataset_id=expected["dataset_id"],
            family=family,
            experimental_seed=experimental_seed,
            class_id=class_id,
            batch_index=batch_index,
        )
        if type(report.get("sampling_seed")) is not int or report["sampling_seed"] != seed:
            raise ValueError(f"worker batch seed mismatch: {directory}")

        key = (class_id, batch_index)
        if key in indexed:
            raise ValueError(f"duplicate worker batch: {key}")
        indexed[key] = (directory / "samples.npz", report.get("samples_sha256"))

    required = {
        (class_id, batch_index)
        for class_id in range(5)
        for batch_index in range(250)
    }
    if indexed.keys() != required:
        raise ValueError("expected complete CIC coverage of 1,250 batches")

    def load_batch(class_id, batch_index, sampling_seed):
        path, expected_digest = indexed[(class_id, batch_index)]
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"expected a regular worker archive: {path}")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_digest:
            raise ValueError(f"worker archive digest mismatch: {path}")

        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            if sorted(archive.files) != ["class_ids", "images"]:
                raise ValueError(f"unexpected worker archive members: {path}")
            class_ids = archive["class_ids"]
            if (
                class_ids.dtype != np.int32
                or class_ids.shape != (8,)
                or not np.all(class_ids == class_id)
            ):
                raise ValueError(f"worker archive class IDs mismatch: {path}")
            return archive["images"]

    return publish_cic_benchmark_pool(
        destination,
        family=family,
        experimental_seed=experimental_seed,
        recipe_sha256=recipe_sha256,
        checkpoint_sha256=checkpoint_sha256,
        sample_batch=load_batch,
    )
