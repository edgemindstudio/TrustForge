#!/usr/bin/env python3
"""Coordinate the accepted CIC AR seed-42 checkpoint-backed full-pool path.

This module plans and validates the existing range-worker partition, then
delegates final publication to the accepted partition finalizer and verifies
the result with the accepted pool preflight. It does not load a checkpoint,
generate images, submit Slurm jobs, or activate production publication.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from trustforge.models.paper01_revision_2026.ar_maf_cic_partition_finalizer import (
    finalize_cic_partitioned_pool,
)
from trustforge.models.paper01_revision_2026.ar_maf_cic_pool_preflight import (
    load_cic_benchmark_pool,
)
from trustforge.models.paper01_revision_2026.ar_maf_execution_inputs import (
    synthesis_batch_seed,
)
from trustforge.models.paper01_revision_2026.ar_cic_pool_authorization import (
    validate_ar_cic_pool_authorization,
)


FAMILY = "ar"
DATASET_ID = "cicmaldroid2020"
EXPERIMENTAL_SEED = 42
CLASS_COUNT = 5
BATCHES_PER_CLASS = 250
BATCH_SIZE = 8
TOTAL_BATCHES = CLASS_COUNT * BATCHES_PER_CLASS
POOL_SAMPLES = TOTAL_BATCHES * BATCH_SIZE

TRAINING_COMMIT = "e4473abcdbc53f0de3a01ebe48858f2f3c7b841a"
CHECKPOINT_SHA256 = (
    "d58a64ad8cadc93e745945191e374a9c27b5668cf569f5bac4761a172b29a0f0"
)
GENERATION = "ar_cic_seed42"
RECIPE_PATH = "studies/paper01_benchmark/revision_2026/ar_maf_integration_recipe.json"
RECIPE_SHA256 = (
    "202db292bb18d3e6f04f6f8cc36658884994c3bd9f4f48e2e93e181d348ea478"
)
CONFIG_SHA256 = (
    "ea1672eba0183b692ed29ffe7e8f4b9a707d487049c27623a38ddddff1a99f8b"
)

DEFAULT_RANGES_PER_CLASS = 10
PROPOSED_RESOURCES = {
    "partition": "talon",
    "tasks": CLASS_COUNT * DEFAULT_RANGES_PER_CLASS,
    "concurrency": 2,
    "cpus_per_task": 8,
    "memory_gb": 32,
    "walltime_seconds": 90 * 60,
    "batches_per_task": BATCHES_PER_CLASS // DEFAULT_RANGES_PER_CLASS,
    "automatic_retries": 0,
    "requeue": False,
    "gpu_enabled": False,
    "resource_status": "PROPOSED_NOT_AUTHORIZED",
}

HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")

# Publication is default-deny: the production finalizer requires a validated
# external authorization record bound to the exact plan and destinations.


def _require_commit(value, name):
    if type(value) is not str or HEX40.fullmatch(value) is None:
        raise ValueError(name + " requires a 40-character commit")
    return value


def _require_digest(value, name):
    if type(value) is not str or HEX64.fullmatch(value) is None:
        raise ValueError(name + " requires a lowercase SHA-256")
    return value


def _sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _require_regular(path, name):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(name + " must be a regular file")
    return path


def source_check(repository_root, expected_head):
    """Require a clean, top-level checkout at the declared source revision."""
    root = Path(repository_root).expanduser().resolve()
    _require_commit(expected_head, "expected source revision")
    if not root.is_dir() or root.is_symlink():
        raise ValueError("repository root is unavailable")
    top = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
        text=True,
    ).strip()
    if Path(top).resolve() != root:
        raise ValueError("repository root is not the checkout top level")
    head = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True,
    ).strip()
    if head != expected_head:
        raise RuntimeError("checkout HEAD differs from the expected source")
    dirty = subprocess.check_output(
        [
            "git", "-C", str(root), "status", "--porcelain",
            "--untracked-files=no",
        ],
        text=True,
    )
    if dirty.strip():
        raise RuntimeError("checkout must be clean")
    return root, head


def partition_ranges(ranges_per_class=DEFAULT_RANGES_PER_CLASS):
    """Return a complete, non-overlapping class/batch partition."""
    if type(ranges_per_class) is not int or not 1 <= ranges_per_class <= BATCHES_PER_CLASS:
        raise ValueError("ranges_per_class must be an integer in [1, 250]")
    if BATCHES_PER_CLASS % ranges_per_class:
        raise ValueError("ranges_per_class must divide 250 exactly")
    width = BATCHES_PER_CLASS // ranges_per_class
    tasks = []
    for class_id in range(CLASS_COUNT):
        for start in range(0, BATCHES_PER_CLASS, width):
            stop = start + width
            tasks.append({
                "task_index": len(tasks),
                "class_id": class_id,
                "batch_start": start,
                "batch_stop": stop,
                "batch_count": stop - start,
                "worker_name": f"c{class_id}-b{start:03d}-{stop - 1:03d}",
                "sampling_seed_start": synthesis_batch_seed(
                    dataset_id=DATASET_ID, family=FAMILY,
                    experimental_seed=EXPERIMENTAL_SEED,
                    class_id=class_id, batch_index=start,
                ),
                "sampling_seed_end": synthesis_batch_seed(
                    dataset_id=DATASET_ID, family=FAMILY,
                    experimental_seed=EXPERIMENTAL_SEED,
                    class_id=class_id, batch_index=stop - 1,
                ),
            })
    expected = {
        (class_id, batch_index)
        for class_id in range(CLASS_COUNT)
        for batch_index in range(BATCHES_PER_CLASS)
    }
    observed = {
        (task["class_id"], batch_index)
        for task in tasks
        for batch_index in range(task["batch_start"], task["batch_stop"])
    }
    if len(observed) != len(expected) or observed != expected:
        raise RuntimeError("internal partition coverage error")
    if len(tasks) != CLASS_COUNT * ranges_per_class:
        raise RuntimeError("internal task count error")
    return tasks


def build_plan(
    *, repository_root, expected_head, run_root, checkpoint_parent,
    dataset_config, config_sha256, ranges_per_class=DEFAULT_RANGES_PER_CLASS,
    destination=None,
):
    """Build the read-only production plan; creates no directories."""
    root, head = source_check(repository_root, expected_head)
    if _require_digest(config_sha256, "dataset configuration") != CONFIG_SHA256:
        raise ValueError("dataset configuration digest differs from the accepted value")
    config = _require_regular(dataset_config, "dataset configuration")
    if _sha256_file(config) != CONFIG_SHA256:
        raise ValueError("dataset configuration digest mismatch")

    recipe = _require_regular(root / RECIPE_PATH, "AR integration recipe")
    if _sha256_file(recipe) != RECIPE_SHA256:
        raise ValueError("AR integration recipe digest mismatch")

    checkpoint_parent = Path(checkpoint_parent).expanduser()
    if checkpoint_parent.is_symlink() or not checkpoint_parent.is_dir():
        raise ValueError("checkpoint parent is unavailable")
    checkpoint_directory = checkpoint_parent / GENERATION
    if checkpoint_directory.is_symlink() or not checkpoint_directory.is_dir():
        raise ValueError("accepted AR checkpoint generation is unavailable")

    run_root = Path(run_root).expanduser().resolve()
    if destination is None:
        destination = run_root / "pool"
    destination = Path(destination).expanduser().resolve()
    if destination.parent != run_root:
        raise ValueError("pool destination must be directly under the run root")
    tasks = partition_ranges(ranges_per_class)
    resources = dict(PROPOSED_RESOURCES)
    resources["tasks"] = len(tasks)
    resources["batches_per_task"] = BATCHES_PER_CLASS // ranges_per_class
    if ranges_per_class != DEFAULT_RANGES_PER_CLASS:
        resources["walltime_seconds"] = None
        resources["resource_status"] = "PENDING_PARTITION_RECALCULATION"
    plan = {
        "schema": "trustforge.paper01.ar-cic-seed42-pool-plan.v1",
        "execution_authorized": False,
        "publication_enabled": False,
        "publication_requires_authorization": True,
        "family": FAMILY,
        "dataset_id": DATASET_ID,
        "experimental_seed": EXPERIMENTAL_SEED,
        "class_mapping": ["0", "1", "2", "3", "4"],
        "training_commit": TRAINING_COMMIT,
        "synthesis_commit": head,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "checkpoint_generation": GENERATION,
        "checkpoint_directory": str(checkpoint_directory),
        "recipe_sha256": RECIPE_SHA256,
        "config_sha256": CONFIG_SHA256,
        "batch_size": BATCH_SIZE,
        "batches_per_class": BATCHES_PER_CLASS,
        "total_batches": TOTAL_BATCHES,
        "pool_samples": POOL_SAMPLES,
        "ranges_per_class": ranges_per_class,
        "run_root": str(run_root),
        "workers_root": str(run_root / "workers"),
        "destination": str(destination),
        "log_root": str(run_root / "logs"),
        "tasks": tasks,
        "resources": resources,
        "finalizer_resources": {
            "partition": "talon",
            "nodes": 1,
            "tasks": 1,
            "cpus_per_task": 8,
            "memory_gb": 32,
            "walltime_seconds": 30 * 60,
            "automatic_retries": 0,
            "requeue": False,
            "gpu_enabled": False,
            "resource_status": "PROPOSED_NOT_AUTHORIZED",
        },
        "finalization": {
            "dependency": "afterok_all_tasks",
            "mode": "coordinator --finalize",
            "publish_manifest_only_after_complete_coverage": True,
            "automatic_retry": False,
            "automatic_resource_escalation": False,
        },
        "resource_assumptions": [
            "AR batch-eight observations: 118.16 s (job 310954), "
            "125.20 s (job 311471), 118.72-118.95 s (range boundary 311664).",
            "25 batches/task at ~125.2 s/batch is about 52.2 minutes; "
            "the 90-minute request is an unverified ceiling.",
            "Concurrency two and 32 GB are proposed, not measured "
            "production guarantees.",
        ],
    }
    return plan


def worker_directory(run_root, task):
    return Path(run_root) / "workers" / task["worker_name"]


def _expected_report_values(plan):
    return {
        "dataset_id": DATASET_ID,
        "family": FAMILY,
        "experimental_seed": EXPERIMENTAL_SEED,
        "recipe_sha256": plan["recipe_sha256"],
        "checkpoint_sha256": plan["checkpoint_sha256"],
        "synthesis_commit": plan["synthesis_commit"],
        "config_sha256": plan["config_sha256"],
    }


def validate_worker_partition(plan, run_root):
    """Validate exact range ownership and return batch dirs in canonical order."""
    run_root = Path(run_root)
    workers_root = run_root / "workers"
    if workers_root.is_symlink() or not workers_root.is_dir():
        raise ValueError("workers root is unavailable")

    expected_names = {task["worker_name"] for task in plan["tasks"]}
    observed_names = {
        path.name for path in workers_root.iterdir()
        if path.name.startswith("c")
    }
    if observed_names != expected_names:
        raise ValueError("worker partition directory set differs from the plan")

    expected = _expected_report_values(plan)
    batch_directories = []
    seen = set()
    for task in plan["tasks"]:
        worker = worker_directory(run_root, task)
        if worker.is_symlink() or not worker.is_dir():
            raise ValueError("worker directory is unavailable: " + worker.name)
        expected_batches = set(range(task["batch_start"], task["batch_stop"]))
        observed_batches = set()
        for entry in worker.iterdir():
            if not entry.name.startswith("batch-"):
                continue
            if entry.is_symlink() or not entry.is_dir():
                raise ValueError("worker batch entry must be a directory: " + entry.name)
            try:
                batch_index = int(entry.name.split("-", 1)[1])
            except (IndexError, ValueError):
                raise ValueError("invalid worker batch directory: " + entry.name)
            if batch_index in observed_batches:
                raise ValueError("duplicate worker batch directory: " + entry.name)
            observed_batches.add(batch_index)
        if observed_batches != expected_batches:
            raise ValueError(
                "worker range coverage differs from the plan: " + worker.name
            )
        for batch_index in range(task["batch_start"], task["batch_stop"]):
            batch_dir = worker / f"batch-{batch_index:03d}"
            report_path = batch_dir / "report.json"
            if report_path.is_symlink() or not report_path.is_file():
                raise ValueError("worker batch report is unavailable: " + str(report_path))
            report = json.loads(report_path.read_bytes())
            if type(report) is not dict:
                raise ValueError("worker batch report is invalid")
            for key, value in expected.items():
                if type(report.get(key)) is not type(value) or report.get(key) != value:
                    raise ValueError(
                        "worker batch identity mismatch: " + str(batch_dir)
                    )
            if (
                type(report.get("class_id")) is not int
                or report["class_id"] != task["class_id"]
                or type(report.get("batch_index")) is not int
                or report["batch_index"] != batch_index
            ):
                raise ValueError("worker batch class/index mismatch: " + str(batch_dir))
            seed = synthesis_batch_seed(
                dataset_id=DATASET_ID, family=FAMILY,
                experimental_seed=EXPERIMENTAL_SEED,
                class_id=task["class_id"], batch_index=batch_index,
            )
            if type(report.get("sampling_seed")) is not int or report["sampling_seed"] != seed:
                raise ValueError("worker batch sampling seed mismatch: " + str(batch_dir))
            _require_digest(report.get("samples_sha256"), "worker archive digest")
            if (
                report.get("shape") != [BATCH_SIZE, 40, 40, 3]
                or report.get("dtype") != "float32"
            ):
                raise ValueError("worker batch canonical metadata mismatch: " + str(batch_dir))
            archive = batch_dir / "samples.npz"
            if archive.is_symlink() or not archive.is_file():
                raise ValueError("worker archive is unavailable: " + str(archive))
            key = (task["class_id"], batch_index)
            if key in seen:
                raise ValueError("duplicate worker batch identity: " + str(key))
            seen.add(key)
            batch_directories.append(batch_dir)

    if len(seen) != TOTAL_BATCHES or len(batch_directories) != TOTAL_BATCHES:
        raise ValueError("expected complete CIC coverage of 1,250 batches")
    return batch_directories


def _verify_pool_arrays(images, labels):
    import numpy as np

    if (
        not isinstance(images, np.ndarray)
        or images.shape != (POOL_SAMPLES, 40, 40, 3)
        or images.dtype != np.float32
        or not np.isfinite(images).all()
        or np.any(images < 0)
        or np.any(images > 1)
    ):
        raise ValueError("published pool images are not canonical")
    if (
        not isinstance(labels, np.ndarray)
        or labels.shape != (POOL_SAMPLES,)
        or labels.dtype != np.int32
    ):
        raise ValueError("published pool labels are invalid")
    counts = [
        int(np.count_nonzero(labels == class_id))
        for class_id in range(CLASS_COUNT)
    ]
    if counts != [2000] * CLASS_COUNT:
        raise ValueError("published pool class coverage is incomplete")
    expected_labels = np.repeat(
        np.arange(CLASS_COUNT, dtype=np.int32), 2000,
    )
    if not np.array_equal(labels, expected_labels):
        raise ValueError("published pool label order is not authoritative")
    return {"class_counts": counts, "images": POOL_SAMPLES}


def finalize_pool(*, plan, run_root, destination, authorization_path):
    """Finalize and verify the full pool; publication is explicitly gated."""
    authorization = validate_ar_cic_pool_authorization(
        authorization_path,
        plan=plan,
        run_root=run_root,
        destination=destination,
    )
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("pool destination already exists")
    run_root = Path(run_root)
    if run_root.is_symlink() or not run_root.is_dir():
        raise ValueError("run root is unavailable")
    batch_directories = validate_worker_partition(plan, run_root)
    manifest = finalize_cic_partitioned_pool(
        destination,
        batch_directories=batch_directories,
        family=FAMILY,
        experimental_seed=EXPERIMENTAL_SEED,
        recipe_sha256=plan["recipe_sha256"],
        checkpoint_sha256=plan["checkpoint_sha256"],
        synthesis_commit=plan["synthesis_commit"],
        config_sha256=plan["config_sha256"],
    )
    images, labels = load_cic_benchmark_pool(
        manifest,
        family=FAMILY,
        experimental_seed=EXPERIMENTAL_SEED,
        recipe_sha256=plan["recipe_sha256"],
        checkpoint_sha256=plan["checkpoint_sha256"],
    )
    summary = _verify_pool_arrays(images, labels)
    summary.update({
        "manifest_path": str(manifest),
        "manifest_sha256": _sha256_file(manifest),
        "batch_count": TOTAL_BATCHES,
        "pool_samples": POOL_SAMPLES,
        "publication_authorized": True,
        "publication_authorization_id": authorization["authorization_id"],
    })
    return summary


def verify_pool(manifest_path):
    """Read-only preflight of an already published AR CIC pool."""
    manifest_path = _require_regular(manifest_path, "pool manifest")
    images, labels = load_cic_benchmark_pool(
        manifest_path,
        family=FAMILY,
        experimental_seed=EXPERIMENTAL_SEED,
        recipe_sha256=RECIPE_SHA256,
        checkpoint_sha256=CHECKPOINT_SHA256,
    )
    summary = _verify_pool_arrays(images, labels)
    summary.update({
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "batch_count": TOTAL_BATCHES,
        "pool_samples": POOL_SAMPLES,
    })
    return summary


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--emit-plan", action="store_true")
    mode.add_argument("--finalize", action="store_true")
    mode.add_argument("--verify", action="store_true")
    parser.add_argument("--repository-root")
    parser.add_argument("--expected-head")
    parser.add_argument("--run-root")
    parser.add_argument("--checkpoint-parent")
    parser.add_argument("--dataset-config")
    parser.add_argument("--config-sha256")
    parser.add_argument("--destination")
    parser.add_argument("--manifest")
    parser.add_argument("--authorization")
    parser.add_argument(
        "--ranges-per-class", type=int, default=DEFAULT_RANGES_PER_CLASS,
    )
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.verify:
        if args.manifest is None:
            raise SystemExit("--verify requires --manifest")
        print(json.dumps(verify_pool(args.manifest), sort_keys=True), flush=True)
        return

    required = (
        "repository_root", "expected_head", "run_root",
        "checkpoint_parent", "dataset_config", "config_sha256",
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        raise SystemExit(
            "missing required arguments: " + ", ".join("--" + name.replace("_", "-") for name in missing)
        )

    plan = build_plan(
        repository_root=args.repository_root,
        expected_head=args.expected_head,
        run_root=args.run_root,
        checkpoint_parent=args.checkpoint_parent,
        dataset_config=args.dataset_config,
        config_sha256=args.config_sha256,
        ranges_per_class=args.ranges_per_class,
        destination=args.destination,
    )
    if args.emit_plan:
        run_root = Path(args.run_root)
        if run_root.exists() or run_root.is_symlink():
            raise FileExistsError("run root must be a fresh destination")
        print(json.dumps(plan, sort_keys=True, allow_nan=False), flush=True)
        return
    if args.finalize:
        if args.authorization is None:
            raise SystemExit("--finalize requires --authorization")
        summary = finalize_pool(
            plan=plan,
            run_root=args.run_root,
            destination=plan["destination"],
            authorization_path=args.authorization,
        )
        print(json.dumps(summary, sort_keys=True), flush=True)
        return
    raise SystemExit("unsupported mode")


if __name__ == "__main__":
    main()
