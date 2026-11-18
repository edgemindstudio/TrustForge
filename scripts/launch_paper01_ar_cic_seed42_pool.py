#!/usr/bin/env python3
"""Submit the bounded AR CIC seed-42 array worker and dependent finalizer.

Dry-run is the default and creates or submits nothing. ``--submit`` requires a
validated external authorization record and creates a fresh run namespace
before invoking the scheduler.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from scripts import run_paper01_ar_cic_seed42_pool as coordinator
from trustforge.models.paper01_revision_2026.ar_cic_pool_authorization import (
    validate_ar_cic_pool_authorization,
)


ARRAY_SCRIPT = "slurm/paper01_revision_2026/ar_cic_seed42_pool.slurm"
FINALIZER_SCRIPT = (
    "slurm/paper01_revision_2026/ar_cic_seed42_pool_finalize.slurm"
)
ARRAY_SIZE = 50
CONCURRENCY = 2


def _sbatch(arguments):
    completed = subprocess.run(
        ["sbatch", *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _job_id(raw):
    match = re.search(r"\b(\d+)\b", raw)
    if match is None:
        raise RuntimeError("scheduler did not return a numeric job identity")
    return match.group(1)


def _environment(plan, *, repository_root, dataset_config, checkpoint_parent,
                 authorization_path):
    return {
        "TRUSTFORGE_REPO_ROOT": str(repository_root),
        "TF_SYNTH_EXPECTED_HEAD": plan["synthesis_commit"],
        "TF_SYNTH_FAMILY": "ar",
        "TF_SYNTH_EXPERIMENTAL_SEED": "42",
        "TF_SYNTH_DATASET_CONFIG": str(dataset_config),
        "TF_SYNTH_CONFIG_SHA256": plan["config_sha256"],
        "TF_SYNTH_CHECKPOINT_PARENT": str(checkpoint_parent),
        "AR_POOL_RUN_ROOT": plan["run_root"],
        "AR_POOL_DESTINATION": plan["destination"],
        "AR_POOL_AUTHORIZATION": str(authorization_path),
    }


def _worker_command(run_root, repository_root):
    return [
        "--parsable",
        f"--array=0-{ARRAY_SIZE - 1}%{CONCURRENCY}",
        "--no-requeue",
        "--export=ALL",
        "--output", str(run_root / "logs" / "worker-%A_%a.out"),
        "--error", str(run_root / "logs" / "worker-%A_%a.err"),
        str(Path(repository_root) / ARRAY_SCRIPT),
    ]


def _finalizer_command(run_root, repository_root, worker_job_id):
    dependency = (
        f"--dependency=afterok:{worker_job_id}"
        if worker_job_id is not None
        else "--dependency=afterok:<worker_array_job_id>"
    )
    return [
        "--parsable",
        dependency,
        "--no-requeue",
        "--export=ALL",
        "--output", str(run_root / "logs" / "finalize-%j.out"),
        "--error", str(run_root / "logs" / "finalize-%j.err"),
        str(Path(repository_root) / FINALIZER_SCRIPT),
    ]


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--checkpoint-parent", required=True)
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--destination")
    parser.add_argument("--submit", action="store_true")
    return parser


def main(argv=None):
    os.umask(0o077)
    args = _parser().parse_args(argv)
    plan = coordinator.build_plan(
        repository_root=args.repository_root,
        expected_head=args.expected_head,
        run_root=args.run_root,
        checkpoint_parent=args.checkpoint_parent,
        dataset_config=args.dataset_config,
        config_sha256=args.config_sha256,
        destination=args.destination,
    )
    validate_ar_cic_pool_authorization(
        args.authorization,
        plan=plan,
        run_root=args.run_root,
        destination=plan["destination"],
    )
    run_root = Path(plan["run_root"])
    if run_root.exists() or run_root.is_symlink():
        raise FileExistsError("run root must be a fresh destination")

    worker_arguments = _worker_command(
        run_root, Path(args.repository_root).resolve(),
    )
    finalizer_arguments = _finalizer_command(
        run_root, Path(args.repository_root).resolve(), None,
    )
    environment = _environment(
        plan,
        repository_root=Path(args.repository_root).resolve(),
        dataset_config=Path(args.dataset_config),
        checkpoint_parent=Path(args.checkpoint_parent),
        authorization_path=Path(args.authorization),
    )

    if not args.submit:
        print(json.dumps({
            "submission": False,
            "worker_command": ["sbatch", *worker_arguments],
            "finalizer_command": ["sbatch", *finalizer_arguments],
            "environment_keys": sorted(environment),
        }, sort_keys=True), flush=True)
        return

    run_root.mkdir(mode=0o700)
    (run_root / "workers").mkdir()
    (run_root / "logs").mkdir()
    os.environ.update(environment)
    worker_job_id = _job_id(_sbatch(worker_arguments))
    finalizer_arguments = _finalizer_command(
        run_root, Path(args.repository_root).resolve(), worker_job_id,
    )
    try:
        finalizer_job_id = _job_id(_sbatch(finalizer_arguments))
    except Exception as exc:
        print(json.dumps({
            "status": "PARTIAL_SUBMISSION",
            "worker_job_id": worker_job_id,
            "finalizer_job_id": None,
            "disposition":
                "WORKER_ARRAY_SUBMITTED_FINALIZER_NOT_SUBMITTED",
            "recovery": "RESEARCHER_DIRECTED_ONLY",
            "automatic_retry": False,
            "automatic_cancel": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }, sort_keys=True), file=sys.stderr, flush=True)
        raise SystemExit(1)
    print(json.dumps({
        "submission": True,
        "worker_job_id": worker_job_id,
        "finalizer_job_id": finalizer_job_id,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
