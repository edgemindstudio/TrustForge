#!/usr/bin/env python3
"""Invoke the governed CIC GAN seed-42 pool publisher once.

Compute-node only. The synthesis grant is read from its fixed external path;
no command-line option can supply an alternative grant, source, checkpoint or
destination. A successful publication is independently preflighted before an
execution record is written.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import socket

from trustforge.evaluation.paper01_gan_pool import GANPoolView
from trustforge.models.paper01_revision_2026.gan_cic_pool_publication import (
    publish_gan_cic_pool,
)
from trustforge.models.paper01_revision_2026.gan_production_authorization import (
    capture_gan_slurm_allocation,
    validate_gan_slurm_allocation,
)
from trustforge.models.paper01_revision_2026.gan_synthesis_authorization import (
    load_gan_synthesis_grant,
    require_gan_synthesis_authorization,
    verify_checkpoint_metadata_digest,
)


EXECUTION_SCHEMA = "trustforge.paper01.gan.cic-pool-execution.v1"
ACCEPTANCE_FAILURE_SCHEMA = (
    "trustforge.paper01.gan.cic-pool-acceptance-failure.v1"
)
LAUNCHER_RELATIVE_PATH = (
    "slurm/paper01_revision_2026/gan_pool_production.slurm"
)
TRAINING_CHECKOUT = (
    Path.home()
    / "trustforge_revision_2026_execution"
    / "paper01"
    / "gan-source-42c40fc"
)
EXECUTION_FILENAME = "execution.json"
ACCEPTANCE_FAILURE_FILENAME = "ACCEPTANCE_FAILURE.json"


def _sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path):
    return _sha256(Path(path).read_bytes())


def require_slurm_context():
    if (
        not os.environ.get("SLURM_JOB_ID", "").strip()
        or not os.environ.get("SLURM_JOB_NODELIST", "").strip()
        or socket.gethostname().split(".")[0].startswith("login")
    ):
        raise RuntimeError("allocated compute node required")


def verify_launcher_sha256(repository_root, expected_sha256):
    path = Path(repository_root) / LAUNCHER_RELATIVE_PATH

    if path.is_symlink() or not path.is_file():
        raise PermissionError("synthesis launcher is unavailable")

    if _sha256_file(path) != expected_sha256:
        raise PermissionError("synthesis launcher digest mismatch")


def _write_json_exclusive(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")


def run(repository_root=None) -> dict:
    repository_root = Path(
        repository_root or Path(__file__).resolve().parents[1]
    ).resolve()

    if repository_root == TRAINING_CHECKOUT.resolve():
        raise RuntimeError(
            "synthesis must use a separate execution checkout"
        )

    require_slurm_context()
    grant = load_gan_synthesis_grant()
    verify_launcher_sha256(
        repository_root,
        grant.launcher_sha256,
    )

    allocation = capture_gan_slurm_allocation()
    validate_gan_slurm_allocation(allocation, grant.resources)

    verify_checkpoint_metadata_digest(
        grant.checkpoint_directory,
        grant.checkpoint_sha256,
    )

    # Re-validate the exact runtime identity before the publisher is called.
    require_gan_synthesis_authorization(
        destination=grant.destination,
        checkpoint_directory=grant.checkpoint_directory,
        dataset_id=grant.dataset_id,
        experimental_seed=grant.experimental_seed,
        repository_root=repository_root,
    )

    manifest_path = None

    try:
        manifest_path = publish_gan_cic_pool(
            grant.destination,
            checkpoint_directory=grant.checkpoint_directory,
            dataset_id=grant.dataset_id,
            experimental_seed=grant.experimental_seed,
            repository_root=repository_root,
        )

        manifest_raw = manifest_path.read_bytes()
        manifest_sha256 = _sha256(manifest_raw)

        view = GANPoolView(
            manifest_path,
            manifest_sha256=manifest_sha256,
            family="gan",
            experimental_seed=grant.experimental_seed,
            recipe_sha256=grant.recipe_sha256,
            checkpoint_sha256=grant.checkpoint_sha256,
        )
        preflight = view.preflight()

        execution = {
            "schema": EXECUTION_SCHEMA,
            "status": "COMPLETE",
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "slurm_job_nodelist": os.environ["SLURM_JOB_NODELIST"],
            "hostname": socket.gethostname(),
            "source_commit": grant.source_commit,
            "authorization_id": grant.authorization_id,
            "run_identity": grant.run_identity,
            "destination": str(grant.destination),
            "checkpoint_run_identity": grant.checkpoint_run_identity,
            "checkpoint_sha256": grant.checkpoint_sha256,
            "manifest_sha256": manifest_sha256,
            "verified_batches": preflight["verified_batches"],
            "verified_images": preflight["verified_images"],
        }
        _write_json_exclusive(
            grant.destination / EXECUTION_FILENAME,
            execution,
        )
    except Exception as exc:
        if grant.destination.is_dir():
            manifest_sha256 = None

            if manifest_path is not None and manifest_path.is_file():
                manifest_sha256 = _sha256_file(manifest_path)

            try:
                _write_json_exclusive(
                    grant.destination / ACCEPTANCE_FAILURE_FILENAME,
                    {
                        "schema": ACCEPTANCE_FAILURE_SCHEMA,
                        "status": "FAILED",
                        "error_type": type(exc).__name__,
                        "manifest_sha256": manifest_sha256,
                    },
                )
            except (FileExistsError, OSError):
                pass

        raise

    print(
        "GAN_POOL_EXECUTION_COMPLETE "
        + json.dumps(execution, sort_keys=True, allow_nan=False),
        flush=True,
    )
    return execution


def main() -> None:
    run()


if __name__ == "__main__":
    main()
