#!/usr/bin/env python3
"""Governed Paper 1 Revision-2026 RBM production-sampling entry point.

One invocation validates one explicit dataset/seed/attempt/class/sample scope,
an execution-specific authorization record, the exact source commit, the
governed checkpoint pair and the actual Slurm allocation, then delegates once
to the accepted fixed finite-step production sampler.

When ``--publish-pool-dir`` is supplied, the same authorized execution result
is additionally published as one complete, independently verified class pool.
Without that option the historical sampling-only behavior is unchanged.

Compute-node only. This entry point does not train, select checkpoints or
enable runtime execution. It performs no retry, resume or requeue logic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_entry_module():
    from trustforge.models.paper01_revision_2026 import (
        restricted_boltzmann_production_entry as entry,
    )

    return entry


def make_parser() -> argparse.ArgumentParser:
    """Build the explicit-identity production argument parser."""

    entry = _load_entry_module()

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--dataset-id",
        required=True,
        choices=tuple(sorted(entry.DATASET_CLASS_COUNTS)),
    )
    parser.add_argument(
        "--experimental-seed",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--attempt",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--class-id",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--sample-start",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--sample-stop",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--expected-source-commit",
        required=True,
    )
    parser.add_argument(
        "--repository-root",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--expected-checkpoint-state-sha256",
        required=True,
    )
    parser.add_argument(
        "--publish-pool-dir",
        default=None,
        type=Path,
        help="optional complete class-pool publication destination",
    )
    parser.add_argument(
        "--expected-checkpoint-metadata-sha256",
        default=None,
        help="required with --publish-pool-dir",
    )
    parser.add_argument(
        "--expected-policy-digest",
        default=None,
        help="required with --publish-pool-dir",
    )

    return parser


def main(argv=None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    entry = _load_entry_module()

    if args.publish_pool_dir is not None:
        missing = [
            option
            for option, value in (
                (
                    "--expected-checkpoint-metadata-sha256",
                    args.expected_checkpoint_metadata_sha256,
                ),
                (
                    "--expected-policy-digest",
                    args.expected_policy_digest,
                ),
            )
            if value is None
        ]
        if missing:
            parser.error(
                "--publish-pool-dir requires " + ", ".join(missing)
            )

    request = entry.RBMProductionSamplingRequest(
        dataset_id=args.dataset_id,
        experimental_seed=args.experimental_seed,
        attempt=args.attempt,
        class_id=args.class_id,
        sample_start=args.sample_start,
        sample_stop=args.sample_stop,
        expected_source_commit=args.expected_source_commit,
        repository_root=args.repository_root,
        checkpoint_directory=args.checkpoint_dir,
        expected_checkpoint_state_sha256=(
            args.expected_checkpoint_state_sha256
        ),
    )

    execution = entry.execute_rbm_production_sampling(request)

    print(
        "RBM_PRODUCTION_SAMPLING_RESULT "
        + json.dumps(
            execution.to_mapping(),
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )

    if args.publish_pool_dir is not None:
        from trustforge.models.paper01_revision_2026 import (
            restricted_boltzmann_class_pool as pool,
        )

        report = pool.publish_rbm_class_pool(
            execution=execution,
            destination=args.publish_pool_dir,
            expected_checkpoint_metadata_sha256=(
                args.expected_checkpoint_metadata_sha256
            ),
            expected_checkpoint_state_sha256=(
                args.expected_checkpoint_state_sha256
            ),
            expected_policy_digest=args.expected_policy_digest,
        )

        print(
            "RBM_CLASS_POOL_PUBLICATION_RESULT "
            + json.dumps(report, sort_keys=True, allow_nan=False),
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
